# 09 - Geospatial Landslide Susceptibility & AI-SDSS (`Landslide4Sense`)

Multi-band satellite landslide segmentation and operational **AI Spatial Decision Support System (AI-SDSS)** built on real Earth-observation tiles from HuggingFace **`ibm-nasa-geospatial/Landslide4sense`**.

## Pipeline & Architecture

1. **Resilient Multi-Band Loader (`src/dataset.py`)**:
   - Streams `ibm-nasa-geospatial/Landslide4sense` tiles, inspects raw image/mask schema dynamically (seamlessly handling 14-band Sentinel-2/DEM arrays in `(H,W,C)` or `(C,H,W)` as well as RGB PIL images), balances positive and negative tiles, standardizes channels, and caches to `data/landslide_cache.npz`.
2. **Segmentation Model (`src/model.py`)**:
   - `SpatialResUNet`: Dynamic `in_channels` Residual U-Net with skip connections trained under `FocalDiceLoss` (class-weighted Binary Focal + Soft Dice Loss) and Turing-safe FP16 AMP.
3. **AI-SDSS Decision Engine (`src/sdss_engine.py`) & FastAPI (`src/api.py`)**:
   - Classifies pixel susceptibility probabilities into 4 operational disaster-risk tiers:
     - `Low (< 0.20)` -> `Normal`
     - `Moderate (0.20 - 0.50)` -> `Monitor`
     - `High (0.50 - 0.75)` -> `Early Warning`
     - `Critical (>= 0.75)` -> `Evacuate`
   - Extracts connected High/Critical zones into a valid **GeoJSON FeatureCollection** with WGS84 bounding polygons, area (`ha`), max/mean probability, and mitigation advisory.
   - Serves `GET /health` and `POST /assess-tile` via FastAPI.

## Reproduction

```bash
# Run offline CPU unit tests (including FastAPI TestClient)
pytest tests/

# Train SpatialResUNet and generate AI-SDSS report + map
python train.py --epochs 15 --batch-size 16 --seed 42

# Optional: run AI-SDSS REST API
uvicorn src.api:app --host 127.0.0.1 --port 8009
```

## Outputs

- `reports/metrics.json` -- Pixel ROC-AUC, PR-AUC, F1, IoU, Landslide Recall, Positive Tile Detection Accuracy, sample GeoJSON assessment, and peak VRAM telemetry.
- `reports/landslide_sdss_map.png` -- 4-panel geospatial strip (Satellite RGB/False-Color, GT Landslide Mask, Predicted Susceptibility, 4-Tier AI-SDSS Zones) plus learning and ROC/PR curves.
- `artifacts/resunet_landslide_best.pt` -- Best validation IoU checkpoint (gitignored).
