# 05 - Indonesian Car Classifier & Used-Car Marketplace API

End-to-end vehicle recognition pipeline combining an EfficientNet-B0 classifier trained on real Stanford Cars classes with an Indonesian IDR used-car valuation engine and FastAPI serving layer.

## Architecture & Methodology
- **Classifier (`src/model.py`)**: ImageNet-pretrained `EfficientNet-B0` fine-tuned on 10 car classes at `160x160` RGB (`AdamW`, `lr=4e-4`, label smoothing `0.05`, cosine LR, FP16 AMP).
- **Indonesian Valuation Engine (`src/marketplace.py`)**: Maps predicted car classes and body segments (`Sedan`, `SUV`, `Hatchback`, `Coupe`, `MPV`) to Indonesian IDR OTR/used bands with compound age depreciation, odometer adjustment, condition multiplier, and searchable listings across Jakarta Selatan, Surabaya, Bandung, and Semarang.
- **FastAPI Serving (`src/api.py`)**:
  - `GET /health` - Service & model readiness.
  - `GET /catalog` - Browse & filter marketplace inventory by city or brand.
  - `POST /predict` - Accepts image upload (raw `image/*`, `multipart/form-data`, or JSON base64) and returns Top-3 models, confidence, IDR valuation band, and matching listings.

## Dataset Provenance
- **Real Dataset**: HuggingFace `tanganke/stanford_cars`, streamed and cached locally under `data/cars_cache/` selecting 10 well-represented classes prioritizing Indonesian-popular brands (Toyota, Honda, Suzuki, Hyundai, BMW, Mercedes-Benz, Audi, Nissan, Mitsubishi, Ford, Chevrolet).

## Reproduction
```bash
# Run offline CPU unit tests (includes FastAPI TestClient routes)
pytest tests/

# Train classifier, benchmark FastAPI p50/p95 latency, and build gallery
python train.py --epochs 10 --batch-size 24 --seed 42
```

## Outputs
- `artifacts/efficientnet_b0_indo_cars.pt` - Trained vehicle classifier checkpoint.
- `reports/metrics.json` - Top-1/Top-3 accuracy, Macro-F1, API p50/p95 latency, peak VRAM.
- `reports/car_predictions_gallery.png` - 2x4 held-out car predictions with IDR valuation bands.
