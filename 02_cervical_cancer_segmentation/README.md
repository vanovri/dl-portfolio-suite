# 02 - Cervical Cancer & Nuclei Segmentation (Plain vs Attention U-Net)

Controlled head-to-head ablation comparing Plain U-Net against Oktay-style Additive Attention U-Net on microscopy nuclei segmentation, followed by downstream Nucleus-to-Cytoplasm (N/C) morphometry on real cervical cancer biopsy slides.

## Architecture & Methodology
- **Plain U-Net (`src/model.py`)**: 4-stage encoder-decoder with direct skip concatenation (`base_channels=32`).
- **Attention U-Net (`src/model.py`)**: Identical encoder-decoder capacity augmented with Additive Attention Gates (`W_g`, `W_x`, `psi`) on all 4 skip connections.
- **Hybrid Loss (`DiceBCEBoundaryLoss`)**: `0.4 * BCEWithLogits + 0.4 * SoftDice + 0.2 * SobelBoundaryL1`, computed in FP32 inside FP16 AMP training for Turing SM 7.5 stability.
- **Metrics**: Dice, IoU, Precision, Recall, Boundary F1 (2 px tolerance), plus downstream biopsy N/C area ratio and connected-component nuclei count.

## Dataset Provenance
1. **Primary Pixel Segmentation**: HuggingFace `DataTorchio/nuclei-segmentation-dsb2018` (`annotations.json` + `images/`), cached locally under `data/dsb2018` and split deterministically 80/10/10 at 128x128.
2. **Downstream Cervical Biopsy Morphometry**: HuggingFace `MakAIHealthLab/cervical-cancer-biopsy-dataset` (40 streamed real H&E cervical biopsy slides).

## Reproduction
```bash
# Run offline CPU unit tests
pytest tests/

# Train Plain U-Net + Attention U-Net ablation and run cervical biopsy N/C analysis
python train.py --epochs 15 --batch-size 16 --seed 42
```

## Outputs
- `artifacts/plain_unet_nuclei.pt` & `artifacts/attention_unet_nuclei.pt`
- `reports/metrics.json` - Ablation table, delta metrics, biopsy N/C morphometry, VRAM stats.
- `reports/segmentation_overlay.png` - DSB2018 side-by-side predictions + Cervical Biopsy N/C panel.
