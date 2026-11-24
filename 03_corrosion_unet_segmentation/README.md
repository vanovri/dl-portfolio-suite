# 03 - Industrial Metal Corrosion Segmentation (ResNet-18 U-Net)

Pixel-level rust detection and ISO 4628-3 style severity grading using a pretrained ResNet-18 U-Net trained on COCO-polygon corrosion annotations.

## Architecture & Methodology
- **Model (`ResNet18UNet`)**: Pretrained `torchvision` ResNet-18 encoder (`conv1`, `layer1..layer4`) coupled to a 4-stage skip-connected U-Net decoder.
- **Loss (`FocalDiceLoss`)**: Binary Focal Loss (`alpha=0.25, gamma=2.0`) + Soft Dice Loss in FP32 inside FP16 AMP training.
- **Severity Grading (`grade_corrosion_severity`)**:
  - `Clean`: `< 1%` corroded pixel area
  - `Light`: `1% - 5%`
  - `Moderate`: `5% - 20%`
  - `Severe`: `>= 20%`
- **Metrics**: Dice, IoU, Precision, Recall, Corrosion Area MAE (percentage points), Severity Grade Accuracy, inference FPS, and peak VRAM.

## Dataset Provenance
- **Real Dataset**: HuggingFace `AiForFly/Rust-segmentation` (`rust.v3i.coco-segmentation/` with `train/`, `valid/`, `test/` COCO polygon annotations).
- Rasterized via `cv2.fillPoly` at `192x192` and cached under `data/`.

## Reproduction
```bash
# Run offline CPU unit tests
pytest tests/

# Train and evaluate ResNet-18 U-Net
python train.py --epochs 20 --batch-size 12 --seed 42
```

## Outputs
- `artifacts/resnet18_unet_corrosion.pt` - Best validation Dice checkpoint.
- `reports/metrics.json` - Pixel metrics, area MAE, severity accuracy, FPS, and VRAM stats.
- `reports/corrosion_predictions.png` - 4 held-out metal samples with GT, predicted masks, and ISO severity overlays.
