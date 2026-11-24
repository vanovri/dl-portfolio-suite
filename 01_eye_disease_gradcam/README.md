# 01 - Eye Disease Grading & Grad-CAM Explainability

Clinical retinal fundus Diabetic Retinopathy (DR) grading with ResNet-18 transfer learning and forward/backward-hook Grad-CAM visual attribution.

## Architecture & Methodology
- **Backbone**: `torchvision.models.resnet18` (ImageNet pretrained) with a regularized `Dropout(0.3) -> Linear(512, 5)` head.
- **Imbalance Handling**: Smoothed inverse-frequency class-weighted `CrossEntropyLoss` + cosine LR decay (`AdamW`, `lr=3e-4`, FP16 AMP on CUDA).
- **Explainability**: `src/gradcam.py` registers forward and full-backward hooks on `layer4[-1]` to produce normalized `[0, 1]` activation maps blended onto RGB fundus images.
- **Metrics**: Test Accuracy, Macro-F1, Quadratic Weighted Kappa (QWK), One-vs-Rest Macro ROC-AUC, inference latency (ms/image), and peak VRAM.

## Dataset Provenance
- **Primary Real Dataset**: MedMNIST v2 `RetinaMNIST` (128x128 RGB fundus images, 5 ordinal DR severity grades `0..4`).
  - Source URL: `https://zenodo.org/records/10519652/files/retinamnist_128.npz`
  - Auto-downloaded and cached in `data/retinamnist_128.npz`.
- **Drop-In Extension**: `src.data.load_eye_disease_imagefolder` supports 4-class Kaggle/GitHub eye disease folder datasets (`cataract`, `diabetic_retinopathy`, `glaucoma`, `normal`; e.g., `dlzcods/eye-disease-classification`).

## Reproduction
```bash
# Run offline CPU unit tests
pytest tests/

# Train and evaluate on GPU (or CPU with --cpu)
python train.py --epochs 12 --batch-size 32 --seed 42
```

## Outputs
- `artifacts/resnet18_retinamnist.pt` - Best validation checkpoint.
- `reports/metrics.json` - Quantitative test metrics and VRAM telemetry.
- `reports/gradcam_gallery.png` - Held-out fundus samples with Grad-CAM overlays.
- `reports/confusion_roc.png` - Confusion matrix and multi-class OvR ROC curves.
