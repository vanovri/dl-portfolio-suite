# 04 - Plant Leaf Disease Transfer Learning (EfficientNet-B0 vs MobileNetV3-Small)

Controlled accuracy-vs-efficiency transfer learning benchmark comparing ImageNet-pretrained `EfficientNet-B0` against `MobileNetV3-Small` on a stratified 10-class PlantVillage subset.

## Architecture & Methodology
- **Models (`src/model.py`)**:
  - `efficientnet_b0` (compound-scaled MBConv backbone, higher capacity)
  - `mobilenet_v3_small` (edge-optimized inverted residual backbone, ultra-low latency)
- **Training**: 8 epochs per model, `AdamW(lr=5e-4)`, `CrossEntropyLoss(label_smoothing=0.05)`, cosine annealing, FP16 AMP on CUDA.
- **Metrics**: Test Accuracy, Macro-F1, Top-3 Accuracy, Parameter Count, Checkpoint Size (MB), Inference Latency (ms/img), and per-model Peak VRAM.

## Dataset Provenance
- **Real Dataset**: HuggingFace `BrandonFors/Plant-Diseases-PlantVillage-Dataset` streamed into a local deterministic cache (`data/plantvillage_cache/`): 10 distinct classes, 160 train / 40 val / 50 test images per class (1,600 train / 400 val / 500 test) resized to 128x128 RGB.

## Reproduction
```bash
# Run offline CPU unit tests
pytest tests/

# Run head-to-head training & Pareto efficiency benchmark
python train.py --epochs 8 --batch-size 32 --seed 42
```

## Outputs
- `artifacts/efficientnet_b0_leaf.pt` & `artifacts/mobilenet_v3_small_leaf.pt`
- `reports/metrics.json` - Per-model metrics, speedup ratio, size ratio, and VRAM stats.
- `reports/model_comparison.png` - Learning curves, Pareto frontier, and held-out leaf predictions.
