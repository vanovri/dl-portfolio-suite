# Project 11: Multimodal Image + Audio Fusion with Gated Cross-Attention

## Overview
This project benchmarks multimodal audio-visual fusion under both clean and corrupted-modality conditions using real public environmental sound recordings and real natural RGB images. It implements four controlled architecture variants to prove why adaptive cross-modal gating beats unimodal models and naive concatenation when one sensor stream degrades.

## Dataset Provenance & Zero-Leakage Alignment
- **Audio Modality**: HuggingFace `ashraq/esc50` (ESC-50 environmental sound clips, decoded directly via SciPy/WAV without `torchcodec` or `librosa`, converted to `(1, 64, 96)` standardized Log-Mel spectrograms at 16 kHz).
- **Image Modality**: `torchvision.datasets.CIFAR10` (32x32 natural RGB images).
- **7-Class Semantic Alignment**:
  - `0: dog` (ESC-50 `dog` + CIFAR-10 `dog`)
  - `1: cat` (ESC-50 `cat` + CIFAR-10 `cat`)
  - `2: bird` (ESC-50 `rooster`, `crow`, `chirping_birds` + CIFAR-10 `bird`)
  - `3: frog` (ESC-50 `frog` + CIFAR-10 `frog`)
  - `4: airplane` (ESC-50 `airplane`, `helicopter` + CIFAR-10 `airplane`)
  - `5: vehicle` (ESC-50 `car_horn`, `engine` + CIFAR-10 `automobile`, `truck`)
  - `6: sea_ship` (ESC-50 `sea_waves` + CIFAR-10 `ship`)
- **Strict Zero-Leakage Split**:
  - **Train**: ESC-50 `fold in {1, 2, 3, 4}` paired with `CIFAR-10 train=True` (~1,680 balanced pairs, 240/class, with SpecAugment and random horizontal flip/crop).
  - **Held-Out Test**: ESC-50 `fold == 5` paired with `CIFAR-10 train=False` (~420 pairs, 60/class). Neither the audio clip nor the RGB image appears in training, verified programmatically via `verify_zero_leakage()`.

## Architecture Ablation (`src/model.py`)
1. `ImageOnlyNet`: Residual 2D CNN (`3x32x32 -> 128-d`) baseline.
2. `AudioOnlyNet`: Residual 2D CNN (`1x64x96 -> 128-d`) baseline.
3. `EarlyConcatFusionNet`: Concatenates `[z_img, z_aud]` (`256-d`) into a fusion MLP.
4. `GatedCrossAttentionFusionNet`: Bidirectional Multi-Head Cross-Attention between 16 visual tokens and 24 acoustic time-frequency tokens, paired with a reliability-aware per-sample modality gate `alpha in (0, 1)`, auxiliary unimodal heads, and stochastic training modality dropout.

## Corrupted-Modality Stress Test
All four models are evaluated across four regimes:
- `clean`: Unaltered held-out audio-visual pairs.
- `visual_corrupt`: Heavy blur, noise, and 56% spatial cutout on images (gate `alpha` shifts down toward audio).
- `audio_corrupt`: High noise and 60% time-frequency masking on spectrograms (gate `alpha` shifts up toward vision).
- `mixed_corrupt`: 50% visual corruption / 50% audio corruption per batch.

## Reproduction
```bash
# Offline CPU unit tests
pytest tests/test_multimodal.py -v

# Train all 4 variants on GPU (GTX 1650 Ti 4 GB VRAM safe, FP16 AMP)
python train.py --epochs 15 --batch-size 32 --seed 42
```

## Outputs
- `reports/metrics.json`: Full 4-way clean & corrupted metrics, gate adaptation deltas, leakage audit, and VRAM telemetry.
- `reports/multimodal_ablation.png`: 4-panel visual proof (paired RGB samples, Log-Mel spectrograms, gate adaptation, robustness bar chart).
- `artifacts/gated_cross_attention_best.pt`: Saved `GatedCrossAttentionFusionNet` weights.
