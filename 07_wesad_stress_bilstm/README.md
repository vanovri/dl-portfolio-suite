# 07 - WESAD Physiological Stress Detection (DSP + Bi-LSTM Temporal Attention)

Subject-independent physiological stress detection on real wearable sensor streams from **WESAD** (`LouisSimon/wesad-parquet`, 132,524 rows across wearable ACC, BVP, EDA, and Skin Temperature channels).

## Pipeline & Architecture

1. **Zero-Phase Butterworth DSP (`src/dsp.py`)**:
   - 2nd-order SOS zero-phase Butterworth bandpass filtering on BVP (`scipy.signal.butter(..., output="sos")` + `sosfiltfilt`).
   - Low-pass tonic/phasic EDA decomposition (SCL baseline + SCR phasic response) and temporal gradient.
   - 3-axis accelerometer magnitude `sqrt(acc_x^2 + acc_y^2 + acc_z^2)` and low-pass filtered skin temperature.
   - Strict **per-subject z-score normalization** and sliding window extraction (`window_len=64`, `stride=32`) grouped strictly by `user` ID.
2. **Zero-Leakage Evaluation Split**:
   - Deterministic group split by `user` (~70% train / ~15% val / ~15% test users). No subject ever appears across train, validation, and test splits.
3. **Model (`src/model.py`)**:
   - `BiLSTMAttentionStressNet`: 2-layer Conv1D temporal stem -> 2-layer Bidirectional LSTM -> Additive Temporal Attention pooling returning `(logits, attention_weights)` -> Classifier head.
   - Benchmarked against classical `LogisticRegression` and `RandomForestClassifier` trained on per-window summary features (`mean, std, min, max, slope`) over the identical subject-independent split.

## Reproduction

```bash
# Run offline CPU unit tests
pytest tests/

# Train and evaluate on real WESAD dataset (GTX 1650 Ti 4 GB safe)
python train.py --epochs 15 --batch-size 64 --seed 42
```

## Outputs

- `reports/metrics.json` -- Subject-independent Test Accuracy, Macro-F1, ROC-AUC, PR-AUC, Bi-LSTM vs Classical delta, and peak VRAM telemetry.
- `reports/stress_signals_attention.png` -- Physiological waveforms (BVP, EDA, Temp) with temporal attention heatmap overlay, learning curves, ROC comparison, and held-out confusion matrix.
- `artifacts/bilstm_stress_best.pt` -- Best validation checkpoint (gitignored).
