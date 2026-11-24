"""Deterministic offline CPU unit tests for 07_wesad_stress_bilstm."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(PROJECT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.dataset import WESADWindowDataset
from src.dsp import (
    FEATURE_NAMES,
    build_subject_windows,
    butter_bandpass_filter,
    butter_lowpass_filter,
    compute_acc_magnitude,
    decompose_eda,
    extract_window_summary_features,
    normalize_stress_labels,
    split_by_subject,
)
from src.model import (
    BiLSTMAttentionStressNet,
    compute_binary_classification_metrics,
    fit_and_eval_classical_baselines,
)
from src.viz import plot_stress_signals_attention


def _make_synthetic_wesad_df(n_subjects: int = 6, rows_per_subject: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_subjects):
        t = np.arange(rows_per_subject, dtype=np.float32) / 4.0
        stress = np.zeros(rows_per_subject, dtype=np.int64)
        stress[rows_per_subject // 2 :] = 1

        bvp = np.sin(2.0 * np.pi * (1.1 + 0.35 * stress) * t) + 0.1 * rng.standard_normal(rows_per_subject)
        eda = 1.5 + 0.8 * stress + 0.15 * np.sin(2.0 * np.pi * 0.25 * t) * stress + 0.05 * rng.standard_normal(rows_per_subject)
        temp = 33.5 - 0.6 * stress + 0.05 * rng.standard_normal(rows_per_subject)
        acc_x = 0.2 * rng.standard_normal(rows_per_subject)
        acc_y = 0.2 * rng.standard_normal(rows_per_subject)
        acc_z = 9.8 + 0.2 * rng.standard_normal(rows_per_subject)

        frames.append(
            pd.DataFrame(
                {
                    "acc_x": acc_x,
                    "acc_y": acc_y,
                    "acc_z": acc_z,
                    "bvp": bvp,
                    "eda": eda,
                    "temp": temp,
                    "stress": stress,
                    "user": f"S{s + 2}",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_butterworth_bandpass_preserves_heart_rate_and_removes_dc():
    fs = 4.0
    t = np.arange(400) / fs
    sig = 50.0 + np.sin(2.0 * np.pi * 1.0 * t)
    filtered = butter_bandpass_filter(sig, fs=fs, low_hz=0.5, high_hz=1.8, order=2)
    assert filtered.shape == sig.shape
    assert np.isfinite(filtered).all()
    assert abs(float(np.mean(filtered))) < 0.2
    assert float(np.std(filtered)) > 0.4


def test_butterworth_lowpass_handles_short_signals():
    short = np.array([1.0, 2.0, 3.0, 2.5, 2.0], dtype=np.float32)
    out = butter_lowpass_filter(short, fs=4.0, cutoff_hz=0.2)
    assert out.shape == short.shape
    assert np.isfinite(out).all()


def test_eda_tonic_phasic_decomposition_reconstructs_signal():
    fs = 4.0
    t = np.arange(320) / fs
    eda = 2.0 + 0.01 * t + 0.4 * np.sin(2.0 * np.pi * 0.4 * t)
    tonic, phasic = decompose_eda(eda, fs=fs, tonic_cutoff_hz=0.05)
    np.testing.assert_allclose(tonic + phasic, eda, atol=1e-4)
    assert float(np.std(phasic)) > 0.15


def test_acc_magnitude_exact():
    mag = compute_acc_magnitude(np.array([3.0, 0.0]), np.array([4.0, 0.0]), np.array([0.0, 5.0]))
    np.testing.assert_allclose(mag, np.array([5.0, 5.0], dtype=np.float32), atol=1e-6)


def test_label_normalizer_handles_wesad_multiclass_and_binary():
    y_bin, info_bin = normalize_stress_labels(pd.Series([0, 1, 0, 1]))
    assert info_bin["mapping_rule"] == "binary_0_1_direct"
    assert y_bin.tolist() == [0, 1, 0, 1]

    y_proto, info_proto = normalize_stress_labels(pd.Series([1, 2, 3, 2, 1]))
    assert info_proto["mapping_rule"] == "wesad_protocol_class_2_as_stress"
    assert y_proto.tolist() == [0, 1, 0, 1, 0]


def test_build_subject_windows_shapes_and_zscore():
    df = _make_synthetic_wesad_df(n_subjects=6, rows_per_subject=200)
    bundle = build_subject_windows(df, window_len=64, stride=32, fs=4.0)
    assert bundle.X.ndim == 3
    assert bundle.X.shape[1:] == (64, len(FEATURE_NAMES))
    assert bundle.raw_channels.shape[1:] == (64, 4)
    assert len(bundle.y) == len(bundle.users) == bundle.X.shape[0]
    assert np.isfinite(bundle.X).all()


def test_strict_subject_independent_split_zero_leakage():
    df = _make_synthetic_wesad_df(n_subjects=8, rows_per_subject=200)
    bundle = build_subject_windows(df, window_len=64, stride=32, fs=4.0)
    split = split_by_subject(bundle, train_frac=0.70, val_frac=0.15, seed=42)

    tr_u = set(split.users_train.tolist())
    va_u = set(split.users_val.tolist())
    te_u = set(split.users_test.tolist())
    assert tr_u.isdisjoint(va_u)
    assert tr_u.isdisjoint(te_u)
    assert va_u.isdisjoint(te_u)
    assert len(tr_u) + len(va_u) + len(te_u) == 8


def test_extract_window_summary_features_shape_and_slope():
    x = np.zeros((4, 32, 3), dtype=np.float32)
    x[:, :, 0] = np.linspace(0.0, 2.0, 32, dtype=np.float32)[None, :]
    feats = extract_window_summary_features(x)
    assert feats.shape == (4, 15)
    # Channel 0 slope over [-1, 1] is 1.0
    np.testing.assert_allclose(feats[:, 12], np.ones(4), atol=1e-4)


def test_bilstm_attention_forward_backward_and_attention_simplex():
    torch.manual_seed(0)
    model = BiLSTMAttentionStressNet(in_features=9, conv_channels=24, hidden_size=32, num_layers=2)
    x = torch.randn(5, 64, 9)
    y = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)

    logits, attn = model(x)
    assert logits.shape == (5, 2)
    assert attn.shape == (5, 64)
    torch.testing.assert_close(attn.sum(dim=-1), torch.ones(5), atol=1e-5, rtol=1e-5)
    assert (attn >= 0).all()

    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    grad_norm = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0.0


def test_classical_baselines_fit_and_eval():
    df = _make_synthetic_wesad_df(n_subjects=6, rows_per_subject=220)
    bundle = build_subject_windows(df, window_len=64, stride=32, fs=4.0)
    split = split_by_subject(bundle, seed=42)
    res = fit_and_eval_classical_baselines(split.X_train, split.y_train, split.X_test, split.y_test, seed=42)
    assert res["best_classical_name"] in {"logistic_regression", "random_forest"}
    assert 0.0 <= res["logistic_regression"]["macro_f1"] <= 1.0
    assert 0.0 <= res["random_forest"]["accuracy"] <= 1.0


def test_wesad_dataset_and_metrics_helper():
    ds = WESADWindowDataset(np.zeros((10, 64, 9), dtype=np.float32), np.array([0, 1] * 5))
    assert len(ds) == 10
    x0, y0 = ds[0]
    assert x0.shape == (64, 9)
    assert y0.dtype == torch.long

    m = compute_binary_classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["roc_auc"] == 1.0


def test_plot_stress_signals_attention_writes_png(tmp_path: Path):
    out = tmp_path / "stress_plot.png"
    raw_win = np.random.default_rng(1).standard_normal((64, 4)).astype(np.float32)
    attn = np.full(64, 1.0 / 64.0, dtype=np.float32)
    history = {"train_loss": [0.7, 0.5], "val_loss": [0.68, 0.52], "val_macro_f1": [0.6, 0.78]}
    y_te = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.3, 0.7, 0.9])
    saved = plot_stress_signals_attention(
        out_path=out,
        raw_window=raw_win,
        attn_weights=attn,
        fs=4.0,
        history=history,
        y_test=y_te,
        bilstm_probs=probs,
        lr_probs=probs,
        rf_probs=probs,
        cm=[[2, 0], [0, 2]],
        sample_meta={"user": "S10", "prob": 0.91},
    )
    assert saved.exists() and saved.stat().st_size > 1000
