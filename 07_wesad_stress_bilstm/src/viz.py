"""Visualization routines for WESAD waveforms, temporal attention, ROC, and confusion matrix."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.cuda_utils import configure_plot_style


def plot_stress_signals_attention(
    out_path: Path,
    raw_window: np.ndarray,
    attn_weights: np.ndarray,
    fs: float,
    history: dict[str, list[float]],
    y_test: np.ndarray,
    bilstm_probs: np.ndarray,
    lr_probs: np.ndarray,
    rf_probs: np.ndarray,
    cm: list[list[int]],
    sample_meta: dict[str, Any] | None = None,
) -> Path:
    """Render 2x2 diagnostic figure: waveforms + attention overlay, learning curves, ROC, and CM."""
    configure_plot_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # Panel 1: Physiological waveforms (BVP, EDA tonic+phasic, Temp) + Temporal Attention heatmap
    ax = axes[0, 0]
    t_len = raw_window.shape[0]
    t_sec = np.arange(t_len, dtype=np.float32) / max(fs, 1e-3)

    def _norm01(v: np.ndarray) -> np.ndarray:
        span = float(np.ptp(v))
        return (v - float(np.min(v))) / (span + 1e-6)

    bvp_n = _norm01(raw_window[:, 0]) * 0.8 + 2.2
    eda_n = _norm01(raw_window[:, 1] + raw_window[:, 2]) * 0.8 + 1.1
    temp_n = _norm01(raw_window[:, 3]) * 0.8 + 0.05

    attn = np.asarray(attn_weights, dtype=np.float32)
    attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    ax.imshow(
        attn_norm[None, :],
        aspect="auto",
        extent=[float(t_sec[0]), float(t_sec[-1]), -0.1, 3.25],
        cmap="magma",
        alpha=0.38,
        origin="lower",
    )
    ax.plot(t_sec, bvp_n, color="#38bdf8", lw=1.6, label="BVP (bandpass)")
    ax.plot(t_sec, eda_n, color="#f59e0b", lw=1.8, label="EDA (tonic+phasic)")
    ax.plot(t_sec, temp_n, color="#34d399", lw=1.6, label="Skin Temp (LP)")
    ax.set_yticks([0.45, 1.5, 2.6])
    ax.set_yticklabels(["Temp", "EDA", "BVP"])
    ax.set_xlabel("Window Time (s)")
    ax.set_ylim(-0.1, 3.25)

    ax_attn = ax.twinx()
    ax_attn.plot(t_sec, attn, color="#f43f5e", lw=1.5, ls="--", label="Attention weight")
    ax_attn.set_ylabel("Temporal Attention", color="#fda4af")
    ax_attn.tick_params(axis="y", colors="#fda4af")

    title_extra = ""
    if sample_meta:
        title_extra = f" (User {sample_meta.get('user', '?')}, P={sample_meta.get('prob', 0.0):.2f})"
    ax.set_title(f"True Stress Window + Attention Overlay{title_extra}")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.3)

    # Panel 2: Training & Validation curves
    ax_hist = axes[0, 1]
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    if len(epochs) > 0:
        ax_hist.plot(epochs, history["train_loss"], color="#38bdf8", lw=2.0, label="Train Loss")
        ax_hist.plot(epochs, history["val_loss"], color="#fb7185", lw=2.0, label="Val Loss")
        ax_hist.set_xlabel("Epoch")
        ax_hist.set_ylabel("Loss")
        ax_f1 = ax_hist.twinx()
        ax_f1.plot(epochs, history["val_macro_f1"], color="#34d399", lw=2.0, ls="--", label="Val Macro-F1")
        ax_f1.set_ylabel("Val Macro-F1", color="#6ee7b7")
        ax_f1.set_ylim(0.0, 1.05)
        ax_f1.tick_params(axis="y", colors="#6ee7b7")
        ax_hist.legend(loc="upper left", fontsize=8)
        ax_f1.legend(loc="lower right", fontsize=8)
    ax_hist.set_title("Bi-LSTM-Attention Learning Dynamics")
    ax_hist.grid(True, alpha=0.35)

    # Panel 3: Subject-Independent ROC comparison
    ax_roc = axes[1, 0]
    if len(np.unique(y_test)) == 2:
        for name, probs, color, ls in [
            ("BiLSTM-Attention", bilstm_probs, "#38bdf8", "-"),
            ("RandomForest (stats)", rf_probs, "#34d399", "--"),
            ("LogReg (stats)", lr_probs, "#fbbf24", ":"),
        ]:
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc_val = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
            ax_roc.plot(fpr, tpr, lw=2.0, color=color, ls=ls, label=f"{name} (AUC={auc_val:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="#64748b", lw=1.2, ls="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Subject-Independent Held-Out ROC")
    ax_roc.legend(loc="lower right", fontsize=8.5)
    ax_roc.grid(True, alpha=0.35)

    # Panel 4: Confusion Matrix
    ax_cm = axes[1, 1]
    cm_arr = np.asarray(cm, dtype=np.int64)
    im = ax_cm.imshow(cm_arr, cmap="Blues")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred Non-Stress", "Pred Stress"])
    ax_cm.set_yticklabels(["True Non-Stress", "True Stress"])
    thresh = cm_arr.max() * 0.5 if cm_arr.size > 0 else 0
    for i in range(2):
        for j in range(2):
            val = int(cm_arr[i, j])
            txt_color = "#0f172a" if val > thresh else "#f8fafc"
            ax_cm.text(j, i, f"{val}", ha="center", va="center", color=txt_color, fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    ax_cm.set_title("Held-Out Subject Confusion Matrix")

    fig.suptitle("WESAD Physiological Stress Detection (Subject-Independent Split)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
