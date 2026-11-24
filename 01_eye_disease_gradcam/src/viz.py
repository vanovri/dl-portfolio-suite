"""Visualization utilities for Eye Disease Grad-CAM gallery and evaluation diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.cuda_utils import configure_plot_style


def plot_gradcam_gallery(
    samples: list[dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """Save a 2x4 gallery showing 4 test fundus images (row 0) and Grad-CAM overlays (row 1)."""
    configure_plot_style()
    n = min(4, len(samples))
    if n == 0:
        raise ValueError("samples must contain at least 1 Grad-CAM entry")

    fig, axes = plt.subplots(2, n, figsize=(3.8 * n, 7.2))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col in range(n):
        item = samples[col]
        rgb = item["rgb"]
        overlay = item["overlay"]
        true_lbl = item["true_label"]
        pred_lbl = item["pred_label"]
        conf = float(item["confidence"])
        correct = true_lbl == pred_lbl
        accent = "#34d399" if correct else "#f87171"

        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"True: {true_lbl}", fontsize=10, color="#f8fafc")
        axes[0, col].axis("off")

        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"Pred: {pred_lbl} ({conf * 100:.1f}%)", fontsize=10, color=accent)
        axes[1, col].axis("off")

    fig.suptitle("RetinaMNIST Held-Out Test Fundus Images & ResNet-18 Grad-CAM", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    return dest


def plot_confusion_and_roc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    out_path: str | Path,
) -> Path:
    """Plot confusion matrix alongside OvR multi-class ROC curves."""
    configure_plot_style()
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    fig, (ax_cm, ax_roc) = plt.subplots(1, 2, figsize=(13.5, 5.5))

    im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    short_names = [f"G{i}" for i in range(num_classes)]
    ax_cm.set_xticks(range(num_classes))
    ax_cm.set_yticks(range(num_classes))
    ax_cm.set_xticklabels(short_names, rotation=30, ha="right")
    ax_cm.set_yticklabels(class_names)
    ax_cm.set_xlabel("Predicted DR Grade")
    ax_cm.set_ylabel("True DR Grade")
    ax_cm.set_title("Test Confusion Matrix")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            color = "#0f172a" if cm[i, j] > thresh else "#f8fafc"
            ax_cm.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color, fontweight="bold")

    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if y_bin.ndim == 1:
        y_bin = np.column_stack([1 - y_bin, y_bin])
    palette = ["#38bdf8", "#a78bfa", "#34d399", "#fbbf24", "#f87171"]
    macro_aucs = []
    for c in range(num_classes):
        if y_bin[:, c].sum() == 0 or y_bin[:, c].sum() == len(y_true):
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        macro_aucs.append(roc_auc)
        ax_roc.plot(fpr, tpr, lw=2.0, color=palette[c % len(palette)], label=f"{class_names[c]} (AUC={roc_auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#64748b", lw=1.2)
    mean_auc = float(np.mean(macro_aucs)) if macro_aucs else 0.0
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.02)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"One-vs-Rest ROC Curves (Macro AUC = {mean_auc:.3f})")
    ax_roc.grid(True, alpha=0.35)
    ax_roc.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    return dest
