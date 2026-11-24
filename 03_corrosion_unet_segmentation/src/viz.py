"""Visualization utilities for metal corrosion segmentation and severity overlays."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.cuda_utils import configure_plot_style

GRADE_COLORS = {
    "Clean": "#34d399",
    "Light": "#38bdf8",
    "Moderate": "#fbbf24",
    "Severe": "#f87171",
}


def make_corrosion_overlay(rgb: np.ndarray, pred_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend predicted corrosion regions in rust-orange with contour outlines."""
    out = rgb.astype(np.float32).copy()
    m = (pred_mask > 0).astype(np.uint8)
    rust_rgb = np.array([249, 115, 22], dtype=np.float32)
    out[m > 0] = (1.0 - alpha) * out[m > 0] + alpha * rust_rgb
    out_u8 = np.clip(np.round(out), 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out_u8, contours, -1, (254, 240, 138), 1)
    return out_u8


def plot_corrosion_predictions(samples: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Plot 4 metal samples across 4 columns: RGB | GT Rust | Pred Rust | Severity Overlay."""
    configure_plot_style()
    n = min(4, len(samples))
    if n == 0:
        raise ValueError("Need at least one corrosion sample to plot")

    fig, axes = plt.subplots(n, 4, figsize=(14.5, 3.4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    headers = ["RGB Metal Surface", "GT Corrosion Mask", "Predicted Rust Mask", "ISO 4628-3 Severity Overlay"]
    for r in range(n):
        s = samples[r]
        rgb = s["rgb"]
        gt = s["gt_mask"]
        pred = s["pred_mask"]
        gt_pct = float(s["gt_area_pct"])
        pred_pct = float(s["pred_area_pct"])
        gt_grade = str(s["gt_grade"])
        pred_grade = str(s["pred_grade"])

        axes[r, 0].imshow(rgb)
        axes[r, 1].imshow(gt, cmap="inferno", vmin=0, vmax=1)
        axes[r, 2].imshow(pred, cmap="inferno", vmin=0, vmax=1)
        axes[r, 3].imshow(make_corrosion_overlay(rgb, pred))

        if r == 0:
            for c, title in enumerate(headers):
                axes[r, c].set_title(title, fontsize=10.5)
        axes[r, 1].set_xlabel(f"GT: {gt_pct:.1f}% ({gt_grade})", fontsize=9)
        axes[r, 2].set_xlabel(f"Pred: {pred_pct:.1f}% ({pred_grade})", fontsize=9)
        axes[r, 3].set_xlabel(
            f"Grade: {pred_grade} | Area: {pred_pct:.1f}%",
            fontsize=9.5,
            color=GRADE_COLORS.get(pred_grade, "#f8fafc"),
            fontweight="bold",
        )
        for c in range(4):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.suptitle("ResNet-18 U-Net Metal Corrosion Segmentation & Severity Grading", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    return dest
