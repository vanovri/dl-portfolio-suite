"""Geospatial visualization of Landslide4Sense tile, GT mask, susceptibility map, and 4-tier SDSS zones."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.metrics import precision_recall_curve, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.cuda_utils import configure_plot_style
from src.dataset import extract_rgb_preview
from src.sdss_engine import classify_risk_zones


def plot_landslide_sdss_map(
    out_path: Path,
    raw_tile: np.ndarray,
    gt_mask: np.ndarray,
    prob_map: np.ndarray,
    history: dict[str, list[float]],
    all_y_true: np.ndarray,
    all_y_prob: np.ndarray,
    tile_summary: dict[str, Any] | None = None,
) -> Path:
    """Render 4-panel geospatial SDSS strip + bottom row learning & ROC/PR diagnostic curves."""
    configure_plot_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 0.95], hspace=0.34, wspace=0.22)

    rgb = extract_rgb_preview(raw_tile)
    zones = classify_risk_zones(prob_map)

    # Top row: 4 geospatial panels
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(rgb)
    ax0.set_title("Satellite Tile (RGB / False-Color)")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(rgb)
    ax1.imshow(np.ma.masked_where(gt_mask < 0.5, gt_mask), cmap="autumn", alpha=0.72, vmin=0, vmax=1)
    ax1.set_title("Ground Truth Landslide Mask")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    im_prob = ax2.imshow(prob_map, cmap="inferno", vmin=0.0, vmax=1.0)
    ax2.set_title("Predicted Susceptibility Prob.")
    ax2.axis("off")
    fig.colorbar(im_prob, ax=ax2, fraction=0.046, pad=0.03)

    ax3 = fig.add_subplot(gs[0, 3])
    tier_cmap = ListedColormap(["#1e3a5f", "#facc15", "#f97316", "#ef4444"])
    tier_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], tier_cmap.N)
    im_zone = ax3.imshow(zones, cmap=tier_cmap, norm=tier_norm)
    advisory = tile_summary.get("advisory", "") if tile_summary else ""
    tier_str = tile_summary.get("overall_risk_tier", "") if tile_summary else ""
    title_suffix = f" ({tier_str}: {advisory})" if tier_str else ""
    ax3.set_title(f"4-Tier AI-SDSS Zones{title_suffix}")
    ax3.axis("off")
    cbar = fig.colorbar(im_zone, ax=ax3, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.03)
    cbar.ax.set_yticklabels(["Low", "Mod", "High", "Crit"])

    # Bottom left: Training curves
    ax_tr = fig.add_subplot(gs[1, 0:2])
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    if len(epochs) > 0:
        ax_tr.plot(epochs, history["train_loss"], color="#38bdf8", lw=2.0, label="Train FocalDice Loss")
        ax_tr.plot(epochs, history["val_loss"], color="#fb7185", lw=2.0, label="Val FocalDice Loss")
        ax_tr.set_xlabel("Epoch")
        ax_tr.set_ylabel("Loss")
        ax_iou = ax_tr.twinx()
        ax_iou.plot(epochs, history["val_iou"], color="#34d399", lw=2.0, ls="--", label="Val Landslide IoU")
        ax_iou.set_ylabel("Val IoU", color="#6ee7b7")
        ax_iou.set_ylim(0.0, 1.0)
        ax_iou.tick_params(axis="y", colors="#6ee7b7")
        ax_tr.legend(loc="upper left", fontsize=8.5)
        ax_iou.legend(loc="lower right", fontsize=8.5)
    ax_tr.set_title("SpatialResUNet Training Dynamics")
    ax_tr.grid(True, alpha=0.35)

    # Bottom right: Pixel ROC & PR curves
    ax_eval = fig.add_subplot(gs[1, 2:4])
    y_flat = all_y_true.reshape(-1)
    p_flat = all_y_prob.reshape(-1)
    if len(np.unique(y_flat)) == 2:
        # Subsample cleanly if massive pixel array for fast plotting
        if y_flat.size > 200_000:
            step = y_flat.size // 200_000
            y_flat = y_flat[::step]
            p_flat = p_flat[::step]
        fpr, tpr, _ = roc_curve(y_flat, p_flat)
        prec, rec, _ = precision_recall_curve(y_flat, p_flat)
        ax_eval.plot(fpr, tpr, color="#38bdf8", lw=2.0, label="Pixel ROC (TPR vs FPR)")
        ax_eval.plot(rec, prec, color="#f59e0b", lw=2.0, ls="--", label="Pixel PR (Precision vs Recall)")
    ax_eval.plot([0, 1], [0, 1], color="#64748b", lw=1.0, ls=":")
    ax_eval.set_xlabel("FPR / Recall")
    ax_eval.set_ylabel("TPR / Precision")
    ax_eval.set_title("Held-Out Pixel ROC & Precision-Recall Curves")
    ax_eval.legend(loc="lower left", fontsize=8.5)
    ax_eval.grid(True, alpha=0.35)

    fig.suptitle("Landslide4Sense SpatialResUNet + Geospatial AI-SDSS", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
