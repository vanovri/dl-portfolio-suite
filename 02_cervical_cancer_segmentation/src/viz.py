"""Visualizations for Plain vs Attention U-Net nuclei ablation and cervical biopsy N/C overlay."""
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


def overlay_mask_contours(
    rgb: np.ndarray,
    binary_mask: np.ndarray,
    fill_color: tuple[int, int, int] = (56, 189, 248),
    contour_color: tuple[int, int, int] = (250, 204, 21),
    alpha: float = 0.35,
) -> np.ndarray:
    """Blend semi-transparent fill and crisp contours for a binary segmentation mask."""
    out = rgb.astype(np.float32).copy()
    m = (binary_mask > 0).astype(np.uint8)
    if m.shape[:2] != rgb.shape[:2]:
        m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    color_arr = np.array(fill_color, dtype=np.float32)
    out[m > 0] = (1.0 - alpha) * out[m > 0] + alpha * color_arr
    out_u8 = np.clip(np.round(out), 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out_u8, contours, -1, contour_color, 1)
    return out_u8


def plot_segmentation_overlay(
    dsb_samples: list[dict[str, Any]],
    biopsy_samples: list[dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """Create a 4x4 panel: 3 rows of DSB2018 ablation + 1 row of Cervical Biopsy N/C morphometry."""
    configure_plot_style()
    n_dsb = min(3, len(dsb_samples))
    n_bio = min(4, len(biopsy_samples))
    rows = n_dsb + (1 if n_bio > 0 else 0)
    if rows == 0:
        raise ValueError("Need at least one DSB or biopsy sample to plot")

    fig, axes = plt.subplots(rows, 4, figsize=(14.5, 3.5 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Original Microscopy", "Ground-Truth Nuclei", "Plain U-Net Pred", "Attention U-Net Pred"]
    for r in range(n_dsb):
        s = dsb_samples[r]
        rgb = s["rgb"]
        gt = s["gt_mask"]
        plain = s["plain_pred"]
        att = s["att_pred"]

        axes[r, 0].imshow(rgb)
        axes[r, 1].imshow(overlay_mask_contours(rgb, gt, (52, 211, 153), (16, 185, 129)))
        axes[r, 2].imshow(overlay_mask_contours(rgb, plain, (248, 113, 113), (251, 191, 36)))
        axes[r, 3].imshow(overlay_mask_contours(rgb, att, (56, 189, 248), (250, 204, 21)))

        if r == 0:
            for c, title in enumerate(col_titles):
                axes[r, c].set_title(title, fontsize=10.5)
        else:
            axes[r, 2].set_title(f"Plain Dice={s.get('plain_dice', 0.0):.3f}", fontsize=9.5)
            axes[r, 3].set_title(f"Attn Dice={s.get('att_dice', 0.0):.3f}", fontsize=9.5)
        if r == 0:
            axes[r, 2].set_title(f"Plain U-Net (Dice={s.get('plain_dice', 0.0):.3f})", fontsize=10)
            axes[r, 3].set_title(f"Attention U-Net (Dice={s.get('att_dice', 0.0):.3f})", fontsize=10)

        for c in range(4):
            axes[r, c].axis("off")

    if n_bio > 0:
        br = rows - 1
        for c in range(4):
            if c < n_bio:
                b = biopsy_samples[c]
                ov = overlay_mask_contours(b["rgb"], b["mask"], (167, 139, 250), (250, 204, 21), alpha=0.4)
                axes[br, c].imshow(ov)
                lbl = str(b.get("label", "biopsy"))[:18]
                axes[br, c].set_title(
                    f"Biopsy [{lbl}]\nN/C={b['nc_ratio']:.3f} | Nuclei={b['nuclei_count']}",
                    fontsize=9.2,
                    color="#c4b5fd",
                )
            axes[br, c].axis("off")

    fig.suptitle(
        "Nuclei Segmentation Ablation (Plain vs Attention U-Net) & Cervical Biopsy N/C Morphometry",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    return dest
