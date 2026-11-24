"""Gallery visualization for Indonesian car classification & IDR marketplace valuation."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.cuda_utils import configure_plot_style


def plot_car_predictions_gallery(samples: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Plot a 2x4 gallery of real held-out car photos with prediction, confidence, and IDR range."""
    configure_plot_style()
    n = min(8, len(samples))
    if n == 0:
        raise ValueError("samples must contain at least 1 car prediction")

    rows = 2 if n > 4 else 1
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(15.5, 4.1 * rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes_flat):
        if i < n:
            s = samples[i]
            ax.imshow(s["rgb"])
            correct = s["true_label"] == s["pred_label"]
            accent = "#34d399" if correct else "#f87171"
            short_pred = s["pred_label"][:26]
            short_true = s["true_label"][:26]
            ax.set_title(
                f"Pred: {short_pred}\nConf: {s['confidence'] * 100:.1f}% | {s['idr_range']}",
                fontsize=8.6,
                color=accent,
            )
            ax.set_xlabel(f"True: {short_true}", fontsize=8.0, color="#cbd5e1")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Indonesian Car Classifier & Marketplace Valuation (Held-Out Stanford Cars Test Set)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    return dest
