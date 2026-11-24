"""Comparison plots for EfficientNet-B0 vs MobileNetV3-Small on PlantVillage."""
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


def plot_model_comparison(
    eff_history: list[dict[str, Any]],
    mob_history: list[dict[str, Any]],
    eff_summary: dict[str, Any],
    mob_summary: dict[str, Any],
    sample_preds: list[dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """Create a multi-panel comparison: training curves, Pareto efficiency chart, and leaf predictions."""
    configure_plot_style()
    fig = plt.figure(figsize=(15.0, 9.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.28)

    # Panel A: Training Loss & Validation Accuracy curves
    ax_curve = fig.add_subplot(gs[0, :2])
    eff_ep = [h["epoch"] for h in eff_history]
    mob_ep = [h["epoch"] for h in mob_history]

    ax_curve.plot(eff_ep, [h["val_accuracy"] for h in eff_history], marker="o", lw=2.2, color="#38bdf8", label="EfficientNet-B0 Val Acc")
    ax_curve.plot(mob_ep, [h["val_accuracy"] for h in mob_history], marker="s", lw=2.2, color="#34d399", label="MobileNetV3-Small Val Acc")
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Validation Accuracy")
    ax_curve.set_ylim(0.0, 1.05)
    ax_curve.grid(True, alpha=0.35)

    ax_loss = ax_curve.twinx()
    ax_loss.plot(eff_ep, [h["train_loss"] for h in eff_history], linestyle="--", alpha=0.75, color="#7dd3fc", label="EffNet-B0 Loss")
    ax_loss.plot(mob_ep, [h["train_loss"] for h in mob_history], linestyle="--", alpha=0.75, color="#6ee7b7", label="MobileNetV3 Loss")
    ax_loss.set_ylabel("Training Loss", color="#cbd5e1")
    ax_loss.tick_params(axis="y", colors="#cbd5e1")
    ax_curve.set_title("Learning Curves: Accuracy & Loss")
    lines1, lbl1 = ax_curve.get_legend_handles_labels()
    lines2, lbl2 = ax_loss.get_legend_handles_labels()
    ax_curve.legend(lines1 + lines2, lbl1 + lbl2, loc="lower right", fontsize=8)

    # Panel B: Pareto Efficiency (Accuracy vs Inference Latency, sized by Weight MB)
    ax_pareto = fig.add_subplot(gs[0, 2:])
    models = [
        ("EfficientNet-B0", eff_summary, "#38bdf8"),
        ("MobileNetV3-Small", mob_summary, "#34d399"),
    ]
    for name, info, color in models:
        ms = float(info["test_metrics"]["inference_ms_per_image"])
        acc = float(info["test_metrics"]["accuracy"]) * 100.0
        size_mb = max(float(info["weight_size_mb"]), 1.0)
        params_m = float(info["total_params"]) / 1e6
        ax_pareto.scatter([ms], [acc], s=size_mb * 45.0, color=color, alpha=0.82, edgecolors="#f8fafc", linewidths=1.5)
        ax_pareto.annotate(
            f"{name}\nAcc: {acc:.1f}% | {ms:.2f} ms/img\n{params_m:.2f}M params ({size_mb:.1f} MB)",
            xy=(ms, acc),
            xytext=(12, -16),
            textcoords="offset points",
            fontsize=8.8,
            color="#f8fafc",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", edgecolor=color, alpha=0.9),
        )
    ax_pareto.set_xlabel("Inference Latency (ms / image, lower is faster)")
    ax_pareto.set_ylabel("Held-Out Test Accuracy (%)")
    ax_pareto.set_title("Accuracy vs Speed Pareto Frontier (Bubble = Weight MB)")
    ax_pareto.grid(True, alpha=0.35)
    ax_pareto.margins(x=0.35, y=0.25)

    # Panel C: 4 Real Held-Out Test Leaf Predictions
    for col in range(4):
        ax_img = fig.add_subplot(gs[1, col])
        if col < len(sample_preds):
            s = sample_preds[col]
            ax_img.imshow(s["rgb"])
            ok = s["true_label"] == s["eff_pred"]
            color = "#34d399" if ok else "#f87171"
            ax_img.set_title(
                f"True: {s['true_label'][:22]}\nEffNet: {s['eff_pred'][:20]} ({s['eff_conf']*100:.0f}%)",
                fontsize=8.5,
                color=color,
            )
            ax_img.set_xlabel(f"MobNet: {s['mob_pred'][:20]} ({s['mob_conf']*100:.0f}%)", fontsize=8)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

    fig.suptitle(
        "PlantVillage Leaf Disease Transfer Learning: EfficientNet-B0 vs MobileNetV3-Small",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    return dest
