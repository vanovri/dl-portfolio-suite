"""Publication-grade visual proof figure for Project 11 multimodal image + audio fusion."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def generate_multimodal_ablation_figure(
    output_path: Path,
    sample_images: np.ndarray,
    sample_specs: np.ndarray,
    sample_labels: list[str],
    sample_alphas: list[float],
    gate_distributions: dict[str, list[float]],
    ablation_results: dict[str, dict[str, dict[str, float]]],
) -> Path:
    """Render 4-panel proof figure: paired RGB + Log-Mel + gate alpha + robustness bars."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15.5, 9.8))
    outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.25)

    # Panel A (Top-Left): Sample paired RGB images with predicted class & visual gate alpha
    n_show = min(4, len(sample_images))
    sub_img = gridspec.GridSpecFromSubplotSpec(1, max(1, n_show), subplot_spec=outer[0, 0], wspace=0.14)
    for i in range(n_show):
        ax = fig.add_subplot(sub_img[0, i])
        img = sample_images[i]
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 1.0)
        ax.imshow(img)
        lbl = sample_labels[i] if i < len(sample_labels) else f"cls_{i}"
        alp = sample_alphas[i] if i < len(sample_alphas) else 0.5
        ax.set_title(f"{lbl}\nalpha={alp:.2f}", fontsize=9.5, fontweight="bold")
        ax.axis("off")

    # Add shared header box for Panel A
    ax_a_title = fig.add_subplot(outer[0, 0])
    ax_a_title.axis("off")
    ax_a_title.set_title(
        "A. Held-Out Paired CIFAR-10 RGB Samples & Learned Visual Gate (alpha)",
        fontsize=11,
        fontweight="bold",
        pad=26,
    )

    # Panel B (Top-Right): Corresponding ESC-50 Log-Mel Spectrograms (64 x 96)
    sub_spec = gridspec.GridSpecFromSubplotSpec(1, max(1, n_show), subplot_spec=outer[0, 1], wspace=0.18)
    for i in range(n_show):
        ax = fig.add_subplot(sub_spec[0, i])
        spec = sample_specs[i]
        if spec.ndim == 3:
            spec = spec[0]
        ax.imshow(spec, origin="lower", aspect="auto", cmap="magma")
        lbl = sample_labels[i] if i < len(sample_labels) else f"cls_{i}"
        ax.set_title(f"{lbl} Mel", fontsize=9.5, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Mel Bins (64)", fontsize=8.5)
        ax.set_xlabel("Frames (96)", fontsize=8)

    ax_b_title = fig.add_subplot(outer[0, 1])
    ax_b_title.axis("off")
    ax_b_title.set_title(
        "B. Paired ESC-50 Log-Mel Spectrograms (16 kHz, 64x96)",
        fontsize=11,
        fontweight="bold",
        pad=26,
    )

    # Panel C (Bottom-Left): Learned Modality Gate alpha Adaptation Across Conditions
    ax_gate = fig.add_subplot(outer[1, 0])
    cond_order = [
        ("clean", "Clean Pairs", "#38bdf8"),
        ("visual_corrupt", "Visual Corrupted\n(trusts Audio -> lower alpha)", "#f43f5e"),
        ("audio_corrupt", "Audio Corrupted\n(trusts Image -> higher alpha)", "#22c55e"),
    ]
    positions = np.arange(len(cond_order))
    means = []
    stds = []
    colors = []
    xlabels = []
    for key, label, color in cond_order:
        vals = np.asarray(gate_distributions.get(key, [0.5]), dtype=np.float32)
        means.append(float(vals.mean()) if vals.size else 0.5)
        stds.append(float(vals.std()) if vals.size else 0.0)
        colors.append(color)
        xlabels.append(label)

    bars = ax_gate.bar(positions, means, yerr=stds, capsize=5, width=0.52, color=colors, alpha=0.88, edgecolor="#e2e8f0")
    ax_gate.axhline(0.5, color="#94a3b8", linestyle="--", linewidth=1.1, label="Equal Weight (0.50)")
    ax_gate.set_ylim(0.0, 1.08)
    ax_gate.set_xticks(positions)
    ax_gate.set_xticklabels(xlabels, fontsize=9)
    ax_gate.set_ylabel("Mean Visual Gate alpha in (0, 1)")
    ax_gate.set_title("C. Gated Cross-Attention Adaptive Reliability Shift", fontsize=11, fontweight="bold")
    ax_gate.grid(axis="y", alpha=0.35)
    for bar, m in zip(bars, means):
        ax_gate.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(1.02, m + 0.05),
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#f8fafc",
        )
    ax_gate.legend(loc="upper right", fontsize=8.5)

    # Panel D (Bottom-Right): 4-Way Clean vs Corrupted Robustness Bar Chart
    ax_bar = fig.add_subplot(outer[1, 1])
    variant_specs = [
        ("image_only", "Image Only", "#64748b"),
        ("audio_only", "Audio Only", "#f59e0b"),
        ("early_concat", "Early Concat", "#a855f7"),
        ("gated_cross_attention", "Gated Cross-Attn", "#10b981"),
    ]
    eval_modes = [
        ("clean", "Clean"),
        ("visual_corrupt", "Visual Corrupt"),
        ("audio_corrupt", "Audio Corrupt"),
        ("mixed_corrupt", "Mixed Corrupt"),
    ]
    x = np.arange(len(eval_modes))
    width = 0.19
    for idx, (vkey, vlabel, vcolor) in enumerate(variant_specs):
        scores = [
            100.0 * float(ablation_results.get(vkey, {}).get(mkey, {}).get("accuracy", 0.0))
            for mkey, _ in eval_modes
        ]
        offset = (idx - 1.5) * width
        rects = ax_bar.bar(x + offset, scores, width=width, label=vlabel, color=vcolor, alpha=0.9)
        for rect, sc in zip(rects, scores):
            ax_bar.text(
                rect.get_x() + rect.get_width() / 2.0,
                sc + 1.2,
                f"{sc:.0f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#e2e8f0",
            )

    ax_bar.set_ylim(0, 112)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([mlabel for _, mlabel in eval_modes], fontsize=9.2)
    ax_bar.set_ylabel("Held-Out Test Accuracy (%)")
    ax_bar.set_title("D. 4-Way Ablation: Clean vs Corrupted-Modality Robustness", fontsize=11, fontweight="bold")
    ax_bar.grid(axis="y", alpha=0.35)
    ax_bar.legend(loc="upper right", ncol=2, fontsize=8.3)

    fig.suptitle(
        "Project 11: Multimodal Image (CIFAR-10) + Audio (ESC-50) Gated Cross-Attention Fusion",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
