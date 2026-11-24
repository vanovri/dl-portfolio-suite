"""Publication-grade 4-panel visualization for Project 13: Classical Literature & Nusantara Pantun."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _plot_token_bars(
    ax: plt.Axes,
    tokens_scores: list[tuple[str, float]],
    title: str,
    color: str,
) -> None:
    if not tokens_scores:
        ax.text(0.5, 0.5, "No salient tokens", ha="center", va="center")
        ax.set_title(title, fontsize=10.2, fontweight="bold")
        return
    top_items = tokens_scores[:10][::-1]
    labels = [t for t, _ in top_items]
    vals = [float(s) for _, s in top_items]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=color, alpha=0.88, edgecolor="#e2e8f0", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean Attention Salience Weight")
    ax.set_title(title, fontsize=10.2, fontweight="bold")
    ax.grid(axis="x", alpha=0.35)


def generate_humanities_stylometry_figure(
    output_path: Path,
    comparison_scores: dict[str, dict[str, float]],
    pantun_roc: dict[str, dict[str, Any]],
    era_top_tokens: dict[str, list[tuple[str, float]]],
    pantun_top_tokens: list[tuple[str, float]],
) -> Path:
    """Render 4-panel proof figure: Era/Theme/Pantun Comparison | Pantun ROC | Era Tokens | Pantun Tokens."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.8))
    plt.subplots_adjust(hspace=0.34, wspace=0.28)

    # Panel A: DualScaleHumanitiesNet vs TF-IDF Baseline across all 3 tasks (Macro-F1)
    ax_a = axes[0, 0]
    tasks = [
        ("era", "Track A1:\nPoetry Era\n(Ren vs Mod)"),
        ("theme", "Track A2:\nPoetry Theme\n(3-Class)"),
        ("pantun", "Track B:\nNusantara Pantun\nCoherence"),
    ]
    x = np.arange(len(tasks))
    width = 0.32
    neural_f1 = [100.0 * float(comparison_scores[k]["neural_macro_f1"]) for k, _ in tasks]
    tfidf_f1 = [100.0 * float(comparison_scores[k]["tfidf_macro_f1"]) for k, _ in tasks]

    r1 = ax_a.bar(x - width / 2, neural_f1, width=width, color="#38bdf8", label="DualScaleHumanitiesNet")
    r2 = ax_a.bar(x + width / 2, tfidf_f1, width=width, color="#f59e0b", label="TF-IDF Char+Word Linear")
    for rects in (r1, r2):
        for rect in rects:
            h = rect.get_height()
            ax_a.text(
                rect.get_x() + rect.get_width() / 2.0,
                h + 1.3,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )
    ax_a.set_ylim(0, 112)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([lbl for _, lbl in tasks], fontsize=9)
    ax_a.set_ylabel("Held-Out Test Macro-F1 (%)")
    ax_a.set_title("A. Stylometry & Literary Coherence Benchmark (Macro-F1)", fontsize=10.5, fontweight="bold")
    ax_a.grid(axis="y", alpha=0.35)
    ax_a.legend(loc="lower right", fontsize=8.5)

    # Panel B: Track B Nusantara Pantun Coherence ROC Curve
    ax_b = axes[0, 1]
    for key, label, color, ls in [
        ("neural", "DualScaleHumanitiesNet (Char-CNN + BiGRU)", "#10b981", "-"),
        ("tfidf", "TF-IDF Char+Word Linear Baseline", "#f43f5e", "--"),
    ]:
        info = pantun_roc[key]
        ax_b.plot(
            info["fpr"],
            info["tpr"],
            label=f"{label} (AUC={info['auc']:.3f})",
            color=color,
            linestyle=ls,
            linewidth=2.3,
        )
    ax_b.plot([0, 1], [0, 1], color="#64748b", linestyle=":", linewidth=1.2)
    ax_b.set_xlabel("False Positive Rate")
    ax_b.set_ylabel("True Positive Rate")
    ax_b.set_title("B. Track B: Nusantara Pantun Sampiran-Isi & Rhyme Coherence ROC", fontsize=10.5, fontweight="bold")
    ax_b.grid(alpha=0.35)
    ax_b.legend(loc="lower right", fontsize=8.3)

    # Panel C: Close-Reading Attention Attribution -- Renaissance vs Modern Poetry Tokens
    ax_c = axes[1, 0]
    ren_toks = era_top_tokens.get("Renaissance", [])[:6]
    mod_toks = era_top_tokens.get("Modern", [])[:6]
    combined_labels = [f"[Ren] {t}" for t, _ in ren_toks][::-1] + [f"[Mod] {t}" for t, _ in mod_toks][::-1]
    combined_vals = [float(v) for _, v in ren_toks][::-1] + [float(v) for _, v in mod_toks][::-1]
    combined_colors = ["#a855f7"] * len(ren_toks) + ["#38bdf8"] * len(mod_toks)
    if combined_labels:
        yc = np.arange(len(combined_labels))
        ax_c.barh(yc, combined_vals, color=combined_colors, alpha=0.9, edgecolor="#e2e8f0", linewidth=0.5)
        ax_c.set_yticks(yc)
        ax_c.set_yticklabels(combined_labels, fontsize=8.6)
        ax_c.set_xlabel("Mean Token Attention Salience")
        ax_c.grid(axis="x", alpha=0.35)
    ax_c.set_title("C. Literary Era Close-Reading: Top Attended Tokens (Ren vs Mod)", fontsize=10.3, fontweight="bold")

    # Panel D: Close-Reading Attention Attribution -- Authentic Nusantara Pantun Tokens
    _plot_token_bars(
        axes[1, 1],
        pantun_top_tokens,
        title="D. Nusantara Pantun Heritage: Top Attended Lexical Markers",
        color="#10b981",
    )

    fig.suptitle(
        "Project 13: Humanities Classical Literature Stylometry & Nusantara Pantun Coherence DL",
        fontsize=13.2,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
