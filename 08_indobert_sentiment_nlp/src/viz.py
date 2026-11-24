"""Publication-grade evaluation plots for IndoBERT-Lite vs TF-IDF baseline on SmSA."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.cuda_utils import configure_plot_style

CLASS_NAMES = ["Negative", "Neutral", "Positive"]


def plot_indobert_evaluation(
    out_path: Path,
    indobert_metrics: dict[str, Any],
    tfidf_metrics: dict[str, Any],
    sample_predictions: list[dict[str, Any]],
) -> Path:
    """Render 3-panel evaluation figure: Confusion Matrix, Per-Class F1 comparison, and Indonesian samples."""
    configure_plot_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.45], wspace=0.28)

    # Panel 1: Confusion Matrix
    ax_cm = fig.add_subplot(gs[0, 0])
    cm = np.asarray(indobert_metrics["confusion_matrix"], dtype=np.int64)
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks(range(3))
    ax_cm.set_yticks(range(3))
    ax_cm.set_xticklabels(CLASS_NAMES, rotation=15)
    ax_cm.set_yticklabels(CLASS_NAMES)
    ax_cm.set_xlabel("Predicted Sentiment")
    ax_cm.set_ylabel("True Sentiment")
    ax_cm.set_title("IndoBERT-Lite Confusion Matrix (Test)")
    thresh = cm.max() * 0.5 if cm.size > 0 else 0
    for i in range(3):
        for j in range(3):
            val = int(cm[i, j])
            c = "#0f172a" if val > thresh else "#f8fafc"
            ax_cm.text(j, i, str(val), ha="center", va="center", color=c, fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    # Panel 2: Per-Class F1 & Macro-F1 Comparison
    ax_bar = fig.add_subplot(gs[0, 1])
    keys = ["negative", "neutral", "positive", "macro_f1"]
    labels = ["Negative", "Neutral", "Positive", "Macro-F1"]
    bert_scores = [
        indobert_metrics["per_class_f1"]["negative"],
        indobert_metrics["per_class_f1"]["neutral"],
        indobert_metrics["per_class_f1"]["positive"],
        indobert_metrics["macro_f1"],
    ]
    tfidf_scores = [
        tfidf_metrics["per_class_f1"]["negative"],
        tfidf_metrics["per_class_f1"]["neutral"],
        tfidf_metrics["per_class_f1"]["positive"],
        tfidf_metrics["macro_f1"],
    ]
    x = np.arange(len(keys))
    width = 0.36
    b1 = ax_bar.bar(x - width / 2, bert_scores, width, label="IndoBERT-Lite", color="#38bdf8")
    b2 = ax_bar.bar(x + width / 2, tfidf_scores, width, label="TF-IDF + LogReg", color="#f59e0b")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.015,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#e2e8f0",
            )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylim(0.0, 1.12)
    ax_bar.set_ylabel("F1 Score")
    ax_bar.set_title("IndoBERT-Lite vs TF-IDF Baseline (SmSA Test)")
    ax_bar.legend(loc="lower right", fontsize=8.5)
    ax_bar.grid(True, axis="y", alpha=0.35)

    # Panel 3: Sample Indonesian Review Predictions
    ax_txt = fig.add_subplot(gs[0, 2])
    ax_txt.axis("off")
    ax_txt.set_title("Sample Indonesian Test Predictions")

    shown = sample_predictions[:5]
    y_top = 0.93
    step = 0.185
    color_map = {"positive": "#34d399", "neutral": "#fbbf24", "negative": "#fb7185"}

    for idx, item in enumerate(shown):
        y_pos = y_top - idx * step
        raw_txt = str(item.get("text", "")).replace("\n", " ").strip()
        short_txt = raw_txt if len(raw_txt) <= 68 else raw_txt[:65] + "..."
        true_lbl = item.get("true_label", "?")
        pred_lbl = item.get("predicted_label", "?")
        probs = item.get("probabilities", {})
        prob_str = (
            f"neg={probs.get('negative', 0.0):.2f} | "
            f"neu={probs.get('neutral', 0.0):.2f} | "
            f"pos={probs.get('positive', 0.0):.2f}"
        )
        badge_color = color_map.get(pred_lbl, "#94a3b8")
        ax_txt.text(
            0.01,
            y_pos,
            f"[{idx + 1}] \"{short_txt}\"",
            fontsize=8.5,
            color="#f8fafc",
            transform=ax_txt.transAxes,
            va="top",
        )
        ax_txt.text(
            0.03,
            y_pos - 0.055,
            f"True: {true_lbl}  ->  Pred: {pred_lbl}  ({prob_str})",
            fontsize=8.0,
            color=badge_color,
            fontweight="bold",
            transform=ax_txt.transAxes,
            va="top",
        )

    fig.suptitle("IndoNLU SmSA Indonesian Sentiment Classification", fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.86, left=0.05, right=0.98, bottom=0.14)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
