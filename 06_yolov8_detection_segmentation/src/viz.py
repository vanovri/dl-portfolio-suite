"""Visual comparison of YOLOv8n detection boxes vs YOLOv8n-seg instance masks + benchmarks."""
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


def plot_yolo_det_vs_seg(
    pairs: list[dict[str, Any]],
    det_metrics: dict[str, Any],
    seg_metrics: dict[str, Any],
    out_path: str | Path,
) -> Path:
    """Render a 2x2 real COCO comparison (Det vs Seg) over a benchmark bar panel."""
    configure_plot_style()
    n_pairs = min(2, len(pairs))
    if n_pairs == 0:
        raise ValueError("pairs must contain at least 1 detection/segmentation image pair")

    fig = plt.figure(figsize=(14.0, 11.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.15, 0.95], hspace=0.30, wspace=0.18)

    for r in range(2):
        if r < n_pairs:
            item = pairs[r]
            ax_d = fig.add_subplot(gs[r, 0])
            ax_s = fig.add_subplot(gs[r, 1])

            ax_d.imshow(item["det_rgb"])
            ax_d.set_title(
                f"YOLOv8n Detection | {item['name']} ({item['det_count']} boxes)",
                fontsize=10,
                color="#38bdf8",
            )
            ax_d.axis("off")

            ax_s.imshow(item["seg_rgb"])
            ax_s.set_title(
                f"YOLOv8n-seg Instance Masks | {item['name']} ({item['seg_count']} masks)",
                fontsize=10,
                color="#34d399",
            )
            ax_s.axis("off")

    # Bottom-left: Accuracy metrics comparison
    ax_acc = fig.add_subplot(gs[2, 0])
    labels = ["mAP@50", "mAP@50-95", "Precision", "Recall"]
    det_vals = [
        float(det_metrics.get("box_map50", 0.0)),
        float(det_metrics.get("box_map50_95", 0.0)),
        float(det_metrics.get("precision", 0.0)),
        float(det_metrics.get("recall", 0.0)),
    ]
    seg_vals = [
        float(seg_metrics.get("mask_map50", 0.0)),
        float(seg_metrics.get("mask_map50_95", 0.0)),
        float(seg_metrics.get("precision", 0.0)),
        float(seg_metrics.get("recall", 0.0)),
    ]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax_acc.bar(x - w / 2, det_vals, width=w, color="#38bdf8", label="YOLOv8n (Box)")
    b2 = ax_acc.bar(x + w / 2, seg_vals, width=w, color="#34d399", label="YOLOv8n-seg (Mask)")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax_acc.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(labels)
    ax_acc.set_ylim(0.0, 1.15)
    ax_acc.set_ylabel("Score")
    ax_acc.set_title("COCO8 Evaluation: Box Detection vs Mask Segmentation")
    ax_acc.grid(True, axis="y", alpha=0.35)
    ax_acc.legend(loc="upper right", fontsize=8.5)

    # Bottom-right: Throughput FPS comparison
    ax_fps = fig.add_subplot(gs[2, 1])
    models = ["YOLOv8n\n(Detection)", "YOLOv8n-seg\n(Segmentation)"]
    fps_vals = [float(det_metrics.get("fps", 0.0)), float(seg_metrics.get("fps", 0.0))]
    ms_vals = [float(det_metrics.get("ms_per_frame", 0.0)), float(seg_metrics.get("ms_per_frame", 0.0))]
    bars_fps = ax_fps.bar(models, fps_vals, width=0.45, color=["#38bdf8", "#34d399"])
    for bar, fps, ms in zip(bars_fps, fps_vals, ms_vals):
        ax_fps.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(fps_vals + [1.0]) * 0.03,
            f"{fps:.1f} FPS\n({ms:.1f} ms)",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
        )
    ax_fps.set_ylim(0.0, max(fps_vals + [10.0]) * 1.28)
    ax_fps.set_ylabel("Throughput (Frames / sec)")
    ax_fps.set_title("Inference Throughput Benchmark")
    ax_fps.grid(True, axis="y", alpha=0.35)

    fig.suptitle("YOLOv8n Detection vs YOLOv8n-seg Instance Segmentation (COCO8)", fontsize=13, fontweight="bold", y=0.985)
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    return dest
