"""Training, evaluation, and throughput benchmark for Project 06: YOLOv8 Detection & Segmentation."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
for _p in (str(REPO_ROOT), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.cuda_utils import (
    cleanup_cuda,
    count_parameters,
    get_device,
    get_vram_stats,
    reset_peak_vram,
    seed_everything,
    timed_block,
    write_json_report,
)
from src.pipeline import (
    benchmark_yolo_inference,
    cleanup_stray_weights,
    configure_ultralytics_dirs,
    extract_det_metrics,
    extract_seg_metrics,
    find_coco8_val_images,
    render_yolo_comparison_pairs,
)
from src.viz import plot_yolo_det_vs_seg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune and benchmark YOLOv8n detection & segmentation on COCO8.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable Ultralytics AMP (disabled by default for GTX 16xx FP16 stability)")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    yolo_device: int | str = 0 if device.type == "cuda" else "cpu"

    data_dir = PROJECT_ROOT / "data"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    reports_dir = PROJECT_ROOT / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    configure_ultralytics_dirs(data_dir=data_dir, artifacts_dir=artifacts_dir)

    from ultralytics import YOLO

    print(f"[06_yolov8_detection_segmentation] Device: {device} (yolo={yolo_device}) | imgsz={args.imgsz}")
    reset_peak_vram()

    # 1. YOLOv8n Object Detection on coco8.yaml
    det_base_pt = artifacts_dir / "yolov8n.pt"
    det_model = YOLO(str(det_base_pt))
    with timed_block() as det_timer:
        det_model.train(
            data="coco8.yaml",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=yolo_device,
            workers=0,
            seed=args.seed,
            amp=args.amp,
            project=str(artifacts_dir / "runs"),
            name="det_coco8",
            exist_ok=True,
            verbose=False,
        )
    det_best_src = artifacts_dir / "runs" / "det_coco8" / "weights" / "best.pt"
    det_best_dst = artifacts_dir / "yolov8n_coco8_best.pt"
    if det_best_src.exists():
        shutil.copy2(det_best_src, det_best_dst)
        det_eval_model = YOLO(str(det_best_dst))
    else:
        det_eval_model = det_model
    det_val = det_eval_model.val(data="coco8.yaml", imgsz=args.imgsz, batch=args.batch_size, device=yolo_device, workers=0, verbose=False)
    det_metrics = extract_det_metrics(det_val)
    det_params = count_parameters(det_eval_model.model)

    # 2. YOLOv8n-seg Instance Segmentation on coco8-seg.yaml
    seg_base_pt = artifacts_dir / "yolov8n-seg.pt"
    seg_model = YOLO(str(seg_base_pt))
    with timed_block() as seg_timer:
        seg_model.train(
            data="coco8-seg.yaml",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=yolo_device,
            workers=0,
            seed=args.seed,
            amp=args.amp,
            project=str(artifacts_dir / "runs"),
            name="seg_coco8",
            exist_ok=True,
            verbose=False,
        )
    seg_best_src = artifacts_dir / "runs" / "seg_coco8" / "weights" / "best.pt"
    seg_best_dst = artifacts_dir / "yolov8n_seg_coco8_best.pt"
    if seg_best_src.exists():
        shutil.copy2(seg_best_src, seg_best_dst)
        seg_eval_model = YOLO(str(seg_best_dst))
    else:
        seg_eval_model = seg_model
    seg_val = seg_eval_model.val(data="coco8-seg.yaml", imgsz=args.imgsz, batch=args.batch_size, device=yolo_device, workers=0, verbose=False)
    seg_metrics = extract_seg_metrics(seg_val)
    seg_params = count_parameters(seg_eval_model.model)

    # 3. Throughput benchmark & real COCO visual proof
    val_images = find_coco8_val_images(data_dir, limit=4)
    det_bench = benchmark_yolo_inference(det_eval_model, val_images, imgsz=args.imgsz, device=yolo_device)
    seg_bench = benchmark_yolo_inference(seg_eval_model, val_images, imgsz=args.imgsz, device=yolo_device)
    det_full = {**det_metrics, **det_bench}
    seg_full = {**seg_metrics, **seg_bench}

    pairs = render_yolo_comparison_pairs(det_eval_model, seg_eval_model, val_images, imgsz=args.imgsz, device=yolo_device)
    fig_path = plot_yolo_det_vs_seg(pairs, det_full, seg_full, reports_dir / "yolo_det_vs_seg.png")

    # Clean up any stray root-level yolo*.pt files into artifacts/
    cleanup_stray_weights([PROJECT_ROOT, REPO_ROOT, Path.cwd()], artifacts_dir)
    vram = get_vram_stats()

    report: dict[str, Any] = {
        "project": "06_yolov8_detection_segmentation",
        "dataset_type": "real_public",
        "datasets": {
            "detection": "Ultralytics official coco8.yaml (COCO128 8-image split)",
            "segmentation": "Ultralytics official coco8-seg.yaml (COCO instance masks)",
            "roboflow_adapter": "src/roboflow_adapter.py (drop-in Roboflow YOLOv8 data.yaml & polygon validator)",
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "imgsz": args.imgsz,
            "workers": 0,
            "det_elapsed_s": det_timer["elapsed_s"],
            "seg_elapsed_s": seg_timer["elapsed_s"],
        },
        "detection": {
            "model": "yolov8n",
            **det_params,
            "metrics": det_full,
        },
        "segmentation": {
            "model": "yolov8n-seg",
            **seg_params,
            "metrics": seg_full,
        },
        "hardware": vram,
        "artifacts": {
            "det_checkpoint": det_best_dst.relative_to(PROJECT_ROOT).as_posix(),
            "seg_checkpoint": seg_best_dst.relative_to(PROJECT_ROOT).as_posix(),
            "comparison_plot": fig_path.relative_to(PROJECT_ROOT).as_posix(),
        },
    }
    write_json_report(reports_dir / "metrics.json", report)
    print(
        f"[06_yolov8_detection_segmentation] Done! "
        f"Det mAP@50={det_full['box_map50']:.4f} ({det_full['fps']} FPS) | "
        f"Seg Mask mAP@50={seg_full['mask_map50']:.4f} ({seg_full['fps']} FPS)"
    )
    cleanup_cuda()


if __name__ == "__main__":
    main()
