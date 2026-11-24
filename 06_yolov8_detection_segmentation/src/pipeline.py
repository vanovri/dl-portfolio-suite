"""YOLOv8 detection & instance segmentation training, benchmarking, and artifact helpers."""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def configure_ultralytics_dirs(data_dir: str | Path, artifacts_dir: str | Path) -> None:
    """Direct Ultralytics dataset, weight, and run caches into project-local folders."""
    from ultralytics import settings

    d_dir = Path(data_dir).resolve()
    a_dir = Path(artifacts_dir).resolve()
    d_dir.mkdir(parents=True, exist_ok=True)
    a_dir.mkdir(parents=True, exist_ok=True)

    desired = {
        "datasets_dir": str(d_dir),
        "weights_dir": str(a_dir),
        "runs_dir": str(a_dir / "runs"),
    }
    updates = {k: v for k, v in desired.items() if k in settings}
    if updates:
        settings.update(updates)


def cleanup_stray_weights(search_dirs: list[Path], artifacts_dir: Path) -> list[str]:
    """Move any stray root-level `yolo*.pt` weight files into `artifacts_dir`."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for sdir in search_dirs:
        if not sdir.exists():
            continue
        for pt_file in sdir.glob("yolo*.pt"):
            if pt_file.parent.resolve() == artifacts_dir.resolve():
                continue
            dest = artifacts_dir / pt_file.name
            if dest.exists():
                pt_file.unlink(missing_ok=True)
            else:
                shutil.move(str(pt_file), str(dest))
            moved.append(pt_file.name)
    return moved


def find_coco8_val_images(data_dir: str | Path, limit: int = 4) -> list[Path]:
    """Locate downloaded COCO8 validation images inside `data_dir` or sibling `datasets/`."""
    d_dir = Path(data_dir)
    valid_exts = {".jpg", ".jpeg", ".png"}
    search_roots = [d_dir, d_dir.parent / "datasets", d_dir.parent.parent / "datasets"]
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in ("**/coco8*/images/val/*", "**/coco8/images/val/*", "**/images/val/*"):
            found = sorted(p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in valid_exts)
            if found:
                candidates = found
                break
        if candidates:
            break
    if not candidates:
        for root in search_roots:
            if not root.exists():
                continue
            found = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts)
            if found:
                candidates = found
                break
    return candidates[:limit]


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def extract_det_metrics(val_results: Any) -> dict[str, float]:
    """Extract standardized Box mAP@50, mAP@50-95, Precision, and Recall from Ultralytics metrics."""
    box = getattr(val_results, "box", None)
    rdict = getattr(val_results, "results_dict", {}) or {}
    map50 = _safe_float(getattr(box, "map50", rdict.get("metrics/mAP50(B)", 0.0)))
    map50_95 = _safe_float(getattr(box, "map", rdict.get("metrics/mAP50-95(B)", 0.0)))
    precision = _safe_float(getattr(box, "mp", rdict.get("metrics/precision(B)", 0.0)))
    recall = _safe_float(getattr(box, "mr", rdict.get("metrics/recall(B)", 0.0)))
    return {
        "box_map50": round(map50, 4),
        "box_map50_95": round(map50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def extract_seg_metrics(val_results: Any) -> dict[str, float]:
    """Extract Box and Mask mAP@50, mAP@50-95, Precision, and Recall from Ultralytics seg metrics."""
    box = getattr(val_results, "box", None)
    seg = getattr(val_results, "seg", None)
    rdict = getattr(val_results, "results_dict", {}) or {}
    return {
        "box_map50": round(_safe_float(getattr(box, "map50", rdict.get("metrics/mAP50(B)", 0.0))), 4),
        "box_map50_95": round(_safe_float(getattr(box, "map", rdict.get("metrics/mAP50-95(B)", 0.0))), 4),
        "mask_map50": round(_safe_float(getattr(seg, "map50", rdict.get("metrics/mAP50(M)", 0.0))), 4),
        "mask_map50_95": round(_safe_float(getattr(seg, "map", rdict.get("metrics/mAP50-95(M)", 0.0))), 4),
        "precision": round(_safe_float(getattr(seg, "mp", rdict.get("metrics/precision(M)", 0.0))), 4),
        "recall": round(_safe_float(getattr(seg, "mr", rdict.get("metrics/recall(M)", 0.0))), 4),
    }


def benchmark_yolo_inference(
    model: Any,
    image_paths: list[Path],
    imgsz: int = 320,
    device: str | int = "cpu",
    warmup: int = 2,
    repeats: int = 12,
) -> dict[str, float]:
    """Measure steady-state inference throughput (FPS) and latency (ms/frame)."""
    if not image_paths:
        return {"fps": 0.0, "ms_per_frame": 0.0, "frames_benchmarked": 0}

    sources = [str(p) for p in image_paths]
    for _ in range(warmup):
        model.predict(source=sources[0], imgsz=imgsz, device=device, verbose=False)

    if torch.cuda.is_available() and str(device) != "cpu":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    frames = 0
    for _ in range(repeats):
        for src in sources:
            model.predict(source=src, imgsz=imgsz, device=device, verbose=False)
            frames += 1
    if torch.cuda.is_available() and str(device) != "cpu":
        torch.cuda.synchronize()
    elapsed_s = max(time.perf_counter() - t0, 1e-6)
    return {
        "fps": round(frames / elapsed_s, 2),
        "ms_per_frame": round((elapsed_s * 1000.0) / max(frames, 1), 2),
        "frames_benchmarked": frames,
    }


def render_yolo_comparison_pairs(
    det_model: Any,
    seg_model: Any,
    image_paths: list[Path],
    imgsz: int = 320,
    device: str | int = "cpu",
) -> list[dict[str, Any]]:
    """Run both detection and segmentation models on real images and return RGB annotated frames."""
    pairs: list[dict[str, Any]] = []
    for p in image_paths[:2]:
        det_res = det_model.predict(source=str(p), imgsz=imgsz, device=device, conf=0.25, verbose=False)[0]
        seg_res = seg_model.predict(source=str(p), imgsz=imgsz, device=device, conf=0.25, verbose=False)[0]
        det_rgb = det_res.plot()[:, :, ::-1].copy()
        seg_rgb = seg_res.plot()[:, :, ::-1].copy()
        n_det = int(len(det_res.boxes)) if det_res.boxes is not None else 0
        n_seg = int(len(seg_res.masks)) if seg_res.masks is not None else 0
        pairs.append(
            {
                "name": p.name,
                "det_rgb": det_rgb,
                "seg_rgb": seg_rgb,
                "det_count": n_det,
                "seg_count": n_seg,
            }
        )
    return pairs
