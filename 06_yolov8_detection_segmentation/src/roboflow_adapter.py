"""Roboflow YOLOv8 data.yaml validator and detection/polygon label parser & formatter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def normalize_class_names(raw_names: Any) -> dict[int, str]:
    """Normalize YOLO `names` field from either list[str] or dict[int|str, str]."""
    if isinstance(raw_names, list):
        return {idx: str(name) for idx, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        return {int(k): str(v) for k, v in sorted(raw_names.items(), key=lambda kv: int(kv[0]))}
    raise ValueError(f"Unsupported YOLO `names` type: {type(raw_names).__name__}")


def validate_roboflow_yaml(yaml_path: str | Path) -> dict[str, Any]:
    """Parse and validate a Roboflow-exported YOLOv8 `data.yaml` specification."""
    ypath = Path(yaml_path)
    if not ypath.exists():
        raise FileNotFoundError(f"Roboflow data.yaml not found: {ypath}")

    data = yaml.safe_load(ypath.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {ypath.name}: expected mapping.")

    if "names" not in data:
        raise ValueError("Missing required key `names` in data.yaml.")
    names = normalize_class_names(data["names"])

    nc = int(data.get("nc", len(names)))
    if nc != len(names):
        raise ValueError(f"Class count mismatch: nc={nc} but len(names)={len(names)}.")

    base_root = Path(data["path"]) if "path" in data and data["path"] else ypath.parent
    if not base_root.is_absolute():
        base_root = (ypath.parent / base_root).resolve()

    split_summary: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        rel = data.get(split)
        if not rel:
            continue
        p = Path(str(rel))
        resolved = p if p.is_absolute() else (base_root / p).resolve()
        if not resolved.exists():
            # Roboflow exports sometimes use ../train/images relative to data.yaml
            alt = (ypath.parent / p).resolve()
            if alt.exists():
                resolved = alt
        img_count = 0
        if resolved.exists() and resolved.is_dir():
            img_count = sum(
                1 for f in resolved.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )
        split_summary[split] = {
            "configured": str(rel),
            "exists": resolved.exists(),
            "image_count": img_count,
        }

    if "train" not in split_summary or "val" not in split_summary:
        raise ValueError("Roboflow data.yaml must declare both `train` and `val` entries.")

    return {
        "nc": nc,
        "names": names,
        "splits": split_summary,
        "roboflow_meta": data.get("roboflow", {}),
    }


def parse_yolo_box_label(line: str, num_classes: int | None = None) -> dict[str, Any]:
    """Parse a single YOLO detection label line: `<class_id> <xc> <yc> <w> <h>`."""
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5 tokens for YOLO box label, got {len(parts)}: '{line.strip()}'")
    cls_id = int(parts[0])
    xc, yc, w, h = [float(v) for v in parts[1:]]
    if cls_id < 0 or (num_classes is not None and cls_id >= num_classes):
        raise ValueError(f"Class id {cls_id} out of range for num_classes={num_classes}")
    for name, val in (("xc", xc), ("yc", yc), ("w", w), ("h", h)):
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"Box coordinate {name}={val} outside normalized [0, 1] range")
    if w <= 0.0 or h <= 0.0:
        raise ValueError(f"Box dimensions must be strictly positive, got w={w}, h={h}")
    return {"class_id": cls_id, "xc": xc, "yc": yc, "w": w, "h": h}


def parse_yolo_polygon_label(line: str, num_classes: int | None = None) -> dict[str, Any]:
    """Parse a single YOLO segmentation polygon line: `<class_id> <x1> <y1> ... <xn> <yn>`."""
    parts = line.strip().split()
    if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
        raise ValueError(
            f"Expected class_id + even >=6 polygon coords, got {len(parts)} tokens: '{line.strip()}'"
        )
    cls_id = int(parts[0])
    if cls_id < 0 or (num_classes is not None and cls_id >= num_classes):
        raise ValueError(f"Class id {cls_id} out of range for num_classes={num_classes}")
    coords = [float(v) for v in parts[1:]]
    for c in coords:
        if not (0.0 <= c <= 1.0):
            raise ValueError(f"Polygon coordinate {c} outside normalized [0, 1] range")
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    return {"class_id": cls_id, "points": points}


def polygon_to_bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """Convert normalized polygon vertices [(x, y), ...] into YOLO (xc, yc, w, h)."""
    if len(points) < 3:
        raise ValueError("Polygon requires at least 3 vertices.")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    w = max(xmax - xmin, 1e-6)
    h = max(ymax - ymin, 1e-6)
    xc = xmin + 0.5 * w
    yc = ymin + 0.5 * h
    return (round(xc, 6), round(yc, 6), round(w, 6), round(h, 6))


def format_yolo_box_label(class_id: int, xc: float, yc: float, w: float, h: float) -> str:
    """Format a normalized YOLO detection box row."""
    return f"{int(class_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def format_yolo_polygon_label(class_id: int, points: list[tuple[float, float]]) -> str:
    """Format a normalized YOLO instance segmentation polygon row."""
    if len(points) < 3:
        raise ValueError("Polygon requires at least 3 points.")
    flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
    return f"{int(class_id)} {flat}"


def validate_label_file(
    label_path: str | Path,
    task: str = "detect",
    num_classes: int | None = None,
) -> list[dict[str, Any]]:
    """Validate all non-empty rows of a YOLO `.txt` label file for detect or segment tasks."""
    lpath = Path(label_path)
    if not lpath.exists():
        raise FileNotFoundError(f"Label file not found: {lpath}")
    records: list[dict[str, Any]] = []
    for raw_line in lpath.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if task == "detect":
            records.append(parse_yolo_box_label(line, num_classes=num_classes))
        elif task == "segment":
            records.append(parse_yolo_polygon_label(line, num_classes=num_classes))
        else:
            raise ValueError(f"Unsupported task '{task}'; expected 'detect' or 'segment'.")
    return records
