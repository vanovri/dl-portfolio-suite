"""Deterministic offline CPU unit tests for Project 06: YOLOv8 Detection & Segmentation."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.pipeline import (
    cleanup_stray_weights,
    extract_det_metrics,
    extract_seg_metrics,
    find_coco8_val_images,
)
from src.roboflow_adapter import (
    format_yolo_box_label,
    format_yolo_polygon_label,
    normalize_class_names,
    parse_yolo_box_label,
    parse_yolo_polygon_label,
    polygon_to_bbox,
    validate_label_file,
    validate_roboflow_yaml,
)
from src.viz import plot_yolo_det_vs_seg


def test_normalize_class_names_list_and_dict() -> None:
    assert normalize_class_names(["person", "car"]) == {0: "person", 1: "car"}
    assert normalize_class_names({"1": "car", "0": "person"}) == {0: "person", 1: "car"}


def test_validate_roboflow_yaml_success(tmp_path: Path) -> None:
    for split in ("train", "val"):
        img_dir = tmp_path / split / "images"
        img_dir.mkdir(parents=True)
        Image.new("RGB", (64, 64), color=(10, 20, 30)).save(img_dir / "a.jpg")
    cfg = {
        "train": "train/images",
        "val": "val/images",
        "nc": 2,
        "names": ["helmet", "vest"],
        "roboflow": {"workspace": "demo", "project": "ppe", "version": 1},
    }
    ypath = tmp_path / "data.yaml"
    ypath.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    parsed = validate_roboflow_yaml(ypath)
    assert parsed["nc"] == 2
    assert parsed["names"] == {0: "helmet", 1: "vest"}
    assert parsed["splits"]["train"]["image_count"] == 1
    assert parsed["splits"]["val"]["exists"] is True


def test_validate_roboflow_yaml_rejects_nc_mismatch(tmp_path: Path) -> None:
    ypath = tmp_path / "bad.yaml"
    ypath.write_text(yaml.safe_dump({"train": "tr", "val": "va", "nc": 3, "names": ["a", "b"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Class count mismatch"):
        validate_roboflow_yaml(ypath)


def test_parse_and_format_yolo_box_roundtrip() -> None:
    line = format_yolo_box_label(2, 0.5, 0.4, 0.25, 0.3)
    parsed = parse_yolo_box_label(line, num_classes=5)
    assert parsed["class_id"] == 2
    assert np.isclose(parsed["xc"], 0.5)
    assert np.isclose(parsed["h"], 0.3)


def test_parse_yolo_box_rejects_out_of_bounds() -> None:
    with pytest.raises(ValueError):
        parse_yolo_box_label("0 1.2 0.5 0.2 0.2", num_classes=2)
    with pytest.raises(ValueError):
        parse_yolo_box_label("5 0.5 0.5 0.2 0.2", num_classes=2)


def test_parse_and_format_yolo_polygon_and_bbox() -> None:
    pts = [(0.1, 0.2), (0.5, 0.2), (0.5, 0.6), (0.1, 0.6)]
    line = format_yolo_polygon_label(1, pts)
    parsed = parse_yolo_polygon_label(line, num_classes=3)
    assert parsed["class_id"] == 1
    assert len(parsed["points"]) == 4

    xc, yc, w, h = polygon_to_bbox(parsed["points"])
    assert np.isclose(xc, 0.3)
    assert np.isclose(yc, 0.4)
    assert np.isclose(w, 0.4)
    assert np.isclose(h, 0.4)


def test_validate_label_file_detect_and_segment(tmp_path: Path) -> None:
    det_txt = tmp_path / "det.txt"
    det_txt.write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.7 0.1 0.15\n", encoding="utf-8")
    det_rows = validate_label_file(det_txt, task="detect", num_classes=2)
    assert len(det_rows) == 2

    seg_txt = tmp_path / "seg.txt"
    seg_txt.write_text("0 0.1 0.1 0.4 0.1 0.4 0.4\n", encoding="utf-8")
    seg_rows = validate_label_file(seg_txt, task="segment", num_classes=2)
    assert len(seg_rows) == 1
    assert len(seg_rows[0]["points"]) == 3


def test_extract_det_and_seg_metrics_from_mock_results() -> None:
    det_mock = SimpleNamespace(box=SimpleNamespace(map50=0.8123, map=0.6234, mp=0.85, mr=0.79), results_dict={})
    det = extract_det_metrics(det_mock)
    assert det == {"box_map50": 0.8123, "box_map50_95": 0.6234, "precision": 0.85, "recall": 0.79}

    seg_mock = SimpleNamespace(
        box=SimpleNamespace(map50=0.80, map=0.61),
        seg=SimpleNamespace(map50=0.76, map=0.55, mp=0.82, mr=0.74),
        results_dict={},
    )
    seg = extract_seg_metrics(seg_mock)
    assert seg["mask_map50"] == 0.76
    assert seg["mask_map50_95"] == 0.55


def test_cleanup_stray_weights_moves_to_artifacts(tmp_path: Path) -> None:
    root_dir = tmp_path / "root"
    art_dir = root_dir / "artifacts"
    root_dir.mkdir()
    stray = root_dir / "yolov8n.pt"
    stray.write_bytes(b"fake-weights")

    moved = cleanup_stray_weights([root_dir], art_dir)
    assert moved == ["yolov8n.pt"]
    assert not stray.exists()
    assert (art_dir / "yolov8n.pt").exists()


def test_find_coco8_val_images(tmp_path: Path) -> None:
    val_dir = tmp_path / "coco8" / "images" / "val"
    val_dir.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (32, 32), color=(i * 40, 20, 80)).save(val_dir / f"img_{i}.jpg")
    found = find_coco8_val_images(tmp_path, limit=2)
    assert len(found) == 2


def test_plot_yolo_det_vs_seg_creates_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(21)
    pairs = [
        {
            "name": f"coco_{i}.jpg",
            "det_rgb": rng.integers(0, 255, (128, 160, 3), dtype=np.uint8),
            "seg_rgb": rng.integers(0, 255, (128, 160, 3), dtype=np.uint8),
            "det_count": 4,
            "seg_count": 4,
        }
        for i in range(2)
    ]
    det_m = {"box_map50": 0.84, "box_map50_95": 0.65, "precision": 0.88, "recall": 0.81, "fps": 92.4, "ms_per_frame": 10.8}
    seg_m = {"mask_map50": 0.78, "mask_map50_95": 0.56, "precision": 0.83, "recall": 0.77, "fps": 74.1, "ms_per_frame": 13.5}
    out = plot_yolo_det_vs_seg(pairs, det_m, seg_m, tmp_path / "yolo_det_vs_seg.png")
    assert out.exists() and out.stat().st_size > 1000
