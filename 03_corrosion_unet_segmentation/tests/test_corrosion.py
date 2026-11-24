"""Deterministic offline CPU unit tests for Project 03: Corrosion U-Net Segmentation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data import CorrosionDataset, locate_coco_root, parse_coco_split_dir, rasterize_rust_polygons
from src.model import (
    FocalDiceLoss,
    ResNet18UNet,
    compute_corrosion_area_pct,
    grade_corrosion_severity,
)
from src.viz import make_corrosion_overlay, plot_corrosion_predictions


def test_resnet18_unet_output_shape_192() -> None:
    model = ResNet18UNet(out_channels=1, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 192, 192)
    y = model(x)
    assert y.shape == (2, 1, 192, 192)
    assert torch.isfinite(y).all()


def test_resnet18_unet_handles_non_power_of_two_shape() -> None:
    model = ResNet18UNet(out_channels=1, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 150, 170)
    y = model(x)
    assert y.shape == (1, 1, 150, 170)


def test_focal_dice_loss_backward() -> None:
    torch.manual_seed(4)
    model = ResNet18UNet(out_channels=1, pretrained=False)
    loss_fn = FocalDiceLoss()
    x = torch.randn(2, 3, 96, 96)
    target = (torch.rand(2, 1, 96, 96) > 0.75).float()
    loss = loss_fn(model(x), target)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_iso_4628_severity_grade_thresholds() -> None:
    assert grade_corrosion_severity(0.0) == "Clean"
    assert grade_corrosion_severity(0.99) == "Clean"
    assert grade_corrosion_severity(1.0) == "Light"
    assert grade_corrosion_severity(4.99) == "Light"
    assert grade_corrosion_severity(5.0) == "Moderate"
    assert grade_corrosion_severity(19.9) == "Moderate"
    assert grade_corrosion_severity(20.0) == "Severe"
    assert grade_corrosion_severity(65.0) == "Severe"


def test_compute_corrosion_area_pct() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:10, :] = 1
    assert compute_corrosion_area_pct(mask) == 10.0
    assert grade_corrosion_severity(compute_corrosion_area_pct(mask)) == "Moderate"


def test_rasterize_rust_polygons_and_bbox() -> None:
    anns = [
        {"segmentation": [[20, 20, 60, 20, 60, 60, 20, 60]]},
        {"segmentation": [], "bbox": [80, 80, 10, 10]},
    ]
    mask = rasterize_rust_polygons(anns, 100, 100)
    assert mask[40, 40] == 1
    assert mask[85, 85] == 1
    assert mask[5, 5] == 0


def test_parse_coco_split_dir_and_locate_root(tmp_path: Path) -> None:
    coco_root = tmp_path / "rust.v3i.coco-segmentation"
    train_dir = coco_root / "train"
    train_dir.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (100, 100), color=(90, 70, 60)).save(train_dir / f"metal_{i}.jpg")
    coco = {
        "images": [{"id": i, "file_name": f"metal_{i}.jpg", "height": 100, "width": 100} for i in range(3)],
        "annotations": [{"id": i, "image_id": i, "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]]} for i in range(3)],
    }
    (train_dir / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")

    found = locate_coco_root(tmp_path)
    assert found == coco_root
    imgs, masks = parse_coco_split_dir(found / "train", img_size=64)
    assert imgs.shape == (3, 64, 64, 3)
    assert masks.shape == (3, 64, 64)
    assert int(masks.sum()) > 0


def test_corrosion_dataset_normalization_and_augmentation() -> None:
    imgs = np.full((4, 64, 64, 3), 128, dtype=np.uint8)
    masks = np.ones((4, 64, 64), dtype=np.uint8)
    ds = CorrosionDataset(imgs, masks, is_train=True)
    x, y = ds[0]
    assert x.shape == (3, 64, 64)
    assert y.shape == (1, 64, 64)
    assert torch.isfinite(x).all()


def test_make_corrosion_overlay_shape() -> None:
    rgb = np.full((64, 64, 3), 100, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 1
    ov = make_corrosion_overlay(rgb, mask)
    assert ov.shape == (64, 64, 3)
    assert ov.dtype == np.uint8
    assert not np.array_equal(ov, rgb)


def test_plot_corrosion_predictions_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(12)
    samples = [
        {
            "rgb": rng.integers(0, 255, (64, 64, 3), dtype=np.uint8),
            "gt_mask": (rng.random((64, 64)) > 0.8).astype(np.uint8),
            "pred_mask": (rng.random((64, 64)) > 0.8).astype(np.uint8),
            "gt_area_pct": 12.5,
            "pred_area_pct": 11.2,
            "gt_grade": "Moderate",
            "pred_grade": "Moderate",
        }
        for _ in range(4)
    ]
    out = plot_corrosion_predictions(samples, tmp_path / "corrosion_predictions.png")
    assert out.exists() and out.stat().st_size > 1000
