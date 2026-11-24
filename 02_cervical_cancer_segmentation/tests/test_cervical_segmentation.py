"""Deterministic offline CPU unit tests for Project 02: Cervical Cancer & Nuclei Segmentation."""
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

from src.data import (
    NucleiSegmentationDataset,
    decode_coco_rle,
    deterministic_split,
    parse_dsb2018_directory,
    rasterize_coco_annotations,
)
from src.metrics import analyze_biopsy_morphometry, boundary_f1_score, compute_segmentation_metrics
from src.model import AdditiveAttentionGate, AttentionUNet, DiceBCEBoundaryLoss, PlainUNet
from src.viz import plot_segmentation_overlay


def test_additive_attention_gate_shape_and_bounds() -> None:
    gate = AdditiveAttentionGate(F_g=16, F_l=16, F_int=8)
    g = torch.randn(2, 16, 32, 32)
    x = torch.ones(2, 16, 32, 32)
    out = gate(g, x)
    assert out.shape == (2, 16, 32, 32)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_plain_and_attention_unet_forward_shapes() -> None:
    x = torch.randn(2, 3, 128, 128)
    plain = PlainUNet(in_channels=3, out_channels=1, base_channels=16)
    att = AttentionUNet(in_channels=3, out_channels=1, base_channels=16)
    y_plain = plain(x)
    y_att = att(x)
    assert y_plain.shape == (2, 1, 128, 128)
    assert y_att.shape == (2, 1, 128, 128)
    assert sum(p.numel() for p in att.parameters()) > sum(p.numel() for p in plain.parameters())


def test_dice_bce_boundary_loss_backward() -> None:
    torch.manual_seed(0)
    model = AttentionUNet(base_channels=8)
    loss_fn = DiceBCEBoundaryLoss()
    x = torch.randn(2, 3, 64, 64)
    y = (torch.rand(2, 1, 64, 64) > 0.7).float()
    logits = model(x)
    loss = loss_fn(logits, y)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_decode_uncompressed_coco_rle() -> None:
    # 4x4 mask in Fortran order: first 4 zeros, next 4 ones, last 8 zeros -> column 1 is 1
    rle = {"size": [4, 4], "counts": [4, 4, 8]}
    mask = decode_coco_rle(rle)
    assert mask.shape == (4, 4)
    assert int(mask[:, 1].sum()) == 4
    assert int(mask.sum()) == 4


def test_rasterize_coco_polygon_and_bbox_fallback() -> None:
    anns = [
        {"segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]]},
        {"segmentation": [], "bbox": [40, 40, 10, 10]},
    ]
    mask = rasterize_coco_annotations(anns, height=64, width=64)
    assert mask.shape == (64, 64)
    assert mask[20, 20] == 1
    assert mask[45, 45] == 1
    assert mask[0, 0] == 0


def test_parse_dsb2018_coco_directory(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (64, 64), color=(30, 20, 40)).save(img_dir / f"cell_{i}.png")
    coco = {
        "images": [{"id": i, "file_name": f"cell_{i}.png", "height": 64, "width": 64} for i in range(4)],
        "annotations": [
            {"id": i, "image_id": i, "segmentation": [[16, 16, 48, 16, 48, 48, 16, 48]]} for i in range(4)
        ],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")
    imgs, masks = parse_dsb2018_directory(tmp_path, img_size=64)
    assert imgs.shape == (4, 64, 64, 3)
    assert masks.shape == (4, 64, 64)
    assert int(masks.sum()) > 0


def test_deterministic_split_reproducibility() -> None:
    imgs = np.zeros((20, 32, 32, 3), dtype=np.uint8)
    masks = np.zeros((20, 32, 32), dtype=np.uint8)
    s1 = deterministic_split(imgs, masks, seed=42)
    s2 = deterministic_split(imgs, masks, seed=42)
    assert len(s1["train"][0]) == 16
    assert len(s1["val"][0]) == 2
    assert len(s1["test"][0]) == 2
    assert np.array_equal(s1["test"][0], s2["test"][0])


def test_nuclei_dataset_tensor_ranges() -> None:
    imgs = np.full((3, 64, 64, 3), 255, dtype=np.uint8)
    masks = np.ones((3, 64, 64), dtype=np.uint8)
    ds = NucleiSegmentationDataset(imgs, masks, is_train=True)
    x, y = ds[0]
    assert x.shape == (3, 64, 64)
    assert y.shape == (1, 64, 64)
    assert 0.0 <= float(x.min()) <= float(x.max()) <= 1.0


def test_segmentation_metrics_perfect_and_zero() -> None:
    gt = np.zeros((2, 1, 32, 32), dtype=np.float32)
    gt[:, :, 8:24, 8:24] = 1.0
    perfect = compute_segmentation_metrics(gt, gt)
    assert perfect["dice"] == 1.0
    assert perfect["iou"] == 1.0
    assert perfect["boundary_f1"] == 1.0

    zeros = compute_segmentation_metrics(np.zeros_like(gt), gt)
    assert zeros["dice"] < 0.01
    assert zeros["boundary_f1"] == 0.0


def test_boundary_f1_tolerance() -> None:
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[10:22, 10:22] = 1
    shifted = np.zeros((32, 32), dtype=np.uint8)
    shifted[11:23, 11:23] = 1
    score = boundary_f1_score(shifted, gt, tolerance=2)
    assert score > 0.85


def test_biopsy_morphometry_nc_ratio_and_count() -> None:
    rgb = np.full((64, 64, 3), (200, 140, 180), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    mask[35:45, 35:45] = 1
    stats = analyze_biopsy_morphometry(rgb, mask)
    assert stats["nuclei_count"] == 2
    assert stats["nuclear_area_px"] == 200
    assert stats["nc_ratio"] > 0.0
    assert 0.0 < stats["nuclear_area_pct"] < 100.0


def test_plot_segmentation_overlay_creates_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    dsb = [
        {
            "rgb": rng.integers(0, 255, (64, 64, 3), dtype=np.uint8),
            "gt_mask": (rng.random((64, 64)) > 0.8).astype(np.uint8),
            "plain_pred": (rng.random((64, 64)) > 0.8).astype(np.uint8),
            "att_pred": (rng.random((64, 64)) > 0.8).astype(np.uint8),
            "plain_dice": 0.78,
            "att_dice": 0.83,
        }
        for _ in range(3)
    ]
    bio = [
        {
            "rgb": rng.integers(0, 255, (64, 64, 3), dtype=np.uint8),
            "mask": (rng.random((64, 64)) > 0.85).astype(np.uint8),
            "label": "HSIL",
            "nc_ratio": 0.24,
            "nuclei_count": 7,
        }
        for _ in range(4)
    ]
    out = plot_segmentation_overlay(dsb, bio, tmp_path / "segmentation_overlay.png")
    assert out.exists() and out.stat().st_size > 1000
