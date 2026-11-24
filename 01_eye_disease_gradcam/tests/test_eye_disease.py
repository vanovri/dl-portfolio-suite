"""Deterministic offline CPU unit tests for Project 01: Eye Disease & Grad-CAM."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data import (
    EYE_DISEASE_CLASSES,
    RETINAMNIST_CLASSES,
    RetinaMNISTDataset,
    compute_class_weights,
    denormalize_image,
    load_eye_disease_imagefolder,
    load_retinamnist_from_npz,
)
from src.gradcam import GradCAM, overlay_gradcam
from src.model import build_eye_resnet18, get_gradcam_target_layer
from src.viz import plot_confusion_and_roc, plot_gradcam_gallery


def test_build_resnet18_architecture_and_head() -> None:
    model = build_eye_resnet18(num_classes=5, pretrained=False, dropout=0.3)
    assert isinstance(model.fc, torch.nn.Sequential)
    assert isinstance(model.fc[0], torch.nn.Dropout)
    assert isinstance(model.fc[1], torch.nn.Linear)
    assert model.fc[1].out_features == 5


def test_resnet18_forward_shape() -> None:
    torch.manual_seed(0)
    model = build_eye_resnet18(num_classes=5, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    logits = model(x)
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()


def test_gradcam_target_layer_is_layer4_last() -> None:
    model = build_eye_resnet18(num_classes=5, pretrained=False)
    target = get_gradcam_target_layer(model)
    assert target is model.layer4[-1]


def test_gradcam_heatmap_range_and_hook_cleanup() -> None:
    torch.manual_seed(7)
    model = build_eye_resnet18(num_classes=5, pretrained=False)
    x = torch.randn(1, 3, 128, 128)
    with GradCAM(model, get_gradcam_target_layer(model)) as cam:
        heatmap, pred_cls, probs = cam.generate(x)
        assert heatmap.shape == (128, 128)
        assert heatmap.dtype == np.float32
        assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0
        assert 0 <= pred_cls < 5
        assert np.isclose(probs.sum(), 1.0, atol=1e-4)
    assert len(cam._handles) == 0


def test_gradcam_explicit_target_class() -> None:
    torch.manual_seed(3)
    model = build_eye_resnet18(num_classes=5, pretrained=False)
    x = torch.randn(3, 64, 64)
    with GradCAM(model, get_gradcam_target_layer(model)) as cam:
        heatmap, _, _ = cam.generate(x, target_class=2)
    assert heatmap.shape == (64, 64)


def test_overlay_gradcam_output_format() -> None:
    rgb = np.full((64, 64, 3), 120, dtype=np.uint8)
    hm = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    out = overlay_gradcam(rgb, hm, alpha=0.5)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_retinamnist_dataset_and_denormalize() -> None:
    rng = np.random.default_rng(42)
    imgs = rng.integers(0, 256, size=(6, 128, 128, 3), dtype=np.uint8)
    lbls = np.array([0, 1, 2, 3, 4, 0], dtype=np.int64)
    ds = RetinaMNISTDataset(imgs, lbls, img_size=128, is_train=False)
    tensor, label = ds[2]
    assert tensor.shape == (3, 128, 128)
    assert label == 2
    rgb = denormalize_image(tensor)
    assert rgb.shape == (128, 128, 3)
    assert rgb.dtype == np.uint8


def test_load_retinamnist_from_npz(tmp_path: Path) -> None:
    npz_file = tmp_path / "retinamnist_128.npz"
    rng = np.random.default_rng(1)
    np.savez(
        npz_file,
        train_images=rng.integers(0, 255, (8, 32, 32, 3), dtype=np.uint8),
        train_labels=np.array([[0], [1], [2], [3], [4], [0], [1], [2]]),
        val_images=rng.integers(0, 255, (4, 32, 32, 3), dtype=np.uint8),
        val_labels=np.array([[0], [1], [2], [3]]),
        test_images=rng.integers(0, 255, (4, 32, 32, 3), dtype=np.uint8),
        test_labels=np.array([[4], [3], [2], [1]]),
    )
    splits = load_retinamnist_from_npz(npz_file)
    assert splits["train"][0].shape == (8, 32, 32, 3)
    assert splits["train"][1].shape == (8,)
    assert splits["test"][1].tolist() == [4, 3, 2, 1]


def test_compute_class_weights_handles_imbalance() -> None:
    labels = np.array([0] * 40 + [1] * 10 + [2] * 5 + [3] * 3 + [4] * 2)
    weights = compute_class_weights(labels, num_classes=5)
    assert weights.shape == (5,)
    assert float(weights[4]) > float(weights[0])
    assert torch.isfinite(weights).all()


def test_eye_disease_imagefolder_dropin(tmp_path: Path) -> None:
    for cls in EYE_DISEASE_CLASSES:
        cls_dir = tmp_path / cls
        cls_dir.mkdir(parents=True)
        Image.new("RGB", (64, 64), color=(80, 40, 30)).save(cls_dir / "sample.png")
    ds, classes = load_eye_disease_imagefolder(tmp_path, img_size=64, is_train=False)
    assert classes == sorted(EYE_DISEASE_CLASSES)
    assert len(ds) == 4
    img, target = ds[0]
    assert img.shape == (3, 64, 64)
    assert 0 <= target < 4


def test_single_training_step_backward_cpu() -> None:
    torch.manual_seed(11)
    model = build_eye_resnet18(num_classes=5, pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4, 3, 64, 64)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_visualization_outputs_valid_pngs(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    samples = [
        {
            "rgb": rng.integers(0, 255, (64, 64, 3), dtype=np.uint8),
            "overlay": rng.integers(0, 255, (64, 64, 3), dtype=np.uint8),
            "true_label": RETINAMNIST_CLASSES[i],
            "pred_label": RETINAMNIST_CLASSES[i],
            "confidence": 0.85,
        }
        for i in range(4)
    ]
    g_path = plot_gradcam_gallery(samples, tmp_path / "gradcam_gallery.png")
    assert g_path.exists() and g_path.stat().st_size > 1000

    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4, 0, 2, 2, 3, 4])
    y_prob = np.eye(5)[y_pred] * 0.8 + 0.04
    c_path = plot_confusion_and_roc(y_true, y_pred, y_prob, RETINAMNIST_CLASSES, tmp_path / "confusion_roc.png")
    assert c_path.exists() and c_path.stat().st_size > 1000
