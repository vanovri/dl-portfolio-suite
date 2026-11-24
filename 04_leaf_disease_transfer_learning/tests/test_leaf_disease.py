"""Deterministic offline CPU unit tests for Project 04: Leaf Disease Transfer Learning."""
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

from src.data import CachedLeafDataset, build_leaf_transforms, clean_class_name
from src.model import SUPPORTED_ARCHS, build_leaf_model, checkpoint_size_mb
from src.viz import plot_model_comparison
from train import top_k_accuracy


def test_build_efficientnet_b0_shape() -> None:
    model = build_leaf_model("efficientnet_b0", num_classes=10, pretrained=False)
    model.eval()
    out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


def test_build_mobilenet_v3_small_shape() -> None:
    model = build_leaf_model("mobilenet_v3_small", num_classes=10, pretrained=False)
    model.eval()
    out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


def test_mobilenet_smaller_than_efficientnet() -> None:
    eff = build_leaf_model("efficientnet_b0", num_classes=10, pretrained=False)
    mob = build_leaf_model("mobilenet_v3_small", num_classes=10, pretrained=False)
    eff_p = sum(p.numel() for p in eff.parameters())
    mob_p = sum(p.numel() for p in mob.parameters())
    assert mob_p < eff_p


def test_unsupported_arch_raises() -> None:
    with pytest.raises(ValueError):
        build_leaf_model("unknown_net", num_classes=10, pretrained=False)


def test_clean_class_name_formatting() -> None:
    assert clean_class_name("Tomato___Early_blight") == "Tomato - Early blight"
    assert clean_class_name("Apple___healthy") == "Apple - healthy"


def test_cached_leaf_dataset_reads_local_pngs(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    records = []
    for i in range(4):
        rel = f"train/00_{i:04d}.png"
        Image.new("RGB", (128, 128), color=(30 + i * 20, 120, 40)).save(tmp_path / rel)
        records.append({"path": rel, "label": i % 2})
    ds = CachedLeafDataset(tmp_path, records, img_size=128, is_train=True)
    x, y = ds[1]
    assert x.shape == (3, 128, 128)
    assert y == 1
    rgb = ds.load_raw_rgb(0)
    assert rgb.shape == (128, 128, 3)


def test_label_smoothing_training_step() -> None:
    torch.manual_seed(0)
    model = build_leaf_model("mobilenet_v3_small", num_classes=10, pretrained=False)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4, 3, 128, 128)
    y = torch.tensor([0, 3, 5, 9], dtype=torch.long)
    loss = criterion(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_top_k_accuracy_calculation() -> None:
    y_true = np.array([0, 2, 4])
    y_prob = np.array(
        [
            [0.6, 0.2, 0.1, 0.05, 0.05],
            [0.5, 0.3, 0.15, 0.03, 0.02],
            [0.5, 0.2, 0.15, 0.14, 0.01],
        ]
    )
    assert np.isclose(top_k_accuracy(y_true, y_prob, k=3), 2.0 / 3.0)


def test_checkpoint_size_mb(tmp_path: Path) -> None:
    f = tmp_path / "dummy.pt"
    f.write_bytes(b"0" * (1024 * 1024))
    assert checkpoint_size_mb(f) == 1.0
    assert checkpoint_size_mb(tmp_path / "missing.pt") == 0.0


def test_plot_model_comparison_creates_png(tmp_path: Path) -> None:
    hist = [{"epoch": e, "train_loss": 1.2 / e, "val_accuracy": 0.5 + 0.1 * e} for e in range(1, 5)]
    eff_sum = {"total_params": 4_020_000, "weight_size_mb": 16.2, "test_metrics": {"accuracy": 0.94, "inference_ms_per_image": 2.4}}
    mob_sum = {"total_params": 1_530_000, "weight_size_mb": 6.3, "test_metrics": {"accuracy": 0.90, "inference_ms_per_image": 1.1}}
    rng = np.random.default_rng(7)
    samples = [
        {
            "rgb": rng.integers(0, 255, (64, 64, 3), dtype=np.uint8),
            "true_label": f"Leaf Class {i}",
            "eff_pred": f"Leaf Class {i}",
            "eff_conf": 0.95,
            "mob_pred": f"Leaf Class {i}",
            "mob_conf": 0.89,
        }
        for i in range(4)
    ]
    out = plot_model_comparison(hist, hist, eff_sum, mob_sum, samples, tmp_path / "model_comparison.png")
    assert out.exists() and out.stat().st_size > 1000
