"""Deterministic offline CPU unit tests for 09_landslide_ai_sdss."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(PROJECT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.api import create_app
from src.dataset import (
    LandslideTileDataset,
    collect_balanced_tiles,
    compute_channel_stats,
    extract_rgb_preview,
    normalize_image_array,
    normalize_mask_array,
)
from src.model import FocalDiceLoss, SpatialResUNet
from src.sdss_engine import assess_susceptibility_tile, classify_risk_zones, extract_geojson_zones
from src.viz import plot_landslide_sdss_map


def test_normalize_image_array_handles_14band_hwc_chw_and_pil():
    hwc = np.ones((64, 64, 14), dtype=np.float32)
    hwc[0, 0, 0] = np.nan
    arr_hwc, info_hwc = normalize_image_array(hwc, tile_size=64)
    assert arr_hwc.shape == (14, 64, 64)
    assert info_hwc["detected_layout"] == "hwc_array"
    assert np.isfinite(arr_hwc).all()

    chw = np.ones((14, 64, 64), dtype=np.float32)
    arr_chw, info_chw = normalize_image_array(chw, tile_size=64)
    assert arr_chw.shape == (14, 64, 64)
    assert info_chw["detected_layout"] == "chw_array"

    pil_img = Image.fromarray(np.full((48, 48, 3), 128, dtype=np.uint8))
    arr_pil, info_pil = normalize_image_array(pil_img, tile_size=32)
    assert arr_pil.shape == (3, 32, 32)
    assert info_pil["detected_layout"] == "pil_image"


def test_normalize_mask_array_handles_2d_3d_and_pil():
    m2d = normalize_mask_array(np.array([[0, 1], [0, 2]], dtype=np.uint8), tile_size=4)
    assert m2d.shape == (4, 4)
    assert set(np.unique(m2d).tolist()).issubset({0, 1})

    m3d = normalize_mask_array(np.zeros((16, 16, 1), dtype=np.float32))
    assert m3d.shape == (16, 16)

    pil_m = Image.fromarray(np.full((16, 16), 255, dtype=np.uint8))
    m_pil = normalize_mask_array(pil_m)
    assert m_pil.shape == (16, 16) and int(m_pil.sum()) == 256


def test_collect_balanced_tiles_from_mock_stream():
    rng = np.random.default_rng(0)
    stream = []
    for i in range(20):
        mask = np.zeros((32, 32), dtype=np.uint8)
        if i % 2 == 0:
            mask[8:16, 8:16] = 1
        stream.append({"image": rng.standard_normal((32, 32, 14)).astype(np.float32), "label": mask})

    X, y, info = collect_balanced_tiles(stream, target_count=10, tile_size=32, pos_ratio=0.6)
    assert X.shape == (10, 14, 32, 32)
    assert y.shape == (10, 32, 32)
    assert info["positive_tiles"] == 6
    assert info["negative_tiles"] == 4


def test_landslide_dataset_standardizes_and_augments():
    X = np.random.default_rng(1).standard_normal((6, 14, 32, 32)).astype(np.float32) * 5.0 + 2.0
    y = np.zeros((6, 32, 32), dtype=np.uint8)
    y[:, 10:15, 10:15] = 1
    mean, std = compute_channel_stats(X)
    ds = LandslideTileDataset(X, y, mean, std, augment=True)
    img0, mask0 = ds[0]
    assert img0.shape == (14, 32, 32)
    assert mask0.shape == (1, 32, 32)
    assert torch.isfinite(img0).all()


def test_spatial_resunet_supports_14ch_and_3ch_and_backprop():
    for c in (3, 14):
        model = SpatialResUNet(in_channels=c, base_channels=16)
        x = torch.randn(2, c, 64, 64)
        y = torch.randint(0, 2, (2, 1, 64, 64)).float()
        logits = model(x)
        assert logits.shape == (2, 1, 64, 64)
        loss = FocalDiceLoss(pos_weight=3.0)(logits, y)
        assert torch.isfinite(loss)
        loss.backward()
        grad_sum = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
        assert grad_sum > 0.0


def test_focal_dice_loss_lower_on_perfect_logits():
    loss_fn = FocalDiceLoss(pos_weight=4.0)
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:10, 4:10] = 1.0
    good_logits = torch.where(target > 0.5, torch.full_like(target, 6.0), torch.full_like(target, -6.0))
    bad_logits = -good_logits
    assert float(loss_fn(good_logits, target)) < float(loss_fn(bad_logits, target))


def test_classify_risk_zones_exact_tier_boundaries():
    probs = np.array([[0.05, 0.19, 0.20, 0.49], [0.50, 0.74, 0.75, 0.99]], dtype=np.float32)
    zones = classify_risk_zones(probs)
    expected = np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.uint8)
    np.testing.assert_array_equal(zones, expected)


def test_extract_geojson_zones_valid_feature_collection_and_area():
    prob_map = np.full((32, 32), 0.05, dtype=np.float32)
    prob_map[2:6, 2:6] = 0.85  # 16 pixels -> Critical (16 * 100 m^2 / 10000 = 0.16 ha)
    prob_map[20:24, 20:24] = 0.62  # 16 pixels -> High
    fc = extract_geojson_zones(prob_map, pixel_resolution_m=10.0, min_pixels=4)

    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 2
    first = fc["features"][0]
    assert first["properties"]["risk_tier"] == "Critical"
    assert first["properties"]["advisory"] == "Evacuate"
    assert abs(first["properties"]["area_ha"] - 0.16) < 1e-4
    ring = first["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1] and len(ring) == 5
    assert fc["features"][1]["properties"]["risk_tier"] == "High"
    assert fc["features"][1]["properties"]["advisory"] == "Early Warning"


def test_assess_susceptibility_tile_low_and_critical_advisories():
    low = assess_susceptibility_tile(np.full((16, 16), 0.05, dtype=np.float32))
    assert low["overall_risk_tier"] == "Low"
    assert low["advisory"] == "Normal"
    assert low["hazard_zone_count"] == 0

    crit_map = np.full((16, 16), 0.1, dtype=np.float32)
    crit_map[4:8, 4:8] = 0.92
    crit = assess_susceptibility_tile(crit_map)
    assert crit["overall_risk_tier"] == "Critical"
    assert crit["advisory"] == "Evacuate"
    assert crit["hazard_zone_count"] == 1


def test_fastapi_health_and_assess_tile_probability_map():
    model = SpatialResUNet(in_channels=3, base_channels=8)
    client = TestClient(create_app(model=model, device="cpu"))

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"
    assert r_health.json()["model_loaded"] is True

    grid = np.full((16, 16), 0.1, dtype=np.float32)
    grid[2:6, 2:6] = 0.88
    r_assess = client.post("/assess-tile", json={"probability_map": grid.tolist(), "pixel_resolution_m": 10.0})
    assert r_assess.status_code == 200
    body = r_assess.json()
    assert body["overall_risk_tier"] == "Critical"
    assert body["geojson"]["type"] == "FeatureCollection"
    assert body["hazard_zone_count"] == 1


def test_fastapi_assess_tile_from_raw_tensor():
    model = SpatialResUNet(in_channels=3, base_channels=8)
    client = TestClient(create_app(model=model, device="cpu"))
    tile = np.zeros((3, 16, 16), dtype=np.float32).tolist()
    resp = client.post("/assess-tile", json={"tile_tensor": tile})
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall_risk_tier"] in {"Low", "Moderate", "High", "Critical"}
    assert "geojson" in body


def test_extract_rgb_preview_and_plot_sdss_map(tmp_path: Path):
    tile = np.random.default_rng(3).standard_normal((14, 32, 32)).astype(np.float32)
    rgb = extract_rgb_preview(tile)
    assert rgb.shape == (32, 32, 3)
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0

    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[8:16, 8:16] = 1
    prob = np.full((32, 32), 0.1, dtype=np.float32)
    prob[8:16, 8:16] = 0.85
    summary = assess_susceptibility_tile(prob)
    out = plot_landslide_sdss_map(
        out_path=tmp_path / "sdss_map.png",
        raw_tile=tile,
        gt_mask=gt,
        prob_map=prob,
        history={"train_loss": [0.6, 0.4], "val_loss": [0.58, 0.42], "val_iou": [0.3, 0.55]},
        all_y_true=gt[None, :, :],
        all_y_prob=prob[None, :, :],
        tile_summary=summary,
    )
    assert out.exists() and out.stat().st_size > 1000
