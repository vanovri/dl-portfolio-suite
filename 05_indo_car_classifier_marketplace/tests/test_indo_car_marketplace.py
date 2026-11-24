"""Deterministic offline CPU unit tests for Project 05: Indo Car Classifier & Marketplace API."""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.api import DEFAULT_DEMO_CLASSES, create_app
from src.data import CachedCarDataset, rank_candidate_class_ids
from src.marketplace import MarketplaceCatalog, estimate_idr_valuation, format_idr_juta
from src.model import CarPredictor, build_car_classifier
from src.viz import plot_car_predictions_gallery


class MockCarPredictor:
    """Fast deterministic mock predictor for offline FastAPI route tests."""

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names

    def predict_image(self, image: Image.Image, top_k: int = 3) -> list[dict[str, Any]]:
        _ = image.size
        return [
            {"class_id": 0, "class_name": self.class_names[0], "confidence": 0.78},
            {"class_id": 1, "class_name": self.class_names[1], "confidence": 0.14},
            {"class_id": 4, "class_name": self.class_names[4], "confidence": 0.05},
        ][:top_k]


def _make_png_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(40, 90, 140)).save(buf, format="PNG")
    return buf.getvalue()


def test_build_car_classifier_forward_shape() -> None:
    model = build_car_classifier(num_classes=10, pretrained=False)
    model.eval()
    logits = model(torch.randn(2, 3, 160, 160))
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


def test_car_predictor_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = build_car_classifier(num_classes=4, pretrained=False)
    ckpt = tmp_path / "car.pt"
    classes = DEFAULT_DEMO_CLASSES[:4]
    torch.save({"model_state_dict": model.state_dict(), "class_names": classes, "img_size": 96}, ckpt)
    predictor = CarPredictor.from_checkpoint(ckpt, device="cpu")
    preds = predictor.predict_image(Image.new("RGB", (96, 96), color=(100, 100, 100)), top_k=3)
    assert len(preds) == 3
    assert preds[0]["confidence"] >= preds[1]["confidence"]


def test_rank_candidate_class_ids_prefers_indo_brands() -> None:
    sample_names = [
        "Acura TL Sedan 2012",
        "Toyota Corolla Sedan 2012",
        "Bugatti Veyron Coupe 2009",
        "Honda Civic Sedan 2012",
        "BMW X5 SUV 2007",
        "Suzuki Kizashi Sedan 2012",
    ]
    ranked = rank_candidate_class_ids(sample_names, max_candidates=4)
    chosen = [sample_names[i] for i in ranked]
    assert any("Toyota" in n for n in chosen)
    assert any("Honda" in n for n in chosen)
    assert any("Suzuki" in n for n in chosen)


def test_valuation_depreciation_by_year_and_odometer() -> None:
    newer = estimate_idr_valuation("Toyota Camry Sedan 2021", year=2021, odometer_km=25_000, condition="excellent")
    older = estimate_idr_valuation("Toyota Camry Sedan 2012", year=2012, odometer_km=140_000, condition="fair")
    assert newer["estimated_idr"] > older["estimated_idr"]
    assert newer["min_idr"] < newer["estimated_idr"] < newer["max_idr"]
    assert "Rp " in newer["formatted_range"]


def test_valuation_luxury_vs_mass_market() -> None:
    bmw = estimate_idr_valuation("BMW 3 Series Sedan 2018", year=2018, odometer_km=40_000)
    suzuki = estimate_idr_valuation("Suzuki SX4 Hatchback 2018", year=2018, odometer_km=40_000)
    assert bmw["estimated_idr"] > suzuki["estimated_idr"] * 2


def test_format_idr_juta_and_miliar() -> None:
    assert "Jt" in format_idr_juta(325_000_000)
    assert "M" in format_idr_juta(1_250_000_000)


def test_marketplace_catalog_search_filters() -> None:
    cat = MarketplaceCatalog(DEFAULT_DEMO_CLASSES)
    assert len(cat.listings) == len(DEFAULT_DEMO_CLASSES) * 3
    toyota_hits = cat.search_listings(brand="Toyota", limit=5)
    assert len(toyota_hits) >= 1
    assert all(h["brand"] == "Toyota" for h in toyota_hits)


def test_fastapi_health_and_catalog_endpoints() -> None:
    app = create_app(predictor=MockCarPredictor(DEFAULT_DEMO_CLASSES))
    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"
    assert r_health.json()["model_loaded"] is True

    r_cat = client.get("/catalog?brand=Honda&limit=5")
    assert r_cat.status_code == 200
    data = r_cat.json()
    assert data["num_classes"] == 10
    assert len(data["listings"]) >= 1


def test_fastapi_predict_raw_bytes_and_multipart_and_json() -> None:
    app = create_app(predictor=MockCarPredictor(DEFAULT_DEMO_CLASSES))
    client = TestClient(app)
    png_bytes = _make_png_bytes()

    # 1. Raw image/png bytes
    r1 = client.post("/predict?year=2019&odometer_km=45000", content=png_bytes, headers={"content-type": "image/png"})
    assert r1.status_code == 200
    body1 = r1.json()
    assert len(body1["top_predictions"]) == 3
    assert body1["valuation"]["year"] == 2019
    assert len(body1["matching_listings"]) >= 1

    # 2. Standard multipart/form-data file upload (parsed without python-multipart)
    r2 = client.post("/predict", files={"file": ("car.png", png_bytes, "image/png")})
    assert r2.status_code == 200
    assert r2.json()["best_match"]["class_name"] == DEFAULT_DEMO_CLASSES[0]

    # 3. JSON base64 payload
    b64 = base64.b64encode(png_bytes).decode("ascii")
    r3 = client.post("/predict", json={"image_b64": b64})
    assert r3.status_code == 200
    assert r3.json()["best_match"]["confidence"] == 0.78


def test_fastapi_predict_rejects_corrupt_bytes() -> None:
    app = create_app(predictor=MockCarPredictor(DEFAULT_DEMO_CLASSES))
    client = TestClient(app)
    resp = client.post("/predict", content=b"not-an-image", headers={"content-type": "image/png"})
    assert resp.status_code == 400


def test_cached_car_dataset_and_gallery_plot(tmp_path: Path) -> None:
    (tmp_path / "test").mkdir()
    records = []
    for i in range(4):
        rel = f"test/00_{i:04d}.png"
        Image.new("RGB", (64, 64), color=(50 * i, 70, 120)).save(tmp_path / rel)
        records.append({"path": rel, "label": i % 2})
    ds = CachedCarDataset(tmp_path, records, img_size=96, is_train=False)
    x, y = ds[1]
    assert x.shape == (3, 96, 96)
    assert y == 1

    samples = [
        {
            "rgb": ds.load_raw_rgb(i % 4),
            "true_label": DEFAULT_DEMO_CLASSES[i],
            "pred_label": DEFAULT_DEMO_CLASSES[i],
            "confidence": 0.91,
            "idr_range": "Rp 210 Jt - Rp 248 Jt",
        }
        for i in range(8)
    ]
    out = plot_car_predictions_gallery(samples, tmp_path / "car_predictions_gallery.png")
    assert out.exists() and out.stat().st_size > 1000
