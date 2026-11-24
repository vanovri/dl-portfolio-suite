"""FastAPI service for Indonesian Car Classification & Used-Car Marketplace Valuation."""
from __future__ import annotations

import base64
import email
import io
import time
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Query, Request
from PIL import Image

from src.marketplace import MarketplaceCatalog, estimate_idr_valuation

DEFAULT_DEMO_CLASSES = [
    "Toyota Camry Sedan 2012",
    "Honda Accord Sedan 2012",
    "Suzuki SX4 Hatchback 2012",
    "Hyundai Elantra Sedan 2007",
    "BMW 3 Series Sedan 2012",
    "Mercedes-Benz C-Class Sedan 2012",
    "Audi A5 Coupe 2012",
    "Nissan Leaf Hatchback 2012",
    "Mitsubishi Lancer Sedan 2012",
    "Ford Focus Sedan 2007",
]


class PredictorProtocol(Protocol):
    class_names: list[str]

    def predict_image(self, image: Image.Image, top_k: int = 3) -> list[dict[str, Any]]:
        ...


def _extract_image_bytes(body: bytes, content_type: str) -> bytes:
    """Extract raw image bytes from image/*, octet-stream, JSON base64, or multipart/form-data."""
    ct = (content_type or "").lower()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body; upload an image.")

    if "multipart/form-data" in ct:
        raw_msg = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
        msg = email.message_from_bytes(raw_msg)
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_filename() or (part.get_content_type() or "").startswith("image/"):
                    payload = part.get_payload(decode=True)
                    if payload:
                        return payload
        raise HTTPException(status_code=400, detail="No image part found in multipart/form-data payload.")

    if "application/json" in ct:
        import json

        try:
            data = json.loads(body.decode("utf-8"))
            b64 = data.get("image_b64") or data.get("image")
            if not b64:
                raise ValueError("Missing 'image_b64' field")
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            return base64.b64decode(b64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON image payload: {exc}") from exc

    return body


def create_app(
    predictor: PredictorProtocol | None = None,
    class_names: list[str] | None = None,
) -> FastAPI:
    """Create the FastAPI application with optional injected predictor for offline testing."""
    classes = list(predictor.class_names) if predictor is not None else list(class_names or DEFAULT_DEMO_CLASSES)
    catalog = MarketplaceCatalog(classes)

    app = FastAPI(
        title="Indonesian Used-Car Classifier & Marketplace API",
        version="1.0.0",
        description="Classifies vehicle photos and returns Indonesian IDR valuation & listings.",
    )
    app.state.predictor = predictor
    app.state.catalog = catalog

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": app.state.predictor is not None,
            "num_classes": len(app.state.catalog.class_names),
        }

    @app.get("/catalog")
    def get_catalog(
        city: str | None = Query(default=None),
        brand: str | None = Query(default=None),
        limit: int = Query(default=12, ge=1, le=50),
    ) -> dict[str, Any]:
        listings = app.state.catalog.search_listings(city=city, brand=brand, limit=limit)
        return {
            "num_classes": len(app.state.catalog.class_names),
            "classes": app.state.catalog.class_names,
            "total_listings": len(app.state.catalog.listings),
            "listings": listings,
        }

    @app.post("/predict")
    async def predict(
        request: Request,
        year: int | None = Query(default=None, ge=1995, le=2025),
        odometer_km: int = Query(default=55_000, ge=0, le=400_000),
        condition: str = Query(default="good"),
        city: str | None = Query(default=None),
    ) -> dict[str, Any]:
        if app.state.predictor is None:
            raise HTTPException(status_code=503, detail="Classifier model is not loaded.")

        body = await request.body()
        img_bytes = _extract_image_bytes(body, request.headers.get("content-type", ""))
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image bytes: {exc}") from exc

        t0 = time.perf_counter()
        top_preds = app.state.predictor.predict_image(pil_img, top_k=3)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        best = top_preds[0]
        valuation = estimate_idr_valuation(
            car_label=best["class_name"],
            year=year,
            odometer_km=odometer_km,
            condition=condition,
        )
        matches = app.state.catalog.search_listings(car_label=best["class_name"], city=city, limit=3)
        return {
            "best_match": best,
            "top_predictions": top_preds,
            "valuation": valuation,
            "matching_listings": matches,
            "inference_ms": round(elapsed_ms, 3),
        }

    return app
