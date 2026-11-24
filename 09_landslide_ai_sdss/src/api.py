"""FastAPI geospatial AI-SDSS inference and GeoJSON hazard-assessment service."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import SpatialResUNet
from src.sdss_engine import DEFAULT_BBOX, assess_susceptibility_tile


class AssessTileRequest(BaseModel):
    """Request schema supporting either a precomputed 2D probability map or a raw (C, H, W) tile."""

    probability_map: list[list[float]] | None = Field(default=None, description="2D (H, W) susceptibility probabilities in [0, 1].")
    tile_tensor: list[list[list[float]]] | None = Field(default=None, description="3D (C, H, W) standardized satellite tile.")
    bbox: list[float] = Field(default_factory=lambda: list(DEFAULT_BBOX), min_length=4, max_length=4)
    pixel_resolution_m: float = Field(default=10.0, gt=0.0)
    min_pixels: int = Field(default=4, ge=1)


def create_app(
    model: SpatialResUNet | None = None,
    in_channels: int = 14,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
) -> FastAPI:
    """Build FastAPI application serving /health and /assess-tile."""
    app = FastAPI(title="Landslide AI-SDSS Service", version="1.0.0")
    dev = torch.device(device)

    loaded_model = model
    model_loaded = loaded_model is not None

    if loaded_model is None and checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=True)
        in_channels = int(ckpt.get("in_channels", in_channels))
        loaded_model = SpatialResUNet(in_channels=in_channels).to(dev)
        loaded_model.load_state_dict(ckpt["model_state_dict"])
        loaded_model.eval()
        model_loaded = True
    elif loaded_model is not None:
        in_channels = int(loaded_model.in_channels)
        loaded_model = loaded_model.to(dev)
        loaded_model.eval()

    app.state.model = loaded_model
    app.state.model_loaded = model_loaded
    app.state.in_channels = in_channels
    app.state.device = str(dev)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": bool(app.state.model_loaded),
            "in_channels": int(app.state.in_channels),
            "device": str(app.state.device),
        }

    @app.post("/assess-tile")
    def assess_tile(payload: AssessTileRequest) -> dict[str, Any]:
        if payload.probability_map is None and payload.tile_tensor is None:
            raise HTTPException(status_code=422, detail="Provide either 'probability_map' or 'tile_tensor'.")

        if payload.probability_map is not None:
            prob_map = np.asarray(payload.probability_map, dtype=np.float32)
            if prob_map.ndim != 2 or prob_map.size == 0:
                raise HTTPException(status_code=422, detail="probability_map must be a non-empty 2D grid.")
        else:
            tile = np.asarray(payload.tile_tensor, dtype=np.float32)
            if tile.ndim != 3 or tile.shape[1] < 8 or tile.shape[2] < 8:
                raise HTTPException(status_code=422, detail="tile_tensor must have shape (C, H, W) with H, W >= 8.")
            net = app.state.model
            if net is None:
                net = SpatialResUNet(in_channels=int(tile.shape[0])).to(dev)
                net.eval()
            elif int(tile.shape[0]) != int(net.in_channels):
                raise HTTPException(
                    status_code=422,
                    detail=f"Expected {net.in_channels} channels, got {tile.shape[0]}.",
                )
            with torch.no_grad():
                x = torch.from_numpy(tile).unsqueeze(0).to(dev)
                prob_map = torch.sigmoid(net(x))[0, 0].cpu().numpy()

        bbox_tuple = (
            float(payload.bbox[0]),
            float(payload.bbox[1]),
            float(payload.bbox[2]),
            float(payload.bbox[3]),
        )
        return assess_susceptibility_tile(
            prob_map=prob_map,
            bbox=bbox_tuple,
            pixel_resolution_m=payload.pixel_resolution_m,
            min_pixels=payload.min_pixels,
        )

    return app


app = create_app()
