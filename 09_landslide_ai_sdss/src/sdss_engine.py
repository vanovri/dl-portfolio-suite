"""Operational AI-SDSS Landslide Decision Engine with 4-Tier Risk Zoning and GeoJSON Export."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import label as connected_components

RISK_TIER_LABELS: dict[int, str] = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Critical",
}

TIER_ADVISORY: dict[str, str] = {
    "Low": "Normal",
    "Moderate": "Monitor",
    "High": "Early Warning",
    "Critical": "Evacuate",
}

DEFAULT_BBOX: tuple[float, float, float, float] = (106.8000, -6.9000, 106.8128, -6.8872)


def classify_risk_zones(prob_map: np.ndarray) -> np.ndarray:
    """Map pixel landslide probabilities into 4 operational risk tiers.

    Tiers:
        0 = Low (< 0.20)
        1 = Moderate ([0.20, 0.50))
        2 = High ([0.50, 0.75))
        3 = Critical (>= 0.75)
    """
    p = np.asarray(prob_map, dtype=np.float32)
    zones = np.zeros(p.shape, dtype=np.uint8)
    zones[(p >= 0.20) & (p < 0.50)] = 1
    zones[(p >= 0.50) & (p < 0.75)] = 2
    zones[p >= 0.75] = 3
    return zones


def extract_geojson_zones(
    prob_map: np.ndarray,
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
    pixel_resolution_m: float = 10.0,
    min_pixels: int = 4,
) -> dict[str, Any]:
    """Export connected High/Critical (>= 0.50) landslide regions as a valid GeoJSON FeatureCollection."""
    p = np.asarray(prob_map, dtype=np.float32)
    if p.ndim != 2:
        raise ValueError(f"Expected 2D probability map, got shape {p.shape}")

    h, w = p.shape
    min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox]
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat

    hazard_mask = p >= 0.50
    labeled, num_features = connected_components(hazard_mask, structure=np.ones((3, 3), dtype=np.int32))
    pixel_area_ha = (float(pixel_resolution_m) ** 2) / 10000.0

    features: list[dict[str, Any]] = []
    for comp_id in range(1, int(num_features) + 1):
        ys, xs = np.where(labeled == comp_id)
        n_pix = int(ys.size)
        if n_pix < min_pixels:
            continue

        comp_probs = p[ys, xs]
        max_p = float(np.max(comp_probs))
        mean_p = float(np.mean(comp_probs))
        tier = "Critical" if max_p >= 0.75 else "High"
        advisory = TIER_ADVISORY[tier]
        area_ha = round(n_pix * pixel_area_ha, 4)

        r0, r1 = int(ys.min()), int(ys.max()) + 1
        c0, c1 = int(xs.min()), int(xs.max()) + 1

        # Map pixel bounds onto geographic WGS84 bounding polygon (row 0 = north/max_lat)
        x_west = round(min_lon + (c0 / max(w, 1)) * lon_span, 6)
        x_east = round(min_lon + (c1 / max(w, 1)) * lon_span, 6)
        y_north = round(max_lat - (r0 / max(h, 1)) * lat_span, 6)
        y_south = round(max_lat - (r1 / max(h, 1)) * lat_span, 6)

        ring = [
            [x_west, y_south],
            [x_east, y_south],
            [x_east, y_north],
            [x_west, y_north],
            [x_west, y_south],
        ]

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "zone_id": len(features) + 1,
                    "risk_tier": tier,
                    "advisory": advisory,
                    "pixel_count": n_pix,
                    "area_ha": area_ha,
                    "max_probability": round(max_p, 4),
                    "mean_probability": round(mean_p, 4),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ring],
                },
            }
        )

    # Sort descending by max_probability then area so most severe hazard appears first
    features.sort(key=lambda f: (f["properties"]["max_probability"], f["properties"]["area_ha"]), reverse=True)
    for idx, feat in enumerate(features, start=1):
        feat["properties"]["zone_id"] = idx

    return {
        "type": "FeatureCollection",
        "metadata": {
            "crs": "EPSG:4326",
            "bbox": [min_lon, min_lat, max_lon, max_lat],
            "pixel_resolution_m": float(pixel_resolution_m),
            "threshold_high_critical": 0.50,
            "hazard_zone_count": len(features),
        },
        "features": features,
    }


def assess_susceptibility_tile(
    prob_map: np.ndarray,
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
    pixel_resolution_m: float = 10.0,
    min_pixels: int = 4,
) -> dict[str, Any]:
    """Compute operational AI-SDSS risk assessment summary + GeoJSON hazard polygons."""
    p = np.clip(np.asarray(prob_map, dtype=np.float32), 0.0, 1.0)
    zones = classify_risk_zones(p)
    total_pix = max(int(zones.size), 1)
    pixel_area_ha = (float(pixel_resolution_m) ** 2) / 10000.0

    counts = {name.lower(): int(np.sum(zones == code)) for code, name in RISK_TIER_LABELS.items()}
    coverage_pct = {k: round(100.0 * v / total_pix, 2) for k, v in counts.items()}

    max_prob = float(np.max(p))
    mean_prob = float(np.mean(p))
    if max_prob >= 0.75:
        overall_tier = "Critical"
    elif max_prob >= 0.50:
        overall_tier = "High"
    elif max_prob >= 0.20:
        overall_tier = "Moderate"
    else:
        overall_tier = "Low"

    geojson = extract_geojson_zones(
        p,
        bbox=bbox,
        pixel_resolution_m=pixel_resolution_m,
        min_pixels=min_pixels,
    )
    high_crit_pixels = counts["high"] + counts["critical"]

    return {
        "overall_risk_tier": overall_tier,
        "advisory": TIER_ADVISORY[overall_tier],
        "max_probability": round(max_prob, 4),
        "mean_probability": round(mean_prob, 4),
        "tier_pixel_counts": counts,
        "tier_coverage_pct": coverage_pct,
        "high_or_critical_area_ha": round(high_crit_pixels * pixel_area_ha, 4),
        "hazard_zone_count": len(geojson["features"]),
        "geojson": geojson,
    }
