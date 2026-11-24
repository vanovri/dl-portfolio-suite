"""Indonesian used-car IDR valuation engine and marketplace listing matcher."""
from __future__ import annotations

import hashlib
import re
from typing import Any

INDO_CITIES = ("Jakarta Selatan", "Surabaya", "Bandung", "Semarang")

BRAND_BASE_OTR_IDR: dict[str, int] = {
    "toyota": 395_000_000,
    "honda": 385_000_000,
    "suzuki": 285_000_000,
    "hyundai": 430_000_000,
    "nissan": 370_000_000,
    "mitsubishi": 410_000_000,
    "ford": 520_000_000,
    "chevrolet": 480_000_000,
    "bmw": 1_180_000_000,
    "mercedes": 1_320_000_000,
    "audi": 1_150_000_000,
    "porsche": 2_450_000_000,
    "lexus": 1_450_000_000,
    "volkswagen": 560_000_000,
    "volvo": 980_000_000,
    "jeep": 1_250_000_000,
    "dodge": 820_000_000,
    "acura": 760_000_000,
}

BODY_MULTIPLIERS: tuple[tuple[str, float, str], ...] = (
    ("convertible", 1.28, "Convertible / Roadster"),
    ("coupe", 1.18, "Sport Coupe"),
    ("suv", 1.15, "SUV"),
    ("minivan", 1.05, "MPV / Minivan"),
    ("van", 0.96, "Commercial / Van"),
    ("cab", 1.12, "Double Cabin / Pickup"),
    ("wagon", 1.04, "Station Wagon"),
    ("hatchback", 0.86, "City Hatchback"),
    ("sedan", 1.00, "Sedan"),
)

CONDITION_MULTIPLIERS: dict[str, float] = {
    "excellent": 1.08,
    "good": 1.00,
    "fair": 0.87,
}


def format_idr_juta(amount_idr: int) -> str:
    """Format an IDR integer into concise Indonesian marketplace notation (Rp XXX Jt / Rp X.XX M)."""
    if amount_idr >= 1_000_000_000:
        return f"Rp {amount_idr / 1_000_000_000:.2f} M"
    return f"Rp {round(amount_idr / 1_000_000):,} Jt".replace(",", ".")


def parse_brand_and_segment(car_label: str) -> tuple[str, str, int, float]:
    """Infer brand, Indonesian market segment, base OTR IDR, and body multiplier from class name."""
    lower = car_label.lower()
    brand = car_label.split()[0] if car_label.strip() else "Vehicle"
    base_otr = 420_000_000
    for key, price in BRAND_BASE_OTR_IDR.items():
        if key in lower:
            brand = "Mercedes-Benz" if key == "mercedes" else key.upper() if key == "bmw" else key.title()
            base_otr = price
            break

    segment = "Passenger Vehicle"
    body_mult = 1.0
    for token, mult, seg_name in BODY_MULTIPLIERS:
        if token in lower:
            body_mult = mult
            segment = seg_name
            break

    return brand, segment, base_otr, body_mult


def extract_model_year(car_label: str, default_year: int = 2018) -> int:
    """Extract 4-digit model year from Stanford Cars label if present."""
    match = re.search(r"\b(199\d|20[0-2]\d)\b", car_label)
    if match:
        return int(match.group(1))
    return default_year


def estimate_idr_valuation(
    car_label: str,
    year: int | None = None,
    odometer_km: int = 55_000,
    condition: str = "good",
    reference_year: int = 2025,
) -> dict[str, Any]:
    """Estimate realistic Indonesian used-car IDR valuation range with year/odometer depreciation."""
    brand, segment, base_otr, body_mult = parse_brand_and_segment(car_label)
    car_year = extract_model_year(car_label, default_year=2018) if year is None else int(year)
    age_years = max(0, reference_year - car_year)

    # Compound annual depreciation (~8.5% per year, floored at 28% residual value)
    age_factor = max(0.28, 0.915 ** age_years)

    # Expected Indonesian urban mileage ~15,000 km/yr; adjust +-0.8% per 10k km deviation
    expected_km = max(10_000, age_years * 15_000)
    km_delta_10k = (int(odometer_km) - expected_km) / 10_000.0
    km_factor = float(min(1.10, max(0.76, 1.0 - 0.012 * km_delta_10k)))

    cond_factor = CONDITION_MULTIPLIERS.get(condition.lower(), 1.0)
    combined_factor = age_factor * km_factor * cond_factor

    raw_est = base_otr * body_mult * combined_factor
    est_idr = int(round(raw_est / 1_000_000.0) * 1_000_000)
    min_idr = int(round((est_idr * 0.92) / 1_000_000.0) * 1_000_000)
    max_idr = int(round((est_idr * 1.08) / 1_000_000.0) * 1_000_000)

    return {
        "car_label": car_label,
        "brand": brand,
        "segment": segment,
        "year": car_year,
        "odometer_km": int(odometer_km),
        "condition": condition.lower(),
        "estimated_idr": est_idr,
        "min_idr": min_idr,
        "max_idr": max_idr,
        "formatted_range": f"{format_idr_juta(min_idr)} - {format_idr_juta(max_idr)}",
        "depreciation_factor": round(combined_factor, 4),
    }


class MarketplaceCatalog:
    """Deterministic Indonesian used-car marketplace catalog for trained car classes."""

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = list(class_names)
        self.listings: list[dict[str, Any]] = self._generate_listings(self.class_names)

    @staticmethod
    def _generate_listings(class_names: list[str]) -> list[dict[str, Any]]:
        listings: list[dict[str, Any]] = []
        dealers = (
            "Nusantara Auto Gallery",
            "Bursa Mobil Kemayoran",
            "Jatim Prima Motor",
            "Bandung Auto Center",
            "Semarang Jaya Showroom",
        )
        conditions = ("excellent", "good", "good", "fair")
        for idx, label in enumerate(class_names):
            base_year = extract_model_year(label, default_year=2018)
            digest = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
            for slot in range(3):
                city = INDO_CITIES[(idx + slot + digest) % len(INDO_CITIES)]
                year = min(2024, max(2008, base_year + (slot - 1)))
                odo = 28_000 + ((digest + slot * 17_500) % 72_000)
                cond = conditions[(idx + slot) % len(conditions)]
                val = estimate_idr_valuation(label, year=year, odometer_km=odo, condition=cond)
                listings.append(
                    {
                        "listing_id": f"IDN-{idx + 1:02d}{slot + 1:02d}",
                        "car_label": label,
                        "brand": val["brand"],
                        "segment": val["segment"],
                        "city": city,
                        "dealer": dealers[(idx + slot) % len(dealers)],
                        "year": year,
                        "odometer_km": odo,
                        "condition": cond,
                        "price_idr": val["estimated_idr"],
                        "formatted_price": format_idr_juta(val["estimated_idr"]),
                        "transmission": "Automatic" if (idx + slot) % 3 != 0 else "Manual",
                    }
                )
        return listings

    def search_listings(
        self,
        car_label: str | None = None,
        city: str | None = None,
        brand: str | None = None,
        max_price_idr: int | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        results = self.listings
        if car_label:
            cl = car_label.lower()
            exact = [r for r in results if r["car_label"].lower() == cl]
            results = exact if exact else [r for r in results if cl in r["car_label"].lower()]
        if city:
            ci = city.lower()
            results = [r for r in results if ci in r["city"].lower()]
        if brand:
            br = brand.lower()
            results = [r for r in results if br in r["brand"].lower()]
        if max_price_idr is not None:
            results = [r for r in results if r["price_idr"] <= int(max_price_idr)]
        return results[: max(1, limit)]
