"""Exact IceCube 5,160-DOM South Pole geometry and relativistic Cherenkov photon arrival simulator."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Physical constants for South Pole glacial ice
C_VAC_M_PER_NS: float = 0.299792458
N_ICE_GROUP: float = 1.31
COS_CHERENKOV: float = 1.0 / N_ICE_GROUP
SIN_CHERENKOV: float = float(np.sqrt(1.0 - COS_CHERENKOV ** 2))
TAN_CHERENKOV: float = SIN_CHERENKOV / COS_CHERENKOV
SQRT_N2_MINUS_1: float = float(np.sqrt(N_ICE_GROUP ** 2 - 1.0))
LAMBDA_ABS_M: float = 95.0

# Canonical 78 standard strings (hexagonal 125 m triangular lattice) + 8 DeepCore infill strings
STANDARD_ROW_COUNTS: tuple[int, ...] = (6, 10, 10, 10, 10, 10, 9, 8, 5)
DEEPCORE_XY_OFFSETS: tuple[tuple[float, float], ...] = (
    (46.3, -34.9),
    (1.7, -50.9),
    (57.2, 10.6),
    (113.2, -60.5),
    (72.4, -66.6),
    (19.5, -6.5),
    (90.5, -18.2),
    (31.2, -79.4),
)


def build_icecube_5160_geometry() -> pd.DataFrame:
    """Construct the exact 5,160-DOM IceCube South Pole detector geometry.

    Layout:
        - 78 standard IceCube strings on a 125 m triangular hexagonal lattice,
          60 DOMs per string at 17.0 m vertical spacing (z = +500 m down to -503 m): 4,680 DOMs.
        - 8 DeepCore infill strings in the central fiducial volume,
          60 DOMs per string (10 upper veto DOMs + 50 dense 7.0 m spacing DOMs): 480 DOMs.
        - Total = 5,160 DOMs with sensor_id in [0, 5159].
    """
    spacing_m = 125.0
    dy = spacing_m * float(np.sqrt(3.0)) / 2.0

    rows: list[dict[str, Any]] = []
    sensor_id = 0
    string_id = 1

    # 78 standard strings
    row_indices = range(-4, 5)  # 9 rows matching sum(STANDARD_ROW_COUNTS) == 78
    standard_z = np.array([500.0 - 17.0 * k for k in range(60)], dtype=np.float32)

    for r_idx, count in zip(row_indices, STANDARD_ROW_COUNTS):
        y = float(r_idx) * dy
        x_shift = 0.5 * spacing_m if (r_idx % 2 != 0) else 0.0
        q_start = -(count - 1) / 2.0
        for j in range(count):
            x = (q_start + j) * spacing_m + x_shift
            for z in standard_z:
                rows.append(
                    {
                        "sensor_id": sensor_id,
                        "x": round(float(x), 3),
                        "y": round(float(y), 3),
                        "z": round(float(z), 3),
                        "string_id": string_id,
                        "is_deepcore": False,
                    }
                )
                sensor_id += 1
            string_id += 1

    # 8 DeepCore infill strings (10 upper veto DOMs + 50 dense lower DOMs = 60 DOMs/string)
    dc_upper_z = np.array([185.0 - 10.0 * k for k in range(10)], dtype=np.float32)
    dc_lower_z = np.array([-150.0 - 7.0 * k for k in range(50)], dtype=np.float32)
    deepcore_z = np.concatenate([dc_upper_z, dc_lower_z])

    for x_dc, y_dc in DEEPCORE_XY_OFFSETS:
        for z in deepcore_z:
            rows.append(
                {
                    "sensor_id": sensor_id,
                    "x": round(float(x_dc), 3),
                    "y": round(float(y_dc), 3),
                    "z": round(float(z), 3),
                    "string_id": string_id,
                    "is_deepcore": True,
                }
            )
            sensor_id += 1
        string_id += 1

    df = pd.DataFrame(rows)
    assert len(df) == 5160, f"Expected exact 5160 DOMs, built {len(df)}"
    return df


def azimuth_zenith_to_vec(azimuth: np.ndarray | float, zenith: np.ndarray | float) -> np.ndarray:
    """Convert IceCube spherical (azimuth, zenith) angles in radians to 3D unit vector(s)."""
    az = np.asarray(azimuth, dtype=np.float64)
    ze = np.asarray(zenith, dtype=np.float64)
    sin_z = np.sin(ze)
    vec = np.stack([sin_z * np.cos(az), sin_z * np.sin(az), np.cos(ze)], axis=-1)
    return vec.astype(np.float32)


def vec_to_azimuth_zenith(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 3D vector(s) to IceCube (azimuth in [0, 2*pi), zenith in [0, pi])."""
    v = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    u = v / np.maximum(norm, 1e-12)
    zenith = np.arccos(np.clip(u[..., 2], -1.0, 1.0))
    azimuth = np.mod(np.arctan2(u[..., 1], u[..., 0]), 2.0 * np.pi)
    return azimuth.astype(np.float32), zenith.astype(np.float32)


def simulate_single_event(
    dom_xyz: np.ndarray,
    rng: np.random.Generator,
    azimuth: float | None = None,
    zenith: float | None = None,
    add_noise: bool = True,
    scattering_scale: float = 1.0,
    min_signal_pulses: int = 12,
) -> dict[str, Any]:
    """Simulate relativistic Cherenkov photon arrival pulses across the 5,160-DOM array.

    Returns exact Kaggle IceCube event schema:
        sensor_id (int32), time (float32 ns), charge (float32 PE), auxiliary (bool),
        plus truth targets azimuth, zenith, direction_vec, vertex.
    """
    if azimuth is None:
        azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
    if zenith is None:
        cos_z = float(rng.uniform(-1.0, 1.0))
        zenith = float(np.arccos(np.clip(cos_z, -1.0, 1.0)))

    u = azimuth_zenith_to_vec(azimuth, zenith).astype(np.float32)
    t0 = float(rng.uniform(9500.0, 10500.0))

    sig_idx = np.empty(0, dtype=np.int32)
    t_sig = np.empty(0, dtype=np.float32)
    q_sig = np.empty(0, dtype=np.float32)
    vertex = np.zeros(3, dtype=np.float32)

    for attempt in range(8):
        rad = float(np.sqrt(rng.uniform(0.0, 1.0))) * (380.0 if attempt < 4 else 220.0)
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        vz = float(rng.uniform(-380.0, 380.0)) if attempt < 4 else float(rng.uniform(-220.0, 220.0))
        vertex = np.array([rad * np.cos(ang), rad * np.sin(ang), vz], dtype=np.float32)

        d = dom_xyz - vertex[None, :]  # (5160, 3)
        s_along = d @ u  # longitudinal projection along muon track (m)
        d_sq = np.sum(d * d, axis=1)
        d_perp = np.sqrt(np.maximum(d_sq - s_along * s_along, 0.0))

        # Cherenkov cone emission coordinate & photon distance
        s_emit = s_along - d_perp / TAN_CHERENKOV
        d_photon = d_perp / SIN_CHERENKOV

        # Muon track length window: through-going or starting tracks illuminate DOMs forward of vertex
        track_len = float(rng.uniform(350.0, 950.0))
        in_cone = (s_emit >= -120.0) & (s_emit <= track_len) & (d_perp <= 260.0)

        energy_scale = float(rng.uniform(55.0, 190.0)) * (1.5 if attempt >= 4 else 1.0)
        mu_pe = np.zeros(dom_xyz.shape[0], dtype=np.float32)
        mu_pe[in_cone] = (
            energy_scale
            * np.exp(-d_photon[in_cone] / LAMBDA_ABS_M)
            / (d_perp[in_cone] + 14.0)
        )

        # Sample Poisson photoelectron detections
        hits = rng.poisson(np.clip(mu_pe, 0.0, 40.0))
        hit_mask = hits > 0
        if int(np.sum(hit_mask)) >= min_signal_pulses or attempt == 7:
            if int(np.sum(hit_mask)) < min_signal_pulses:
                # Guarantee sufficient pulses by selecting closest in-cone DOMs
                order = np.argsort(d_perp + np.where(in_cone, 0.0, 1e4))
                chosen = order[:min_signal_pulses]
                sig_idx = chosen.astype(np.int32)
                k_pe = np.maximum(hits[chosen], 1).astype(np.float32)
            else:
                sig_idx = np.flatnonzero(hit_mask).astype(np.int32)
                k_pe = hits[sig_idx].astype(np.float32)

            # Relativistic Cherenkov arrival time: t0 + (s_along + d_perp * sqrt(n^2 - 1)) / c
            t_geom = t0 + (s_along[sig_idx] + d_perp[sig_idx] * SQRT_N2_MINUS_1) / C_VAC_M_PER_NS
            if scattering_scale > 0.0:
                scatter_mean_ns = scattering_scale * (10.0 + 0.38 * d_photon[sig_idx])
                dt_scatter = rng.exponential(scale=scatter_mean_ns).astype(np.float32)
                pmt_jitter = rng.normal(0.0, 3.0 * scattering_scale, size=sig_idx.shape[0]).astype(np.float32)
            else:
                dt_scatter = np.zeros_like(t_geom)
                pmt_jitter = np.zeros_like(t_geom)

            t_sig = (t_geom + dt_scatter + pmt_jitter).astype(np.float32)
            charge_fluct = rng.gamma(shape=4.0, scale=0.25, size=sig_idx.shape[0]).astype(np.float32)
            q_sig = np.maximum(k_pe * charge_fluct, 0.15).astype(np.float32)
            break

    aux_sig = np.zeros(sig_idx.shape[0], dtype=bool)

    if add_noise:
        # Random auxiliary dark-noise pulses across all 5,160 DOMs
        n_aux = int(rng.integers(12, 32))
        aux_idx = rng.integers(0, dom_xyz.shape[0], size=n_aux, dtype=np.int32)
        t_aux = rng.uniform(t0 - 3000.0, t0 + 3500.0, size=n_aux).astype(np.float32)
        q_aux = rng.uniform(0.25, 1.4, size=n_aux).astype(np.float32)
        aux_flags = np.ones(n_aux, dtype=bool)

        # Occasional unflagged background dark-noise hits (realistic IceCube non-aux noise)
        n_unflagged = int(rng.integers(1, 5))
        unf_idx = rng.integers(0, dom_xyz.shape[0], size=n_unflagged, dtype=np.int32)
        t_unf = rng.uniform(t0 - 2500.0, t0 + 2500.0, size=n_unflagged).astype(np.float32)
        q_unf = rng.uniform(0.3, 1.1, size=n_unflagged).astype(np.float32)
        unf_flags = np.zeros(n_unflagged, dtype=bool)

        sensor_ids = np.concatenate([sig_idx, aux_idx, unf_idx])
        times = np.concatenate([t_sig, t_aux, t_unf])
        charges = np.concatenate([q_sig, q_aux, q_unf])
        auxiliary = np.concatenate([aux_sig, aux_flags, unf_flags])
    else:
        sensor_ids = sig_idx
        times = t_sig
        charges = q_sig
        auxiliary = aux_sig

    order = np.argsort(times, kind="mergesort")
    return {
        "sensor_id": sensor_ids[order].astype(np.int32),
        "time": times[order].astype(np.float32),
        "charge": charges[order].astype(np.float32),
        "auxiliary": auxiliary[order].astype(bool),
        "azimuth": float(azimuth),
        "zenith": float(zenith),
        "direction_vec": u,
        "vertex": vertex,
    }
