"""Classical IceCube Charge-Weighted LineFit analytical track reconstruction baseline."""
from __future__ import annotations

import numpy as np


def linefit_reconstruct_event(
    pulse_features: np.ndarray,
    valid_mask: np.ndarray,
    charge_power: float = 0.5,
) -> np.ndarray:
    """Analytical IceCube LineFit estimator: r(t) = r0 + v * t -> u_hat = v / ||v||.

    Args:
        pulse_features: (K, 6) normalized pulse features [x/500, y/500, z/500, rel_t/1000, log1p(q), aux]
        valid_mask: (K,) bool mask of real (non-padded) pulses
        charge_power: exponent applied to photoelectron charge for weighting
    Returns:
        (3,) float32 unit direction vector
    """
    valid = np.asarray(valid_mask, dtype=bool)
    feats = np.asarray(pulse_features, dtype=np.float64)

    # Prefer non-auxiliary signal pulses; fall back to all valid pulses if < 2 non-aux
    non_aux = valid & (feats[:, 5] < 0.5)
    sel = non_aux if int(np.sum(non_aux)) >= 2 else valid
    if int(np.sum(sel)) < 2:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    r = feats[sel, 0:3] * 500.0  # meters
    t = feats[sel, 3] * 1000.0  # nanoseconds
    q = np.maximum(np.expm1(feats[sel, 4]), 1e-3)
    w = np.power(q, charge_power)

    # One-pass MAD temporal outlier filter to suppress far unflagged dark-noise pulses
    if t.size >= 6:
        t_med = float(np.median(t))
        mad = float(np.median(np.abs(t - t_med))) + 25.0
        inliers = np.abs(t - t_med) <= 3.5 * mad
        if int(np.sum(inliers)) >= 3:
            r, t, w = r[inliers], t[inliers], w[inliers]

    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    r_mean = np.sum(w[:, None] * r, axis=0) / w_sum
    t_mean = float(np.sum(w * t)) / w_sum

    dt = t - t_mean
    denom = float(np.sum(w * dt * dt))
    if denom <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    v = np.sum((w * dt)[:, None] * (r - r_mean[None, :]), axis=0) / denom
    v_norm = float(np.linalg.norm(v))
    if v_norm <= 1e-12 or not np.isfinite(v_norm):
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (v / v_norm).astype(np.float32)


def linefit_predict_batch(
    X: np.ndarray,
    masks: np.ndarray,
    charge_power: float = 0.5,
) -> np.ndarray:
    """Run Classical Charge-Weighted LineFit across an entire batch of events."""
    n = X.shape[0]
    preds = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        preds[i] = linefit_reconstruct_event(X[i], masks[i], charge_power=charge_power)
    return preds
