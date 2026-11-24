"""3D von Mises-Fisher (vMF) spherical loss and exact Kaggle IceCube angular error metrics."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VonMisesFisher3DLoss(nn.Module):
    """Exact 3D von Mises-Fisher negative log-likelihood on S^2 (FP32 numerically stable).

    Density on S^2: p(u | mu, kappa) = C_3(kappa) * exp(kappa * mu^T u),
    where C_3(kappa) = kappa / (4 * pi * sinh(kappa)).

    Negative log-likelihood (dropping constant log(4*pi)):
        L = -kappa * (mu^T u) + log(sinh(kappa) / kappa)
    """

    def __init__(self, max_kappa: float = 100.0) -> None:
        super().__init__()
        self.max_kappa = float(max_kappa)

    def forward(
        self,
        pred_unit_vec: torch.Tensor,
        kappa: torch.Tensor,
        target_unit_vec: torch.Tensor,
    ) -> torch.Tensor:
        pred_u = F.normalize(pred_unit_vec.float(), dim=-1, eps=1e-7)
        targ_u = F.normalize(target_unit_vec.float(), dim=-1, eps=1e-7)
        k = kappa.float().reshape(-1).clamp(min=1e-5, max=self.max_kappa)

        cos_sim = (pred_u * targ_u).sum(dim=-1).clamp(-1.0, 1.0)

        # Stable log(sinh(k) / k) across all k in [1e-5, max_kappa]:
        #   - Small k (< 1e-2): Taylor series k^2/6 - k^4/180
        #   - Larger k: k + log((1 - exp(-2k)) / (2k))
        small = k < 1e-2
        k_safe = torch.where(small, torch.ones_like(k), k)
        exact_term = k_safe + torch.log((-torch.expm1(-2.0 * k_safe)).clamp(min=1e-12) / (2.0 * k_safe))
        taylor_term = (k * k) / 6.0 - (k ** 4) / 180.0
        log_sinh_over_k = torch.where(small, taylor_term, exact_term)

        nll = -k * cos_sim + log_sinh_over_k
        return nll.mean()


def angular_error_radians(pred_vec: np.ndarray, target_vec: np.ndarray) -> np.ndarray:
    """Exact Kaggle IceCube per-event angular error arccos(clip(u_hat . u, -1, 1)) in radians."""
    p = np.asarray(pred_vec, dtype=np.float64)
    t = np.asarray(target_vec, dtype=np.float64)
    p_u = p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-12)
    t_u = t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-12)
    dot = np.clip(np.sum(p_u * t_u, axis=-1), -1.0, 1.0)
    return np.arccos(dot).astype(np.float64)


def compute_angular_metrics(pred_vec: np.ndarray, target_vec: np.ndarray) -> dict[str, Any]:
    """Compute Mean Angular Error (rad & deg), Median Angular Error (deg), and 68%/90% containment."""
    err_rad = angular_error_radians(pred_vec, target_vec)
    err_deg = np.degrees(err_rad)
    return {
        "mean_angular_error_rad": round(float(np.mean(err_rad)), 4),
        "mean_angular_error_deg": round(float(np.mean(err_deg)), 3),
        "median_angular_error_deg": round(float(np.median(err_deg)), 3),
        "containment_68_deg": round(float(np.percentile(err_deg, 68.0)), 3),
        "containment_90_deg": round(float(np.percentile(err_deg, 90.0)), 3),
    }
