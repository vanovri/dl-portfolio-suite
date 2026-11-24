"""IceCubePulseTransformer with sinusoidal spacetime Fourier encoding and 3D vMF output head."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalFourierEncoding(nn.Module):
    """Multi-scale sinusoidal Fourier positional encoding for continuous 4D spacetime (x, y, z, t)."""

    def __init__(self, in_dims: int = 4, num_bands: int = 4) -> None:
        super().__init__()
        freqs = torch.tensor([float(2 ** i) * math.pi for i in range(num_bands)], dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)
        self.out_dims = in_dims * num_bands * 2

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, K, in_dims) -> angles: (B, K, in_dims, num_bands)
        angles = coords.unsqueeze(-1) * self.freqs
        enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return enc.flatten(start_dim=2)


class IceCubePulseTransformer(nn.Module):
    """4-layer Pre-Norm Pulse Transformer predicting 3D unit neutrino vector + vMF concentration."""

    def __init__(
        self,
        pulse_dim: int = 6,
        fourier_bands: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fourier = SinusoidalFourierEncoding(in_dims=4, num_bands=fourier_bands)
        in_total = pulse_dim + self.fourier.out_dims

        self.input_proj = nn.Sequential(
            nn.Linear(in_total, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.pool_norm = nn.LayerNorm(d_model)
        self.attn_score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1, bias=False),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 4),  # (ux, uy, uz, raw_kappa)
        )

    def forward(
        self,
        pulses: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            pulses: (B, K, 6) normalized pulse features
            valid_mask: (B, K) bool tensor (True = real pulse, False = pad)
        Returns:
            unit_vec: (B, 3) normalized 3D neutrino direction vector
            kappa: (B,) positive vMF concentration parameter
        """
        mask = valid_mask.bool()
        # Defensively prevent all-padded rows from causing attention NaNs
        empty_rows = ~mask.any(dim=1)
        if empty_rows.any():
            mask = mask.clone()
            mask[empty_rows, 0] = True

        fourier_feats = self.fourier(pulses[:, :, :4])
        x = torch.cat([pulses, fourier_feats], dim=-1)
        h = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=~mask)
        h = self.pool_norm(h)

        # Compute masked attention pooling in FP32 for complete AMP numerical safety
        scores = self.attn_score(h).squeeze(-1).float()
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1).to(h.dtype)
        pooled = torch.bmm(weights.unsqueeze(1), h).squeeze(1)

        out = self.head(pooled).float()
        unit_vec = F.normalize(out[:, :3], dim=-1, eps=1e-7)
        kappa = F.softplus(out[:, 3]).clamp(min=1e-2, max=100.0)
        return unit_vec, kappa
