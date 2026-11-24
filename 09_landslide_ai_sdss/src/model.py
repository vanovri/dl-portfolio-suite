"""SpatialResUNet architecture and class-imbalance-aware FocalDiceLoss for landslide mapping."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm, GELU, and 1x1 residual skip connection."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.act(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return self.act(h + residual)


class SpatialResUNet(nn.Module):
    """Dynamic multi-band Residual U-Net predicting per-pixel landslide susceptibility logits."""

    def __init__(self, in_channels: int = 14, base_channels: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_channels = in_channels
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        self.enc1 = ResidualConvBlock(in_channels, c1)
        self.enc2 = ResidualConvBlock(c1, c2)
        self.enc3 = ResidualConvBlock(c2, c3, dropout=dropout)
        self.bottleneck = ResidualConvBlock(c3, c4, dropout=dropout)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ResidualConvBlock(c3 * 2, c3, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ResidualConvBlock(c2 * 2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ResidualConvBlock(c1 * 2, c1)

        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    @staticmethod
    def _align_and_cat(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([up, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(self._align_and_cat(self.up3(b), e3))
        d2 = self.dec2(self._align_and_cat(self.up2(d3), e2))
        d1 = self.dec1(self._align_and_cat(self.up1(d2), e1))
        return self.head(d1)


class FocalDiceLoss(nn.Module):
    """Combined class-weighted Binary Focal Loss + Soft Dice Loss (FP32 stable under AMP)."""

    def __init__(
        self,
        pos_weight: float = 4.0,
        gamma: float = 2.0,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.gamma = float(gamma)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_f = logits.float()
        targets_f = targets.float()
        pw = torch.tensor(self.pos_weight, dtype=torch.float32, device=logits.device)

        bce = F.binary_cross_entropy_with_logits(logits_f, targets_f, pos_weight=pw, reduction="none")
        probs = torch.sigmoid(logits_f)
        pt = torch.where(targets_f > 0.5, probs, 1.0 - probs)
        focal = ((1.0 - pt).clamp(min=1e-6) ** self.gamma) * bce
        focal_loss = focal.mean()

        inter = (probs * targets_f).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets_f.sum(dim=(1, 2, 3))
        dice_loss = (1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)).mean()

        return (1.0 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss
