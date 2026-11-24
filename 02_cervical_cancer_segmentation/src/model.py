"""Plain U-Net, Additive Attention U-Net, and Dice+BCE+Sobel Boundary Loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two consecutive 3x3 Conv-BatchNorm-ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AdditiveAttentionGate(nn.Module):
    """Oktay et al. (2018) Additive Attention Gate for U-Net skip connections."""

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        alpha = self.psi(self.relu(g1 + x1))
        return x * alpha


class PlainUNet(nn.Module):
    """4-stage baseline U-Net without skip-connection attention gates."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.bottleneck = ConvBlock(c4, c5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(c4 * 2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 * 2, c1)

        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    @staticmethod
    def _match_and_concat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if upsampled.shape[2:] != skip.shape[2:]:
            upsampled = F.interpolate(upsampled, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, upsampled], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(self._match_and_concat(self.up4(b), e4))
        d3 = self.dec3(self._match_and_concat(self.up3(d4), e3))
        d2 = self.dec2(self._match_and_concat(self.up2(d3), e2))
        d1 = self.dec1(self._match_and_concat(self.up1(d2), e1))
        return self.head(d1)


class AttentionUNet(nn.Module):
    """4-stage Attention U-Net using Additive Attention Gates on all 4 skip connections."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.bottleneck = ConvBlock(c4, c5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.att4 = AdditiveAttentionGate(F_g=c4, F_l=c4, F_int=c4 // 2)
        self.dec4 = ConvBlock(c4 * 2, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.att3 = AdditiveAttentionGate(F_g=c3, F_l=c3, F_int=c3 // 2)
        self.dec3 = ConvBlock(c3 * 2, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.att2 = AdditiveAttentionGate(F_g=c2, F_l=c2, F_int=c2 // 2)
        self.dec2 = ConvBlock(c2 * 2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.att1 = AdditiveAttentionGate(F_g=c1, F_l=c1, F_int=max(c1 // 2, 8))
        self.dec1 = ConvBlock(c1 * 2, c1)

        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    @staticmethod
    def _match_and_concat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if upsampled.shape[2:] != skip.shape[2:]:
            upsampled = F.interpolate(upsampled, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, upsampled], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        g4 = self.up4(b)
        d4 = self.dec4(self._match_and_concat(g4, self.att4(g4, e4)))

        g3 = self.up3(d4)
        d3 = self.dec3(self._match_and_concat(g3, self.att3(g3, e3)))

        g2 = self.up2(d3)
        d2 = self.dec2(self._match_and_concat(g2, self.att2(g2, e2)))

        g1 = self.up1(d2)
        d1 = self.dec1(self._match_and_concat(g1, self.att1(g1, e1)))
        return self.head(d1)


class DiceBCEBoundaryLoss(nn.Module):
    """Combined BCEWithLogits + Soft Dice + Sobel Boundary Edge Loss (FP16 AMP safe)."""

    def __init__(
        self,
        bce_weight: float = 0.4,
        dice_weight: float = 0.4,
        boundary_weight: float = 0.2,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.smooth = smooth

        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _edge_magnitude(self, mask: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(mask, self.sobel_x, padding=1)
        gy = F.conv2d(mask, self.sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_f = logits.float()
        targets_f = targets.float()

        bce = F.binary_cross_entropy_with_logits(logits_f, targets_f)

        probs = torch.sigmoid(logits_f)
        dims = (1, 2, 3)
        inter = (probs * targets_f).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets_f.sum(dim=dims)
        dice_loss = 1.0 - ((2.0 * inter + self.smooth) / (denom + self.smooth)).mean()

        pred_edges = self._edge_magnitude(probs)
        target_edges = self._edge_magnitude(targets_f)
        boundary_loss = F.l1_loss(pred_edges, target_edges)

        return (
            self.bce_weight * bce
            + self.dice_weight * dice_loss
            + self.boundary_weight * boundary_loss
        )
