"""ResNet-18 U-Net architecture, Focal+Dice loss, and ISO 4628-3 corrosion severity grading."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

SEVERITY_GRADES = ("Clean", "Light", "Moderate", "Severe")


def compute_corrosion_area_pct(mask: np.ndarray) -> float:
    """Compute corroded pixel area percentage in [0.0, 100.0]."""
    arr = (np.asarray(mask) > 0).astype(np.float32)
    if arr.size == 0:
        return 0.0
    return round(float(arr.mean() * 100.0), 3)


def grade_corrosion_severity(area_pct: float) -> str:
    """ISO 4628-3 inspired rust severity grading from corroded area percentage."""
    pct = float(area_pct)
    if pct < 1.0:
        return "Clean"
    if pct < 5.0:
        return "Light"
    if pct < 20.0:
        return "Moderate"
    return "Severe"


class DecoderBlock(nn.Module):
    """Upsample + skip concatenation + two 3x3 Conv-BN-ReLU layers."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


class ResNet18UNet(nn.Module):
    """U-Net decoder backed by a ResNet-18 encoder skipping conv1 and layer1..layer4."""

    def __init__(self, out_channels: int = 1, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64, H/2
        self.maxpool = backbone.maxpool
        self.enc1 = backbone.layer1  # 64, H/4
        self.enc2 = backbone.layer2  # 128, H/8
        self.enc3 = backbone.layer3  # 256, H/16
        self.enc4 = backbone.layer4  # 512, H/32

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 48)
        self.final_refine = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[2:]
        s0 = self.stem(x)
        s1 = self.enc1(self.maxpool(s0))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        d4 = self.dec4(s4, s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)
        up = F.interpolate(d1, size=input_hw, mode="bilinear", align_corners=False)
        return self.final_refine(up)


class FocalDiceLoss(nn.Module):
    """Combined Binary Focal Loss (from logits) + Soft Dice Loss for imbalanced rust masks."""

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        focal_weight: float = 0.4,
        dice_weight: float = 0.6,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_f = logits.float()
        targets_f = targets.float()

        bce = F.binary_cross_entropy_with_logits(logits_f, targets_f, reduction="none")
        probs = torch.sigmoid(logits_f)
        p_t = probs * targets_f + (1.0 - probs) * (1.0 - targets_f)
        alpha_t = self.alpha * targets_f + (1.0 - self.alpha) * (1.0 - targets_f)
        focal = (alpha_t * ((1.0 - p_t).clamp(min=1e-6) ** self.gamma) * bce).mean()

        dims = (1, 2, 3)
        inter = (probs * targets_f).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets_f.sum(dim=dims)
        dice_loss = 1.0 - ((2.0 * inter + self.smooth) / (denom + self.smooth)).mean()

        return self.focal_weight * focal + self.dice_weight * dice_loss
