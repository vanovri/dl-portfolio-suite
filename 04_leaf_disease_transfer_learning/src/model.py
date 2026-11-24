"""Model factory for EfficientNet-B0 vs MobileNetV3-Small head-to-head comparison."""
from __future__ import annotations

from pathlib import Path

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
)

SUPPORTED_ARCHS = ("efficientnet_b0", "mobilenet_v3_small")


def build_leaf_model(
    arch: str,
    num_classes: int = 10,
    pretrained: bool = True,
) -> nn.Module:
    """Instantiate an EfficientNet-B0 or MobileNetV3-Small classifier for leaf diseases."""
    arch_key = arch.lower().strip()
    if arch_key == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if arch_key == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported architecture '{arch}'. Choose from {SUPPORTED_ARCHS}.")


def checkpoint_size_mb(path: str | Path) -> float:
    """Return checkpoint file size in megabytes."""
    p = Path(path)
    if not p.exists():
        return 0.0
    return round(p.stat().st_size / (1024.0 * 1024.0), 2)
