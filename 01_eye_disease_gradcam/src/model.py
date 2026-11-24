"""ResNet-18 classifier with regularized dropout head for retinal disease grading."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_eye_resnet18(
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Build a ResNet-18 backbone adapted to `num_classes` with a dropout classifier head."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return the final residual block (`layer4[-1]`) for Grad-CAM saliency."""
    if not hasattr(model, "layer4"):
        raise AttributeError("Expected a ResNet model exposing `layer4`.")
    return model.layer4[-1]
