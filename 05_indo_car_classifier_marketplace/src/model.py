"""EfficientNet-B0 car model classifier and inference wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_car_transforms(img_size: int = 160, is_train: bool = False) -> transforms.Compose:
    """Build RGB transforms for vehicle exterior classification."""
    ops: list[Any] = [transforms.Resize((img_size, img_size))]
    if is_train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.92, 1.08)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ]
        )
    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)


def build_car_classifier(
    num_classes: int = 10,
    pretrained: bool = True,
    dropout: float = 0.25,
) -> nn.Module:
    """Build an EfficientNet-B0 vehicle classifier."""
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


class CarPredictor:
    """Stateful inference wrapper returning top-k car class predictions from PIL images."""

    def __init__(
        self,
        model: nn.Module,
        class_names: list[str],
        img_size: int = 160,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.class_names = list(class_names)
        self.img_size = img_size
        self.transform = build_car_transforms(img_size=img_size, is_train=False)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path, device: torch.device | str = "cpu") -> "CarPredictor":
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        class_names = list(state["class_names"])
        img_size = int(state.get("img_size", 160))
        model = build_car_classifier(num_classes=len(class_names), pretrained=False)
        model.load_state_dict(state["model_state_dict"])
        return cls(model=model, class_names=class_names, img_size=img_size, device=device)

    @torch.no_grad()
    def predict_image(self, image: Image.Image, top_k: int = 3) -> list[dict[str, Any]]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits.float(), dim=1)[0].cpu().numpy()
        k = min(top_k, len(self.class_names))
        top_idx = np.argsort(-probs)[:k]
        return [
            {
                "class_id": int(idx),
                "class_name": self.class_names[int(idx)],
                "confidence": round(float(probs[int(idx)]), 4),
            }
            for idx in top_idx
        ]
