"""Grad-CAM implementation using PyTorch forward and full-backward hooks."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Compute class-discriminative Grad-CAM heatmaps from a target conv/residual layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _backward_hook(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        if grad_output and grad_output[0] is not None:
            self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_hooks()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Generate a normalized [0, 1] Grad-CAM heatmap matching input spatial resolution.

        Args:
            input_tensor: Single image tensor of shape (1, C, H, W) or (C, H, W).
            target_class: Optional class index; defaults to argmax predicted class.

        Returns:
            Tuple of (heatmap [H, W] float32 in [0, 1], pred_class int, class_probs [K]).
        """
        if input_tensor.ndim == 3:
            x = input_tensor.unsqueeze(0)
        elif input_tensor.ndim == 4 and input_tensor.shape[0] == 1:
            x = input_tensor
        else:
            raise ValueError(f"Expected (C, H, W) or (1, C, H, W) tensor, got {tuple(input_tensor.shape)}")

        was_training = self.model.training
        self.model.eval()
        self._activations = None
        self._gradients = None

        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x_req)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred_class = int(np.argmax(probs))
            cls_idx = pred_class if target_class is None else int(target_class)

            score = logits[0, cls_idx]
            score.backward()

        if was_training:
            self.model.train()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # Global average pool gradients over spatial dims -> channel importance weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        heatmap = cam[0, 0].cpu().numpy().astype(np.float32)
        h_min, h_max = float(heatmap.min()), float(heatmap.max())
        if h_max - h_min > 1e-8:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = np.zeros_like(heatmap, dtype=np.float32)
        return np.clip(heatmap, 0.0, 1.0), pred_class, probs


def overlay_gradcam(
    rgb_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a [0, 1] Grad-CAM heatmap with a uint8 RGB image using a Jet colormap."""
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got {rgb_image.shape}")
    base = rgb_image.astype(np.uint8)
    hm = np.clip(heatmap, 0.0, 1.0)
    if hm.shape[:2] != base.shape[:2]:
        hm = cv2.resize(hm.astype(np.float32), (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    hm_u8 = np.round(hm * 255.0).astype(np.uint8)
    jet_bgr = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)
    blended = (1.0 - alpha) * base.astype(np.float32) + alpha * jet_rgb.astype(np.float32)
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)
