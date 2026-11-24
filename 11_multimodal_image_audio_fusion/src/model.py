"""Multimodal encoders and 4 ablation variants (ImageOnly, AudioOnly, EarlyConcat, GatedCrossAttention)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock2d(nn.Module):
    """Lightweight 2D residual block suitable for 4 GB VRAM training."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.05) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.gelu(out + identity)


class ImageEncoder(nn.Module):
    """Residual 2D CNN on 32x32 RGB images -> 128-d embedding + spatial tokens (B, 16, 128)."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.stage1 = ResidualBlock2d(32, 48, stride=2)   # 16x16
        self.stage2 = ResidualBlock2d(48, 96, stride=2)   # 8x8
        self.stage3 = ResidualBlock2d(96, embed_dim, stride=2)  # 4x4 -> 16 tokens
        self.token_norm = nn.LayerNorm(embed_dim)

    def forward_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)  # (B, D, 4, 4)
        tokens = feat.flatten(2).transpose(1, 2)  # (B, 16, D)
        tokens = self.token_norm(tokens)
        pooled = tokens.mean(dim=1)
        return pooled, tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.forward_tokens(x)
        return pooled


class AudioEncoder(nn.Module):
    """Residual 2D CNN on (1, 64, 96) Log-Mel spectrograms -> 128-d embedding + 24 TF tokens."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False),  # 32x48
            nn.BatchNorm2d(24),
            nn.GELU(),
        )
        self.stage1 = ResidualBlock2d(24, 48, stride=2)         # 16x24
        self.stage2 = ResidualBlock2d(48, 96, stride=2)         # 8x12
        self.stage3 = ResidualBlock2d(96, embed_dim, stride=2)  # 4x6 -> 24 tokens
        self.token_norm = nn.LayerNorm(embed_dim)

    def forward_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)  # (B, D, 4, 6)
        tokens = feat.flatten(2).transpose(1, 2)  # (B, 24, D)
        tokens = self.token_norm(tokens)
        pooled = tokens.mean(dim=1)
        return pooled, tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.forward_tokens(x)
        return pooled


class ImageOnlyNet(nn.Module):
    """Unimodal visual baseline."""

    def __init__(self, num_classes: int = 7, embed_dim: int = 128) -> None:
        super().__init__()
        self.encoder = ImageEncoder(embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(
        self, image: torch.Tensor, audio: torch.Tensor, return_details: bool = False
    ) -> torch.Tensor | dict[str, Any]:
        del audio
        z_img = self.encoder(image)
        logits = self.head(z_img)
        if return_details:
            alpha = torch.ones((image.size(0), 1), device=image.device, dtype=logits.dtype)
            return {"logits": logits, "alpha": alpha}
        return logits


class AudioOnlyNet(nn.Module):
    """Unimodal acoustic baseline."""

    def __init__(self, num_classes: int = 7, embed_dim: int = 128) -> None:
        super().__init__()
        self.encoder = AudioEncoder(embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(
        self, image: torch.Tensor, audio: torch.Tensor, return_details: bool = False
    ) -> torch.Tensor | dict[str, Any]:
        del image
        z_aud = self.encoder(audio)
        logits = self.head(z_aud)
        if return_details:
            alpha = torch.zeros((audio.size(0), 1), device=audio.device, dtype=logits.dtype)
            return {"logits": logits, "alpha": alpha}
        return logits


class EarlyConcatFusionNet(nn.Module):
    """Naive feature concatenation multimodal baseline."""

    def __init__(self, num_classes: int = 7, embed_dim: int = 128) -> None:
        super().__init__()
        self.img_encoder = ImageEncoder(embed_dim=embed_dim)
        self.aud_encoder = AudioEncoder(embed_dim=embed_dim)
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(0.25),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(
        self, image: torch.Tensor, audio: torch.Tensor, return_details: bool = False
    ) -> torch.Tensor | dict[str, Any]:
        z_img = self.img_encoder(image)
        z_aud = self.aud_encoder(audio)
        fused = torch.cat([z_img, z_aud], dim=-1)
        logits = self.fusion_head(fused)
        if return_details:
            alpha = torch.full((image.size(0), 1), 0.5, device=image.device, dtype=logits.dtype)
            return {"logits": logits, "alpha": alpha}
        return logits


class GatedCrossAttentionFusionNet(nn.Module):
    """Bidirectional Multi-Head Cross-Attention + Learned Per-Sample Modality Gate alpha in (0, 1).

    During training, stochastic modality corruption/dropout teaches the reliability gate alpha to
    dynamically weight the cleaner modality while auxiliary unimodal heads prevent gradient starvation.
    """

    def __init__(
        self,
        num_classes: int = 7,
        embed_dim: int = 128,
        num_heads: int = 4,
        modality_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.modality_dropout = float(modality_dropout)

        self.img_encoder = ImageEncoder(embed_dim=embed_dim)
        self.aud_encoder = AudioEncoder(embed_dim=embed_dim)

        # Bidirectional cross-modal multi-head attention.
        self.img_to_aud_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.aud_to_img_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.img_cross_norm = nn.LayerNorm(embed_dim)
        self.aud_cross_norm = nn.LayerNorm(embed_dim)

        # Per-modality reliability scorers + confidence-aware fusion gate.
        self.img_aux_head = nn.Linear(embed_dim, num_classes)
        self.aud_aux_head = nn.Linear(embed_dim, num_classes)

        # Gate receives pooled embeddings, interaction features, and unimodal entropy/confidence cues.
        gate_in_dim = embed_dim * 4 + 4
        self.gate_net = nn.Sequential(
            nn.LayerNorm(gate_in_dim),
            nn.Linear(gate_in_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes),
        )

    def _apply_train_modality_dropout(
        self, image: torch.Tensor, audio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly corrupt a subset of samples in one modality during training for gate calibration."""
        bsz = image.size(0)
        device = image.device
        # 0 = both clean, 1 = visual degraded, 2 = audio degraded
        drop_state = torch.zeros(bsz, dtype=torch.long, device=device)
        if (not self.training) or self.modality_dropout <= 0.0:
            return image, audio, drop_state

        img_out = image.clone()
        aud_out = audio.clone()
        probs = torch.rand(bsz, device=device)
        half_p = self.modality_dropout * 0.5

        vis_mask = probs < half_p
        aud_mask = (probs >= half_p) & (probs < self.modality_dropout)

        if vis_mask.any():
            noise = torch.randn_like(img_out[vis_mask]) * 1.2
            img_out[vis_mask] = img_out[vis_mask] * 0.1 + noise
            drop_state[vis_mask] = 1

        if aud_mask.any():
            noise = torch.randn_like(aud_out[aud_mask]) * 1.8
            aud_out[aud_mask] = aud_out[aud_mask] * 0.1 + noise
            drop_state[aud_mask] = 2

        return img_out, aud_out, drop_state

    @staticmethod
    def _modality_confidence_stats(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits.float(), dim=-1)
        max_conf = probs.max(dim=-1, keepdim=True).values
        entropy = -(probs * torch.log(probs.clamp_min(1e-7))).sum(dim=-1, keepdim=True)
        norm_entropy = entropy / torch.log(
            torch.tensor(float(logits.size(-1)), device=logits.device, dtype=torch.float32)
        )
        return max_conf.to(logits.dtype), norm_entropy.to(logits.dtype)

    def forward(
        self, image: torch.Tensor, audio: torch.Tensor, return_details: bool = False
    ) -> torch.Tensor | dict[str, Any]:
        img_in, aud_in, drop_state = self._apply_train_modality_dropout(image, audio)

        z_img, img_tokens = self.img_encoder.forward_tokens(img_in)
        z_aud, aud_tokens = self.aud_encoder.forward_tokens(aud_in)

        # Unimodal predictions supply explicit reliability cues to the modality gate.
        img_aux_logits = self.img_aux_head(z_img)
        aud_aux_logits = self.aud_aux_head(z_aud)

        img_conf, img_ent = self._modality_confidence_stats(img_aux_logits.detach())
        aud_conf, aud_ent = self._modality_confidence_stats(aud_aux_logits.detach())

        # Bidirectional cross-modal attention across spatial RGB and time-frequency audio tokens.
        img_attended, img_attn_weights = self.img_to_aud_attn(
            query=img_tokens, key=aud_tokens, value=aud_tokens, need_weights=True
        )
        aud_attended, aud_attn_weights = self.aud_to_img_attn(
            query=aud_tokens, key=img_tokens, value=img_tokens, need_weights=True
        )

        h_img = self.img_cross_norm(img_tokens + 0.5 * img_attended).mean(dim=1)
        h_aud = self.aud_cross_norm(aud_tokens + 0.5 * aud_attended).mean(dim=1)

        gate_features = torch.cat(
            [
                z_img,
                z_aud,
                h_img * h_aud,
                torch.abs(h_img - h_aud),
                img_conf,
                img_ent,
                aud_conf,
                aud_ent,
            ],
            dim=-1,
        )
        alpha = torch.sigmoid(self.gate_net(gate_features))  # (B, 1) in (0, 1)

        # Combine modality-specific and cross-attended representations under adaptive gate alpha.
        visual_branch = 0.65 * z_img + 0.35 * h_img
        audio_branch = 0.65 * z_aud + 0.35 * h_aud
        fused = alpha * visual_branch + (1.0 - alpha) * audio_branch
        logits = self.classifier(fused)

        if return_details:
            return {
                "logits": logits,
                "alpha": alpha,
                "img_aux_logits": img_aux_logits,
                "aud_aux_logits": aud_aux_logits,
                "drop_state": drop_state,
                "img_to_aud_attn": img_attn_weights,
                "aud_to_img_attn": aud_attn_weights,
            }
        return logits


def build_model(variant: str, num_classes: int = 7, embed_dim: int = 128) -> nn.Module:
    """Factory for the 4 multimodal ablation architectures."""
    registry = {
        "image_only": lambda: ImageOnlyNet(num_classes=num_classes, embed_dim=embed_dim),
        "audio_only": lambda: AudioOnlyNet(num_classes=num_classes, embed_dim=embed_dim),
        "early_concat": lambda: EarlyConcatFusionNet(num_classes=num_classes, embed_dim=embed_dim),
        "gated_cross_attention": lambda: GatedCrossAttentionFusionNet(
            num_classes=num_classes, embed_dim=embed_dim
        ),
    }
    if variant not in registry:
        raise ValueError(f"Unknown model variant '{variant}'. Valid: {sorted(registry)}")
    return registry[variant]()
