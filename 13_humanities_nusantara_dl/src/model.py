"""Dual-Scale Humanities Neural Architecture: Multi-Scale Char-Phonetic CNN + Word BiGRU Self-Attention."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveSelfAttention(nn.Module):
    """Interpretable Bahdanau-style additive self-attention over word sequence hidden states."""

    def __init__(self, hidden_dim: int, attn_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return context vector (B, H) and normalized token attention weights (B, L)."""
        scores = self.v(torch.tanh(self.proj(hidden_states))).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(~mask, -1e4)
        weights = F.softmax(scores, dim=-1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights


class CharPhoneticCNN(nn.Module):
    """Multi-scale 1D CNN (kernels 3, 5, 7) capturing suffix rhymes, archaic spelling, and cadence."""

    def __init__(
        self,
        char_vocab_size: int,
        char_embed_dim: int = 32,
        num_filters: int = 48,
        out_dim: int = 128,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(char_embed_dim, num_filters, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(num_filters),
                    nn.GELU(),
                )
                for k in (3, 5, 7)
            ]
        )
        conv_total = num_filters * 3
        self.line_proj = nn.Sequential(
            nn.Linear(conv_total, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        # Global max + avg pool (2 * conv_total) + 6 pairwise cross-line/stanza phonetic similarities
        self.proj = nn.Sequential(
            nn.Linear(conv_total * 2 + 6, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def _encode_seq(self, char_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(char_ids).transpose(1, 2)  # (B, E, L)
        feats = [conv(x) for conv in self.convs]
        return torch.cat(feats, dim=1)  # (B, 3*F, L)

    def _encode_line_suffixes(self, line_suffix_chars: torch.Tensor) -> torch.Tensor:
        """Encode (B, 4, L_suf) line-ending characters and compute 6 pairwise rhyme similarities."""
        bsz, n_lines, suf_len = line_suffix_chars.shape
        flat = line_suffix_chars.reshape(bsz * n_lines, suf_len)
        seq = self._encode_seq(flat)  # (B*4, 3*F, L_suf)
        pooled = seq.max(dim=-1).values
        line_vec = F.normalize(self.line_proj(pooled).view(bsz, n_lines, -1), dim=-1)
        pairs = ((0, 2), (1, 3), (0, 1), (2, 3), (0, 3), (1, 2))
        sims = [(line_vec[:, i] * line_vec[:, j]).sum(dim=-1) for i, j in pairs]
        return torch.stack(sims, dim=-1)  # (B, 6)

    def forward(
        self, char_ids: torch.Tensor, line_suffix_chars: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self._encode_seq(char_ids)  # (B, 3*F, L_c)
        max_pool = seq.max(dim=-1).values
        avg_pool = seq.mean(dim=-1)

        if line_suffix_chars is not None:
            rhyme_sims = self._encode_line_suffixes(line_suffix_chars)
        else:
            # Split character sequence into 4 stanza quarters when explicit line suffixes are omitted.
            chunks = torch.chunk(seq, 4, dim=-1)
            q_vecs = [
                F.normalize(self.line_proj(ch.max(dim=-1).values), dim=-1)
                for ch in chunks
            ]
            while len(q_vecs) < 4:
                q_vecs.append(q_vecs[-1])
            pairs = ((0, 2), (1, 3), (0, 1), (2, 3), (0, 3), (1, 2))
            rhyme_sims = torch.stack(
                [(q_vecs[i] * q_vecs[j]).sum(dim=-1) for i, j in pairs], dim=-1
            )

        combined = torch.cat([max_pool, avg_pool, rhyme_sims], dim=-1)
        return self.proj(combined), rhyme_sims


class DualScaleHumanitiesNet(nn.Module):
    """Combines Multi-Scale Char-Phonetic CNN + Word BiGRU + Additive Self-Attention for close reading."""

    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int,
        num_classes: int,
        word_embed_dim: int = 64,
        char_embed_dim: int = 32,
        gru_hidden: int = 64,
        char_out_dim: int = 128,
        dropout: float = 0.25,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = int(pad_idx)
        self.num_classes = int(num_classes)

        self.word_embed = nn.Embedding(word_vocab_size, word_embed_dim, padding_idx=pad_idx)
        self.word_drop = nn.Dropout(dropout)
        self.bigru = nn.GRU(
            input_size=word_embed_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.word_attn = AdditiveSelfAttention(hidden_dim=gru_hidden * 2, attn_dim=64)
        self.word_norm = nn.LayerNorm(gru_hidden * 2)

        self.char_cnn = CharPhoneticCNN(
            char_vocab_size=char_vocab_size,
            char_embed_dim=char_embed_dim,
            num_filters=48,
            out_dim=char_out_dim,
            pad_idx=pad_idx,
        )

        fused_dim = gru_hidden * 2 + char_out_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        line_suffix_chars: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        word_mask = word_ids.ne(self.pad_idx)
        # Ensure at least one token is unmasked to avoid NaN softmax on empty strings.
        empty_rows = ~word_mask.any(dim=1)
        if empty_rows.any():
            word_mask = word_mask.clone()
            word_mask[empty_rows, 0] = True

        w_emb = self.word_drop(self.word_embed(word_ids))
        gru_out, _ = self.bigru(w_emb)
        word_ctx, attn_weights = self.word_attn(gru_out, word_mask)
        word_ctx = self.word_norm(word_ctx)

        char_ctx, rhyme_sims = self.char_cnn(char_ids, line_suffix_chars=line_suffix_chars)

        fused = torch.cat([word_ctx, char_ctx], dim=-1)
        logits = self.classifier(fused)

        if return_details:
            return {
                "logits": logits,
                "token_attention": attn_weights,
                "rhyme_similarities": rhyme_sims,
            }
        return logits
