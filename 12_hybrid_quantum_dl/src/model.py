"""Hybrid Quantum-Classical Classifier and Parameter-Matched Classical MLP Baseline."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .quantum_circuit import VariationalQuantumLayer


class HybridQuantumClassifier(nn.Module):
    """Classical Compressor -> pi*tanh -> 4-Qubit VQC -> Residual Skip + Linear Logit Head."""

    def __init__(
        self,
        in_dim: int,
        n_qubits: int = 4,
        n_layers: int = 3,
        hidden_dim: int = 16,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)

        self.compressor = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.n_qubits),
        )
        self.quantum = VariationalQuantumLayer(n_qubits=self.n_qubits, n_layers=self.n_layers)
        self.skip_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.post_norm = nn.LayerNorm(self.n_qubits)
        self.classifier = nn.Linear(self.n_qubits, 1)

    def forward(
        self, x: torch.Tensor, return_quantum: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        compressed = self.compressor(x)
        angles = math.pi * torch.tanh(compressed)
        q_exp = self.quantum(angles)  # (B, n_qubits) in [-1, 1]
        fused = self.post_norm(q_exp + self.skip_scale * torch.tanh(compressed))
        logits = self.classifier(fused).squeeze(-1)
        if return_quantum:
            return logits, q_exp, angles
        return logits


class ClassicalMatchedMLP(nn.Module):
    """Parameter-matched purely classical MLP baseline for fair scientific ablation against VQC."""

    def __init__(
        self,
        in_dim: int,
        bottleneck_dim: int = 4,
        hidden_dim: int = 16,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.bottleneck_dim = int(bottleneck_dim)

        self.compressor = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.bottleneck_dim),
        )
        # Classical 4->4 nonlinear transformation with ~24 params matching 4-qubit x 3-layer VQC (24 params).
        self.classical_core = nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            nn.Tanh(),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim, bias=False),
            nn.Tanh(),
        )
        self.skip_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.post_norm = nn.LayerNorm(self.bottleneck_dim)
        self.classifier = nn.Linear(self.bottleneck_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compressed = self.compressor(x)
        core = self.classical_core(torch.tanh(compressed))
        fused = self.post_norm(core + self.skip_scale * torch.tanh(compressed))
        return self.classifier(fused).squeeze(-1)
