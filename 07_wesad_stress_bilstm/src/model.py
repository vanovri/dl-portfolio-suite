"""Bi-LSTM Temporal Attention network and classical tabular baselines for WESAD stress detection."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.dsp import extract_window_summary_features


class AdditiveTemporalAttention(nn.Module):
    """Bahdanau-style additive temporal attention pooling over sequence hidden states."""

    def __init__(self, hidden_dim: int, attn_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted sequence summary and normalized temporal weights.

        Args:
            hidden_states: (B, T, H)
        Returns:
            context: (B, H)
            weights: (B, T) summing to 1.0 along time dimension
        """
        energy = torch.tanh(self.proj(hidden_states))
        scores = self.score(energy).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights


class BiLSTMAttentionStressNet(nn.Module):
    """Conv1D stem + 2-layer Bi-LSTM + Additive Temporal Attention + Classifier head."""

    def __init__(
        self,
        in_features: int = 9,
        conv_channels: int = 48,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.conv_stem = nn.Sequential(
            nn.Conv1d(in_features, conv_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_size * 2
        self.attention = AdditiveTemporalAttention(lstm_out_dim, attn_dim=hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, T, C) physiological sequence tensor.
        Returns:
            logits: (B, num_classes)
            attention_weights: (B, T)
        """
        h = self.conv_stem(x.transpose(1, 2)).transpose(1, 2)
        seq_out, _ = self.bilstm(h)
        context, attn_weights = self.attention(seq_out)
        logits = self.classifier(context)
        return logits, attn_weights


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute subject-independent classification metrics cleanly handling edge cases."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob_arr >= threshold).astype(np.int64)

    has_both = len(np.unique(y_true_arr)) == 2
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr)) if has_both else float("nan")
    pr_auc = float(average_precision_score(y_true_arr, y_prob_arr)) if has_both else float("nan")
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).tolist()

    return {
        "accuracy": round(float(accuracy_score(y_true_arr, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)), 4),
        "stress_f1": round(float(f1_score(y_true_arr, y_pred, pos_label=1, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4) if np.isfinite(roc_auc) else None,
        "pr_auc": round(pr_auc, 4) if np.isfinite(pr_auc) else None,
        "confusion_matrix": cm,
    }


def fit_and_eval_classical_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    """Benchmark LogisticRegression and RandomForest on window summary stats on the exact same split."""
    F_train = extract_window_summary_features(X_train)
    F_test = extract_window_summary_features(X_test)

    logreg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ]
    )
    logreg.fit(F_train, y_train)
    lr_probs = logreg.predict_proba(F_test)[:, 1]
    lr_metrics = compute_binary_classification_metrics(y_test, lr_probs)

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(F_train, y_train)
    rf_probs = rf.predict_proba(F_test)[:, 1]
    rf_metrics = compute_binary_classification_metrics(y_test, rf_probs)

    best_name = "random_forest" if rf_metrics["macro_f1"] >= lr_metrics["macro_f1"] else "logistic_regression"
    best_metrics = rf_metrics if best_name == "random_forest" else lr_metrics

    return {
        "logistic_regression": {**lr_metrics, "probs": lr_probs},
        "random_forest": {**rf_metrics, "probs": rf_probs},
        "best_classical_name": best_name,
        "best_classical_metrics": best_metrics,
    }
