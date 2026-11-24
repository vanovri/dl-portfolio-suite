"""TF-IDF + LogisticRegression classical NLP baseline on exact IndoNLU SmSA splits."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from src.dataset import ID2LABEL


def compute_multiclass_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> dict[str, Any]:
    """Compute Accuracy, Macro-F1, per-class F1, and 3x3 confusion matrix."""
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    per_class = f1_score(y_t, y_p, labels=[0, 1, 2], average=None, zero_division=0)
    cm = confusion_matrix(y_t, y_p, labels=[0, 1, 2]).tolist()
    return {
        "accuracy": round(float(accuracy_score(y_t, y_p)), 4),
        "macro_f1": round(float(f1_score(y_t, y_p, average="macro", zero_division=0)), 4),
        "per_class_f1": {
            ID2LABEL[i]: round(float(per_class[i]), 4) for i in range(3)
        },
        "confusion_matrix": cm,
    }


def train_and_eval_tfidf_baseline(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    seed: int = 42,
    max_features: int = 20000,
) -> dict[str, Any]:
    """Fit TF-IDF (1,2)-grams + balanced LogisticRegression and evaluate on exact test split."""
    pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    t_fit0 = time.perf_counter()
    pipe.fit(train_texts, train_labels)
    fit_s = round(time.perf_counter() - t_fit0, 3)

    t_inf0 = time.perf_counter()
    preds = pipe.predict(test_texts)
    inf_ms = (time.perf_counter() - t_inf0) * 1000.0 / max(len(test_texts), 1)

    metrics = compute_multiclass_metrics(test_labels, preds)
    metrics["fit_elapsed_s"] = fit_s
    metrics["latency_ms_per_text"] = round(float(inf_ms), 4)
    metrics["model_type"] = "TfidfVectorizer(1,2) + LogisticRegression(balanced)"
    return metrics
