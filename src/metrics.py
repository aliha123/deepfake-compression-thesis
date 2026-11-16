"""Evaluation helpers for binary deepfake-detection experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    f1: float
    roc_auc: float | None


def evaluate_binary_predictions(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Compute common binary classification metrics.

    ROC AUC is returned as None when only one class is present.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have identical shapes")

    predictions = (scores >= threshold).astype(int)

    roc_auc = None
    if np.unique(y_true).size > 1:
        roc_auc = float(roc_auc_score(y_true, scores))

    return BinaryMetrics(
        accuracy=float(accuracy_score(y_true, predictions)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        roc_auc=roc_auc,
    )


def robustness_drop(reference: float, compressed: float) -> float:
    """Return absolute performance degradation after compression."""
    return float(reference - compressed)
