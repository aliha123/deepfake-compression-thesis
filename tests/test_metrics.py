import numpy as np
import pytest

from src.metrics import evaluate_binary_predictions, robustness_drop


def test_binary_metrics_are_computed():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.7, 0.9])

    metrics = evaluate_binary_predictions(y_true, scores)

    assert metrics.accuracy == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0


def test_single_class_auc_is_none():
    y_true = np.array([1, 1])
    scores = np.array([0.6, 0.8])

    metrics = evaluate_binary_predictions(y_true, scores)

    assert metrics.roc_auc is None


def test_robustness_drop():
    assert robustness_drop(0.91, 0.84) == pytest.approx(0.07)
