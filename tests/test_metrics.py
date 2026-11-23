import numpy as np
import pytest

from src.metrics import compute_metrics, is_better


def test_binary_perfect_and_constant():
    y = np.array([0, 0, 1, 1])
    perfect = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), "binary")
    assert perfect == {"roc_auc": 1.0, "accuracy": 1.0}
    constant = compute_metrics(y, np.full(4, 0.5), "binary")
    assert constant["roc_auc"] == 0.5


def test_binary_threshold_is_half():
    y = np.array([0, 1])
    assert compute_metrics(y, np.array([0.49, 0.5]), "binary")["accuracy"] == 1.0


def test_rmse_known_value():
    out = compute_metrics(np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, -1.0, 1.0, -1.0]), "regression")
    assert out == {"rmse": pytest.approx(1.0)}


def test_rejects_bad_predictions():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1]), np.array([0.2, 1.7]), "binary")  # not probabilities
    with pytest.raises(ValueError):
        compute_metrics(np.array([0.0, 1.0]), np.array([np.nan, 1.0]), "regression")
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1]), np.array([0.5]), "binary")


def test_direction():
    assert is_better("roc_auc", 0.9, 0.8)
    assert is_better("rmse", 1.0, 2.0)
