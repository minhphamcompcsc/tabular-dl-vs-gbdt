"""Evaluation metrics.

OWNER: C (Minh). Report section 2.7 / 3.1.

* "binary":     ROC-AUC (primary) and accuracy at threshold 0.5.
* "regression": RMSE (primary, lower is better).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

from src.interfaces import Split

HIGHER_IS_BETTER: dict[str, bool] = {"roc_auc": True, "accuracy": True, "rmse": False}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("predictions contain NaN or inf")
    if task == "binary":
        if y_pred.min() < 0 or y_pred.max() > 1:
            raise ValueError("binary models must return probabilities P(y=1) in [0, 1]")
        return {
            "roc_auc": float(roc_auc_score(y_true, y_pred)),
            "accuracy": float(accuracy_score(y_true, (y_pred >= 0.5).astype(int))),
        }
    if task == "regression":
        return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))}
    raise ValueError(f"unknown task '{task}'")


def evaluate(model, split: Split, task: str) -> dict[str, float]:
    """Score a fitted model on the test part of an already preprocessed split."""
    return compute_metrics(split.test.y, model.predict(split.test.X), task)


def is_better(metric: str, a: float, b: float) -> bool:
    """True if value ``a`` is strictly better than ``b`` for this metric."""
    return a > b if HIGHER_IS_BETTER[metric] else a < b
