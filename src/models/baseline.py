"""Simple baselines required by the assignment (section 4.1, step 4).

OWNER: C (Minh).

* ``dummy``:  predicts the train prior (binary) or the train mean (regression).
  Any real model must beat it; used as a sanity check.
* ``linear``: Logistic regression (binary) or Ridge/Linear regression
  (regression), using the "nn" preprocessing (standardized + one-hot).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge


class DummyModel:
    preprocess_kind = "nn"
    device = "cpu"

    def __init__(self, task: str, seed: int = 0, **params):
        self.task, self.seed = task, seed
        self.n_params = 1
        self._value = 0.0

    def fit(self, X_tr, y_tr, X_val=None, y_val=None):
        self._value = float(np.mean(y_tr))
        return self

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self._value, dtype=float)


class LinearModel:
    """Logistic regression for "binary", Ridge for "regression".

    Params: ``C`` (inverse L2 strength, binary) and ``alpha`` (L2 strength,
    regression; ``alpha=0`` gives ordinary least squares).
    """

    preprocess_kind = "nn"
    device = "cpu"

    def __init__(self, task: str, seed: int = 0, C: float = 1.0, alpha: float = 1.0, max_iter: int = 2000):
        self.task, self.seed = task, seed
        if task == "binary":
            self.est = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
        elif alpha == 0:
            self.est = LinearRegression()
        else:
            self.est = Ridge(alpha=alpha, random_state=seed)
        self.n_params = 0

    def fit(self, X_tr, y_tr, X_val=None, y_val=None):
        self.est.fit(X_tr, y_tr)
        self.n_params = int(np.size(self.est.coef_) + np.size(self.est.intercept_))
        return self

    def predict(self, X) -> np.ndarray:
        if self.task == "binary":
            return self.est.predict_proba(X)[:, 1]
        return self.est.predict(X)
