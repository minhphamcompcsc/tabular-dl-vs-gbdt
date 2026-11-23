"""Synthetic datasets that follow the data contract of src/data.py.

OWNER: C (Minh). These are test doubles so scripts, metrics, statistics and
plots can be built and tested before A's real ``src/data.py`` is ready.
Dataset names: ``fake_binary`` and ``fake_regression``.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.interfaces import DatasetMeta, Part, Split

FAKE_DATASETS = ("fake_binary", "fake_regression")
N_SAMPLES = 3000
N_NUM = 6


@functools.lru_cache(maxsize=None)
def _make(name: str) -> tuple[pd.DataFrame, np.ndarray, DatasetMeta]:
    rng = np.random.default_rng(0)
    if name == "fake_binary":
        X, y = make_classification(n_samples=N_SAMPLES, n_features=N_NUM, n_informative=4,
                                   weights=[0.7, 0.3], random_state=0)
        task = "binary"
    elif name == "fake_regression":
        X, y = make_regression(n_samples=N_SAMPLES, n_features=N_NUM, n_informative=4,
                               noise=10.0, random_state=0)
        task = "regression"
    else:
        raise KeyError(f"Unknown fake dataset '{name}'. Known: {FAKE_DATASETS}")
    num_cols = [f"x{i}" for i in range(N_NUM)]
    df = pd.DataFrame(X, columns=num_cols)
    df["color"] = rng.choice(["red", "green", "blue"], size=N_SAMPLES)
    meta = DatasetMeta(name=name, task=task, num_cols=num_cols, cat_cols=["color"],
                       source="synthetic (sklearn.datasets)", license="n/a")
    return df, np.asarray(y, dtype=float if task == "regression" else int), meta


def load_dataset(name: str):
    X, y, meta = _make(name)
    return X.copy(), y.copy(), meta


def make_split(X: pd.DataFrame, y: np.ndarray, seed: int, train_size: int | None = None,
               fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> Split:
    is_binary = set(np.unique(y)) <= {0, 1}
    _, f_val, f_test = fractions
    idx = np.arange(len(y))
    idx_rest, idx_test = train_test_split(idx, test_size=f_test, random_state=seed,
                                          stratify=y if is_binary else None)
    idx_train, idx_val = train_test_split(idx_rest, test_size=f_val / (1 - f_test), random_state=seed,
                                          stratify=y[idx_rest] if is_binary else None)
    if train_size is not None and train_size < len(idx_train):
        idx_train, _ = train_test_split(idx_train, train_size=train_size, random_state=seed,
                                        stratify=y[idx_train] if is_binary else None)
    part = lambda i: Part(X.iloc[i].reset_index(drop=True), y[i])  # noqa: E731
    return Split(train=part(idx_train), val=part(idx_val), test=part(idx_test), seed=seed)


def build_preprocessor(kind: str, meta: DatasetMeta) -> ColumnTransformer:
    # Same transform for both kinds; the real data.py may differ for "gbdt".
    return ColumnTransformer([
        ("num", StandardScaler(), meta.num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), meta.cat_cols),
    ])


# ------------------------------------------------------------ smoke models
# Fast scikit-learn stand-ins so the full pipeline (including the
# "DL vs best GBDT" statistics and figures) can be exercised before A's and
# B's models exist. They are NOT part of the study; do not report them.

class SkHGBModel:
    """Stand-in for a GBDT: sklearn HistGradientBoosting."""

    preprocess_kind = "gbdt"
    device = "cpu"

    def __init__(self, task: str, seed: int = 0, **params):
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        cls = HistGradientBoostingClassifier if task == "binary" else HistGradientBoostingRegressor
        self.task = task
        self.est = cls(random_state=seed, max_iter=params.get("max_iter", 100))
        self.n_params = 0

    def fit(self, X_tr, y_tr, X_val=None, y_val=None):
        self.est.fit(X_tr, y_tr)
        try:
            self.n_params = int(sum(len(p.nodes) for preds in self.est._predictors for p in preds))
        except AttributeError:
            self.n_params = int(self.est.n_iter_)
        return self

    def predict(self, X) -> np.ndarray:
        return self.est.predict_proba(X)[:, 1] if self.task == "binary" else self.est.predict(X)


class SkMLPModel:
    """Stand-in for a deep learning model: sklearn MLP with early stopping."""

    preprocess_kind = "nn"
    device = "cpu"

    def __init__(self, task: str, seed: int = 0, **params):
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        cls = MLPClassifier if task == "binary" else MLPRegressor
        self.task = task
        self.est = cls(hidden_layer_sizes=params.get("hidden", (64, 64)), early_stopping=True,
                       max_iter=params.get("max_iter", 300), random_state=seed)
        self.n_params = 0

    def fit(self, X_tr, y_tr, X_val=None, y_val=None):
        y = y_tr if self.task == "binary" else (y_tr - y_tr.mean()) / (y_tr.std() + 1e-12)
        self._mu, self._sd = (0.0, 1.0) if self.task == "binary" else (y_tr.mean(), y_tr.std() + 1e-12)
        self.est.fit(X_tr, y)
        self.n_params = int(sum(w.size for w in self.est.coefs_) + sum(b.size for b in self.est.intercepts_))
        return self

    def predict(self, X) -> np.ndarray:
        if self.task == "binary":
            return self.est.predict_proba(X)[:, 1]
        return self.est.predict(X) * self._sd + self._mu
