"""Datasets, splitting and preprocessing.

OWNER: A (Huấn). This file is a stub that only fixes the function signatures
from ``src/interfaces.py``. Replace the bodies with the real implementation.

Until this file is implemented, scripts can still run on the synthetic
datasets ``fake_binary`` and ``fake_regression`` (see ``src/fakes.py``).

Checklist for A (also checked by ``tests/test_contract.py``):
1. ``load_dataset`` returns y as np.ndarray; for "binary" the labels are 0/1.
2. ``make_split`` uses 60/20/20 from ``configs/experiment.yaml``, stratified
   for "binary", deterministic for a given seed. ``train_size`` subsamples the
   train part only (val and test stay the same for that seed).
3. ``build_preprocessor`` is fitted on the train part only (no leakage).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.interfaces import DatasetMeta, PreprocessKind, Preprocessor, Split

DATASETS = ["adult", "california_housing", "german_credit"]


def load_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray, DatasetMeta]:
    """Load one dataset by name (see ``configs/datasets.yaml``)."""
    raise NotImplementedError("src/data.py: load_dataset chưa được cài đặt (owner A)")


def make_split(X: pd.DataFrame, y: np.ndarray, seed: int, train_size: int | None = None) -> Split:
    """Split into train/val/test (60/20/20). ``train_size`` subsamples train."""
    raise NotImplementedError("src/data.py: make_split chưa được cài đặt (owner A)")


def build_preprocessor(kind: PreprocessKind, meta: DatasetMeta) -> Preprocessor:
    """Return an unfitted object with ``fit(X_train)`` and ``transform(X)``.

    kind="nn":   standardize numeric columns, one-hot (or ordinal) categorical.
    kind="gbdt": whatever the GBDT libraries need (e.g. pandas category dtype).
    """
    raise NotImplementedError("src/data.py: build_preprocessor chưa được cài đặt (owner A)")
