"""XGBoost, LightGBM and CatBoost behind one interface.

OWNER: A (Huấn). Stub with the agreed interface only (see src/interfaces.py).
"""
from __future__ import annotations

import numpy as np


class _GBDTStub:
    preprocess_kind = "gbdt"
    device = "cpu"

    def __init__(self, task: str, seed: int = 0, **params):
        self.task, self.seed, self.params = task, seed, params
        self.n_params = 0

    def fit(self, X_tr, y_tr, X_val, y_val):
        raise NotImplementedError(f"{type(self).__name__} chưa được cài đặt (owner A)")

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} chưa được cài đặt (owner A)")


class XGBoostModel(_GBDTStub):
    pass


class LightGBMModel(_GBDTStub):
    pass


class CatBoostModel(_GBDTStub):
    pass
