"""MLP and ResNet for tabular data (PyTorch), AdamW + early stopping.

OWNER: B (Thu). Stub with the agreed interface only (see src/interfaces.py).
Remember to set ``self.device`` to "cuda" when training on GPU, so the
results CSV records the right hardware.
"""
from __future__ import annotations

import numpy as np


class _NNStub:
    preprocess_kind = "nn"
    device = "cpu"

    def __init__(self, task: str, seed: int = 0, **params):
        self.task, self.seed, self.params = task, seed, params
        self.n_params = 0

    def fit(self, X_tr, y_tr, X_val, y_val):
        raise NotImplementedError(f"{type(self).__name__} chưa được cài đặt (owner B)")

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} chưa được cài đặt (owner B)")


class MLPModel(_NNStub):
    pass


class ResNetModel(_NNStub):
    pass
