"""Hyperparameter tuning with Optuna, same budget for every model.

OWNER: A (Huấn). Stub with the agreed signature only.

Expected behaviour:
1. Tune on the seed-0 split, score on the validation part with the primary
   metric (ROC-AUC for "binary", RMSE for "regression").
2. Use exactly ``n_trials`` trials for every model (fair budget).
3. Read search spaces from ``configs/search_spaces/<model_name>.yaml``.
4. Return the best params and the full history (one row per trial).
"""
from __future__ import annotations

import pandas as pd


def tune(model_name: str, dataset: str, n_trials: int, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    raise NotImplementedError("src/tune.py: tune chưa được cài đặt (owner A)")
