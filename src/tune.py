"""Hyperparameter tuning with Optuna, same budget for every model.

OWNER: A (Huấn).

Expected behaviour:
1. Tune on the seed-0 split, score on the validation part with the primary
   metric (ROC-AUC for "binary", RMSE for "regression").
2. Use exactly ``n_trials`` trials for every model (fair budget).
3. Read search spaces from ``configs/search_spaces/<model_name>.yaml``.
4. Return the best params and the full history (one row per trial).

Reads per-model YAML search spaces and runs the requested trial budget.
Maximizes validation ROC-AUC for binary classification and minimizes
validation RMSE for regression. GBDT models use validation-based early
stopping. Returns the best parameters and trial history without scoring
the test set.
"""
from __future__ import annotations

import pandas as pd


from pathlib import Path

import optuna
import yaml

from src.data import build_preprocessor, load_dataset, make_split
from src.interfaces import PRIMARY_METRIC
from src.metrics import HIGHER_IS_BETTER, compute_metrics
from src.models import build_model, get_model_class

SEARCH_SPACES = Path(__file__).resolve().parents[1] / "configs" / "search_spaces"


def tune(
    model_name: str,
    dataset: str,
    n_trials: int,
    seed: int = 0,
) -> tuple[dict, pd.DataFrame]:
    if n_trials < 1:
        raise ValueError("n_trials phải lớn hơn 0")

    model_class = get_model_class(model_name)
    space_path = SEARCH_SPACES / f"{model_name}.yaml"
    with space_path.open(encoding="utf-8") as config_file:
        space = yaml.safe_load(config_file)

    X, y, meta = load_dataset(dataset)
    split = make_split(X, y, seed=seed)
    pre = build_preprocessor(model_class.preprocess_kind, meta)
    pre.fit(split.train.X)
    X_train = pre.transform(split.train.X)
    X_val = pre.transform(split.val.X)

    metric = PRIMARY_METRIC[meta.task]
    direction = "maximize" if HIGHER_IS_BETTER[metric] else "minimize"

    def objective(trial: optuna.Trial) -> float:
        params = {}
        for name, spec in space.items():
            kind = spec["type"]
            if kind == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif kind == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif kind == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"{model_name}.{name}: kiểu tham số lạ {kind!r}")

        model = build_model(model_name, meta.task, params, seed=seed)
        model.fit(X_train, split.train.y, X_val, split.val.y)
        scores = compute_metrics(split.val.y, model.predict(X_val), meta.task)
        return scores[metric]

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)

    history = study.trials_dataframe(
        attrs=("number", "value", "state", "params")
    )
    history["metric_name"] = metric
    return study.best_params, history
