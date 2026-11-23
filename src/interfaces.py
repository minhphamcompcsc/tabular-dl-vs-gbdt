"""Shared contract between the three parts of the project (A, B, C).

Every module in this repo talks to the others only through the types below.
If you need to change something here, tell the whole group first, because
scripts, tests and the statistics code all rely on it.

Summary of the contract
-----------------------
Data (owner A, ``src/data.py``)
    load_dataset(name)                        -> (X: pd.DataFrame, y: np.ndarray, meta: DatasetMeta)
    make_split(X, y, seed, train_size=None)   -> Split   (60/20/20, stratified for "binary")
    build_preprocessor(kind, meta)            -> Preprocessor  (kind is "nn" or "gbdt")

Models (owner A for GBDT, B for neural nets and TabPFN, C for baselines)
    class SomeModel:
        preprocess_kind = "nn" | "gbdt"
        def __init__(self, task, seed=0, **params): ...
        def fit(self, X_tr, y_tr, X_val, y_val) -> self
        def predict(self, X) -> np.ndarray   # P(y=1) for "binary", values for "regression"
        n_params: int                        # set after fit
        device: str = "cpu"                  # "cuda" if the model trained on GPU

Tuning (owner A, ``src/tune.py``)
    tune(model_name, dataset, n_trials, seed=0) -> (best_params: dict, history: pd.DataFrame)

Evaluation (owner C, ``src/metrics.py``)
    evaluate(model, split, task) -> dict[str, float]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

import numpy as np

TaskType = Literal["binary", "regression"]
PreprocessKind = Literal["nn", "gbdt"]

# Primary metric used for ranking and statistical tests, per task type.
PRIMARY_METRIC: dict[str, str] = {"binary": "roc_auc", "regression": "rmse"}

# Fixed column order of every results CSV. Do not change without telling C.
RESULT_COLUMNS: list[str] = [
    "dataset",
    "model",
    "seed",
    "n_trials",
    "n_train",
    "metric_name",
    "metric_value",
    "train_time_s",
    "n_params",
    "hardware",
    "git_commit",
]


@dataclass
class DatasetMeta:
    """Description of one dataset, returned by ``load_dataset``."""

    name: str
    task: TaskType
    num_cols: list[str] = field(default_factory=list)
    cat_cols: list[str] = field(default_factory=list)
    source: str = ""
    license: str = ""


class Part(NamedTuple):
    """One part of a split. ``X`` is a DataFrame before preprocessing and
    whatever the model accepts after preprocessing (usually np.ndarray)."""

    X: Any
    y: np.ndarray


@dataclass
class Split:
    train: Part
    val: Part
    test: Part
    seed: int = 0

    @property
    def n_train(self) -> int:
        return int(len(self.train.y))


@runtime_checkable
class Preprocessor(Protocol):
    def fit(self, X: Any) -> "Preprocessor": ...

    def transform(self, X: Any) -> Any: ...


@runtime_checkable
class Model(Protocol):
    preprocess_kind: str
    n_params: int

    def fit(self, X_tr: Any, y_tr: np.ndarray, X_val: Any, y_val: np.ndarray) -> "Model": ...

    def predict(self, X: Any) -> np.ndarray: ...
