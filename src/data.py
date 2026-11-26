"""Datasets, splitting and preprocessing.

OWNER: A (Huấn).

Checklist for A (also checked by ``tests/test_contract.py``):
1. ``load_dataset`` returns y as np.ndarray; for "binary" the labels are 0/1.
2. ``make_split`` uses 60/20/20 from ``configs/experiment.yaml``, stratified
   for "binary", deterministic for a given seed. ``train_size`` subsamples the
   train part only (val and test stay the same for that seed).
3. ``build_preprocessor`` is fitted on the train part only (no leakage).

Loads Adult, California Housing and German Credit with dataset metadata.
Creates deterministic train/validation/test splits from the shared config,
with stratification for binary targets and optional train-only subsampling.
Builds unfitted preprocessing pipelines for GBDT and neural networks;
callers must fit them on training data only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pathlib import Path
import yaml

from src.interfaces import DatasetMeta, Part, PreprocessKind, Preprocessor, Split

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASETS = ["adult", "california_housing", "german_credit"]


def load_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray, DatasetMeta]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "datasets.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        datasets = yaml.safe_load(config_file)

    if name not in datasets:
        raise ValueError(f"Dataset không được hỗ trợ: {name}")

    config = datasets[name]

    if name == "california_housing":
        from sklearn.datasets import fetch_california_housing

        dataset = fetch_california_housing(as_frame=True)
        X = dataset.data.copy()
        y = dataset.target.to_numpy(dtype=float)
        num_cols = X.columns.tolist()
        cat_cols = []
    else:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=config["uci_id"])
        X = dataset.data.features.copy()
        target = dataset.data.targets.iloc[:, 0]
        variable_types = dataset.variables.set_index("name")["type"]
        cat_cols = [
            col for col in X.columns
            if variable_types[col] in {"Categorical", "Binary"}
        ]
        num_cols = [col for col in X.columns if col not in cat_cols]

        if name == "adult":
            X[cat_cols] = X[cat_cols].replace("?", np.nan)
            labels = target.astype("string").str.strip().str.rstrip(".")
            encoded = labels.map({"<=50K": 0, ">50K": 1})
        else:
            encoded = target.map({1: 0, 2: 1})

        if encoded.isna().any():
            raise ValueError(f"{name}: có nhãn thiếu hoặc ngoài ánh xạ")
        y = encoded.to_numpy(dtype=int)

    expected_shape = (config["n_samples"], config["n_features"])
    if X.shape != expected_shape or y.shape != (len(X),):
        raise ValueError(f"{name}: kích thước dữ liệu không khớp cấu hình")

    meta = DatasetMeta(
        name=name,
        task=config["task"],
        num_cols=num_cols,
        cat_cols=cat_cols,
        source=config["source"],
        license=config["license"],
    )
    return X, y, meta


def make_split(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    train_size: int | None = None,
) -> Split:
    """Chia dữ liệu theo cấu hình, rồi lấy mẫu con từ train nếu cần."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        fractions = yaml.safe_load(config_file)["split"]

    # Chữ ký không nhận meta. Trong repo này, loader mã hóa nhãn nhị phân
    # thành 0/1, còn nhãn California Housing là giá trị liên tục.
    is_binary = np.array_equal(np.unique(y), [0, 1])
    indices = np.arange(len(y))

    remaining, test_idx = train_test_split(
        indices,
        test_size=fractions["test"],
        random_state=seed,
        stratify=y if is_binary else None,
    )

    train_idx, val_idx = train_test_split(
        remaining,
        test_size=fractions["val"] / (fractions["train"] + fractions["val"]),
        random_state=seed,
        stratify=y[remaining] if is_binary else None,
    )

    if train_size is not None:
        if not 1 <= train_size <= len(train_idx):
            raise ValueError(f"train_size phải nằm trong [1, {len(train_idx)}]")
        if train_size < len(train_idx):
            train_idx, _ = train_test_split(
                train_idx,
                train_size=train_size,
                random_state=seed,
                stratify=y[train_idx] if is_binary else None,
            )

    def part(idx: np.ndarray) -> Part:
        return Part(X.iloc[idx].reset_index(drop=True), y[idx])

    return Split(
        train=part(train_idx),
        val=part(val_idx),
        test=part(test_idx),
        seed=seed,
    )


def build_preprocessor(kind: PreprocessKind, meta: DatasetMeta) -> Preprocessor:
    """Tạo bộ tiền xử lý chưa fit cho một dataset."""
    if kind not in {"nn", "gbdt"}:
        raise ValueError(f"Kiểu tiền xử lý không được hỗ trợ: {kind}")

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if kind == "nn":
        numeric_steps.append(("scale", StandardScaler()))

    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), meta.num_cols),
        ("cat", categorical, meta.cat_cols),
    ])
