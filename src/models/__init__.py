"""Model registry: maps a model name used in configs and CSVs to its class.

Imports are lazy, so a missing library (e.g. torch on a CPU-only laptop)
only matters when that model is actually requested.
"""
from __future__ import annotations

import importlib
from typing import Any

# name -> (module, class, family)
REGISTRY: dict[str, tuple[str, str, str]] = {
    # baselines (owner C)
    "dummy": ("src.models.baseline", "DummyModel", "baseline"),
    "linear": ("src.models.baseline", "LinearModel", "baseline"),
    # GBDT (owner A)
    "xgboost": ("src.models.gbdt", "XGBoostModel", "gbdt"),
    "lightgbm": ("src.models.gbdt", "LightGBMModel", "gbdt"),
    "catboost": ("src.models.gbdt", "CatBoostModel", "gbdt"),
    # deep learning (owner B)
    "mlp": ("src.models.nn", "MLPModel", "dl"),
    "resnet": ("src.models.nn", "ResNetModel", "dl"),
    "ft_transformer": ("src.models.ft_transformer", "FTTransformerModel", "dl"),
    # tabular foundation model (owner B)
    "tabpfn": ("src.models.tabpfn_wrapper", "TabPFNModel", "foundation"),
    # smoke-test stand-ins (sklearn), only for --smoke and tests; never report them
    "sk_hgb": ("src.fakes", "SkHGBModel", "gbdt"),
    "sk_mlp": ("src.fakes", "SkMLPModel", "dl"),
}


def family(name: str) -> str:
    """Return "baseline", "gbdt", "dl", "foundation" or "other"."""
    return REGISTRY.get(name, ("", "", "other"))[2]


def get_model_class(name: str) -> type:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(REGISTRY)}")
    module_name, class_name, _ = REGISTRY[name]
    return getattr(importlib.import_module(module_name), class_name)


def build_model(name: str, task: str, params: dict[str, Any] | None = None, seed: int = 0):
    return get_model_class(name)(task=task, seed=seed, **(params or {}))
