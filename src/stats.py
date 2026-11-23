"""Aggregation and statistical tests on results CSVs.

OWNER: C (Minh). Report section 2.7 (method) and 3.3 (Tables 2 and 3).

Method
------
* Every model is run on the same 10 seeds, so results are paired by seed.
* Wilcoxon signed-rank test on the per-seed differences (two-sided).
* Holm correction inside one family = all comparisons on one dataset and metric.
* Default comparison: each deep learning model vs. the best GBDT on that
  dataset (fewer pairs, so Holm is less strict).

Note on the number of seeds: with n paired seeds and no ties, the smallest
possible two-sided Wilcoxon p-value is 2 / 2**n (see ``min_achievable_p``):
n=3 -> 0.25, n=5 -> 0.0625, n=10 -> 0.00195. With k comparisons in a Holm
family, a difference can only be significant if 2 / 2**n < 0.05 / k.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.interfaces import PRIMARY_METRIC
from src.metrics import HIGHER_IS_BETTER
from src.models import family

ALPHA = 0.05


# ------------------------------------------------------------------ basics

def min_achievable_p(n: int) -> float:
    """Smallest two-sided exact Wilcoxon p-value with n non-zero, untied pairs."""
    return min(1.0, 2.0 / 2 ** n)


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def wilcoxon_p(a, b) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired samples a and b."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) == 0:
        return float("nan")
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def primary_metric_of(df: pd.DataFrame, dataset: str) -> str:
    """ROC-AUC for binary datasets, RMSE for regression ones."""
    metrics = set(df.loc[df["dataset"] == dataset, "metric_name"])
    for m in (PRIMARY_METRIC["binary"], PRIMARY_METRIC["regression"]):
        if m in metrics:
            return m
    raise ValueError(f"no primary metric for dataset '{dataset}' (found {sorted(metrics)})")


# --------------------------------------------------------------- aggregation

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over seeds per (dataset, model, n_trials, n_train, metric)."""
    keys = ["dataset", "model", "n_trials", "n_train", "metric_name"]
    out = (df.groupby(keys, as_index=False)
             .agg(mean=("metric_value", "mean"),
                  std=("metric_value", "std"),
                  n_seeds=("seed", "nunique"),
                  train_time_s=("train_time_s", "mean"),
                  n_params=("n_params", "median")))
    out["std"] = out["std"].fillna(0.0)
    out["family"] = out["model"].map(family)
    return out


def paired(df: pd.DataFrame, dataset: str, model_a: str, model_b: str, metric: str) -> pd.DataFrame:
    """Per-seed values of two models on one dataset, aligned by seed."""
    sub = df[(df["dataset"] == dataset) & (df["metric_name"] == metric)]
    a = sub[sub["model"] == model_a].groupby("seed")["metric_value"].mean()
    b = sub[sub["model"] == model_b].groupby("seed")["metric_value"].mean()
    both = pd.concat({"a": a, "b": b}, axis=1).dropna()
    dropped = set(a.index) ^ set(b.index)
    if dropped:
        warnings.warn(f"{dataset}: seeds {sorted(dropped)} missing for {model_a} or {model_b}; dropped")
    return both


def compare(df: pd.DataFrame, model_a: str, model_b: str, dataset: str, metric: str) -> dict:
    """Paired comparison of two models on one dataset (raw, unadjusted p-value).

    ``mean_diff`` is mean(a - b) in the metric's own units; ``a_better`` tells
    whether that difference favours ``model_a`` (accounts for RMSE being
    lower-is-better).
    """
    both = paired(df, dataset, model_a, model_b, metric)
    n = len(both)
    diff = float((both["a"] - both["b"]).mean()) if n else float("nan")
    return {
        "dataset": dataset,
        "metric": metric,
        "model_a": model_a,
        "model_b": model_b,
        "n_seeds": n,
        "mean_a": float(both["a"].mean()) if n else float("nan"),
        "mean_b": float(both["b"].mean()) if n else float("nan"),
        "mean_diff": diff,
        "a_better": bool(diff > 0) if HIGHER_IS_BETTER[metric] else bool(diff < 0),
        "p_raw": wilcoxon_p(both["a"], both["b"]) if n else float("nan"),
        "min_achievable_p": min_achievable_p(n),
    }


def best_model(df: pd.DataFrame, dataset: str, metric: str, candidates: list[str]) -> str | None:
    """Candidate with the best mean on ``metric`` for ``dataset``."""
    sub = df[(df["dataset"] == dataset) & (df["metric_name"] == metric) & (df["model"].isin(candidates))]
    if sub.empty:
        return None
    means = sub.groupby("model")["metric_value"].mean()
    return str(means.idxmax() if HIGHER_IS_BETTER[metric] else means.idxmin())


def compare_to_reference(df: pd.DataFrame, challengers: list[str] | None = None,
                         reference: str = "best_gbdt", alpha: float = ALPHA) -> pd.DataFrame:
    """Table 3: every challenger vs. a reference model, Holm-corrected per dataset.

    ``reference="best_gbdt"`` picks, per dataset, the GBDT with the best mean
    primary metric. ``challengers`` defaults to all deep learning models.
    Caveat for the report: picking the best GBDT on the same test results
    slightly favours the GBDT side.
    """
    rows = []
    for dataset in sorted(df["dataset"].unique()):
        metric = primary_metric_of(df, dataset)
        present = sorted(df.loc[df["dataset"] == dataset, "model"].unique())
        if reference == "best_gbdt":
            ref = best_model(df, dataset, metric, [m for m in present if family(m) == "gbdt"])
        else:
            ref = reference if reference in present else None
        if ref is None:
            warnings.warn(f"{dataset}: no reference model found; skipped")
            continue
        chall = challengers or [m for m in present if family(m) == "dl"]
        family_rows = [compare(df, m, ref, dataset, metric) for m in chall if m in present and m != ref]
        if not family_rows:
            continue
        p_adj = holm([r["p_raw"] for r in family_rows])
        for r, p in zip(family_rows, p_adj):
            r["p_holm"] = float(p)
            r["significant"] = bool(p < alpha)
            r["winner"] = (r["model_a"] if r["a_better"] else r["model_b"]) if r["significant"] else "tie"
            rows.append(r)
    return pd.DataFrame(rows)


def average_rank(df: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """Rank models per dataset by mean primary metric (1 = best), then average.

    Only datasets where every listed model has results are used.
    """
    per_dataset = []
    for dataset in sorted(df["dataset"].unique()):
        metric = primary_metric_of(df, dataset)
        sub = df[(df["dataset"] == dataset) & (df["metric_name"] == metric)]
        means = sub.groupby("model")["metric_value"].mean()
        if models is not None:
            if not set(models) <= set(means.index):
                continue
            means = means[models]
        ranks = means.rank(ascending=not HIGHER_IS_BETTER[metric], method="average")
        per_dataset.append(ranks.rename(dataset))
    if not per_dataset:
        return pd.DataFrame(columns=["model", "avg_rank", "n_datasets"])
    table = pd.concat(per_dataset, axis=1)
    out = pd.DataFrame({
        "model": table.index,
        "avg_rank": table.mean(axis=1).to_numpy(),
        "n_datasets": table.notna().sum(axis=1).to_numpy(),
    })
    return out.sort_values("avg_rank").reset_index(drop=True)
