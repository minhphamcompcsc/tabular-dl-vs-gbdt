"""Run one (dataset, model, seed) and turn it into results rows.

OWNER: C (Minh). Used by every script in scripts/.

Pipeline for one run (report section 2.1, Figure 1):
    load_dataset -> make_split(seed, train_size) -> preprocessor.fit(train only)
    -> model.fit(train, val) -> evaluate on test -> one CSV row per metric
"""
from __future__ import annotations

import functools
import json
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from src import env_info, fakes
from src.config import gbdt_early_stopping_rounds
from src.interfaces import Part, Split
from src.metrics import evaluate
from src.models import build_model
from src.results_io import RunKey, append_rows, done_keys

ROOT = Path(__file__).resolve().parents[1]
BEST_PARAMS_DIR = ROOT / "results" / "best_params"


def data_api(dataset: str) -> ModuleType:
    """Synthetic datasets come from src.fakes, real ones from src.data (owner A)."""
    if dataset in fakes.FAKE_DATASETS:
        return fakes
    from src import data  # imported lazily so a broken data.py does not break smoke runs
    return data


@functools.lru_cache(maxsize=None)
def _load(dataset: str):
    return data_api(dataset).load_dataset(dataset)


def run_one(dataset: str, model_name: str, seed: int, params: dict[str, Any] | None = None,
            n_trials: int = 0, train_size: int | None = None,
            skip: set[RunKey] | None = None) -> list[dict]:
    """Train and evaluate one model. Returns [] if the run is already in ``skip``."""
    api = data_api(dataset)
    X, y, meta = _load(dataset)
    split = api.make_split(X, y, seed=seed, train_size=train_size)
    if skip and (dataset, model_name, seed, n_trials, split.n_train) in skip:
        return []

    model = build_model(model_name, meta.task, params, seed=seed)
    pre = api.build_preprocessor(model.preprocess_kind, meta)
    pre.fit(split.train.X)  # fitted on train only: no leakage from val/test
    split_t = Split(
        train=Part(pre.transform(split.train.X), split.train.y),
        val=Part(pre.transform(split.val.X), split.val.y),
        test=Part(pre.transform(split.test.X), split.test.y),
        seed=seed,
    )

    start = time.perf_counter()
    model.fit(split_t.train.X, split_t.train.y, split_t.val.X, split_t.val.y)
    train_time = time.perf_counter() - start

    scores = evaluate(model, split_t, meta.task)
    common = {
        "dataset": dataset,
        "model": model_name,
        "seed": int(seed),
        "n_trials": int(n_trials),
        "n_train": split.n_train,
        "train_time_s": round(train_time, 4),
        "n_params": int(getattr(model, "n_params", 0)),
        "hardware": env_info.hardware_for(model),
        "git_commit": env_info.git_commit(),
    }
    return [{**common, "metric_name": k, "metric_value": v} for k, v in scores.items()]


def run_grid(datasets: list[str], models: list[str], seeds: list[int], output: str | Path,
             params_for: Callable[[str, str], dict | None] = lambda d, m: None,
             n_trials: int = 0, train_sizes: list[int | None] = (None,),
             resume: bool = True, strict: bool = False, log: Callable[[str], None] = print) -> int:
    """Run every combination and append rows to ``output``. Returns rows written.

    Models that are still stubs (NotImplementedError) are skipped with a
    warning, unless ``strict`` is set.
    """
    skip = done_keys(output) if resume else set()
    written, unavailable = 0, set()
    for dataset in datasets:
        for model_name in models:
            if (dataset, model_name) in unavailable:
                continue
            try:
                params = params_for(dataset, model_name)
            except NotImplementedError as exc:
                if strict:
                    raise
                log(f"[skip] {dataset}/{model_name}: {exc}")
                continue
            for train_size in train_sizes:
                for seed in seeds:
                    try:
                        rows = run_one(dataset, model_name, seed, params, n_trials, train_size, skip)
                    except NotImplementedError as exc:
                        if strict:
                            raise
                        log(f"[skip] {dataset}/{model_name}: {exc}")
                        unavailable.add((dataset, model_name))
                        break
                    append_rows(output, rows)
                    written += len(rows)
                    if rows:
                        r0 = rows[0]
                        skip.add((dataset, model_name, int(seed), int(n_trials), int(r0["n_train"])))
                        summary = ", ".join(f"{r['metric_name']}={r['metric_value']:.4f}" for r in rows)
                        log(f"[done] {dataset}/{model_name} seed={seed} n_train={rows[0]['n_train']}: {summary}")
                if (dataset, model_name) in unavailable:
                    break
    return written


# ---------------------------------------------------------------- best params

def best_params_path(dataset: str, model_name: str, n_trials: int) -> Path:
    policy = (f"__es{gbdt_early_stopping_rounds()}_v1"
              if model_name in {"xgboost", "lightgbm", "catboost"} else "")
    return BEST_PARAMS_DIR / f"{dataset}__{model_name}__n{n_trials}{policy}.json"


def save_best_params(dataset: str, model_name: str, n_trials: int, params: dict) -> Path:
    path = best_params_path(dataset, model_name, n_trials)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def load_best_params(dataset: str, model_name: str, n_trials: int) -> dict | None:
    path = best_params_path(dataset, model_name, n_trials)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_or_tune(dataset: str, model_name: str, n_trials: int, tuning_seed: int = 0,
                log: Callable[[str], None] = print) -> dict | None:
    """Load cached best params, or call A's ``tune()`` and cache the result.

    Returns None (default params) when ``n_trials == 0``.
    """
    if n_trials <= 0:
        return None
    cached = load_best_params(dataset, model_name, n_trials)
    if cached is not None:
        return cached
    if dataset in fakes.FAKE_DATASETS:
        return None  # no tuning on synthetic data: default params, n_trials is only a label
    from src.tune import tune  # owner A
    log(f"[tune] {dataset}/{model_name} with {n_trials} trials")
    best, history = tune(model_name, dataset, n_trials, seed=tuning_seed)
    save_best_params(dataset, model_name, n_trials, best)
    path = best_params_path(dataset, model_name, n_trials)
    hist_path = path.with_name(f"{path.stem}__history.csv")
    history.to_csv(hist_path, index=False)
    return best
