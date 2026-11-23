"""Append-only results CSV with resume support.

OWNER: C (Minh). Every row follows ``RESULT_COLUMNS`` in src/interfaces.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.interfaces import RESULT_COLUMNS

RunKey = tuple[str, str, int, int, int]  # dataset, model, seed, n_trials, n_train


def read_results(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = pd.read_csv(path)
    missing = [c for c in RESULT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df[RESULT_COLUMNS]


def done_keys(path: str | Path) -> set[RunKey]:
    df = read_results(path)
    return {
        (str(r.dataset), str(r.model), int(r.seed), int(r.n_trials), int(r.n_train))
        for r in df.itertuples(index=False)
    }


def append_rows(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[RESULT_COLUMNS]
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
