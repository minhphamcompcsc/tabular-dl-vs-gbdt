"""Load configs/experiment.yaml merged with configs/experiments/<name>.yaml.

OWNER: C (Minh).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

SMOKE_DATASETS = ["fake_binary", "fake_regression"]
SMOKE_SEEDS = [0, 1, 2]
SMOKE_MODELS = ["dummy", "linear", "sk_hgb", "sk_mlp"]


def gbdt_early_stopping_rounds() -> int:
    """Read the shared policy used by both tuning and final GBDT fits."""
    with (CONFIG_DIR / "experiment.yaml").open(encoding="utf-8") as f:
        rounds = yaml.safe_load(f)["gbdt_early_stopping_rounds"]
    if type(rounds) is not int or rounds < 1:
        raise ValueError("gbdt_early_stopping_rounds must be a positive integer")
    return rounds


def load_config(experiment: str, config_dir: Path | str = CONFIG_DIR) -> dict[str, Any]:
    config_dir = Path(config_dir)
    with open(config_dir / "experiment.yaml", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    exp_path = config_dir / "experiments" / f"{experiment}.yaml"
    if exp_path.exists():
        with open(exp_path, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    cfg.setdefault("name", experiment)
    return cfg


def apply_cli_overrides(cfg: dict[str, Any], args) -> dict[str, Any]:
    """Apply --smoke, --datasets, --models, --seeds, --n-trials, --out."""
    cfg = dict(cfg)
    if getattr(args, "smoke", False):
        cfg["datasets"] = SMOKE_DATASETS
        cfg["seeds"] = SMOKE_SEEDS
        cfg["models"] = SMOKE_MODELS
        cfg["output"] = str(Path("results/smoke") / Path(cfg.get("output", f"{cfg['name']}.csv")).name)
        if cfg.get("train_sizes"):
            cfg["train_sizes"] = [100, 300, None]
    if getattr(args, "datasets", None):
        cfg["datasets"] = _split_list(args.datasets)
    if getattr(args, "models", None):
        cfg["models"] = _split_list(args.models)
    if getattr(args, "seeds", None):
        cfg["seeds"] = [int(s) for s in _split_list(args.seeds)]
    if getattr(args, "n_trials", None) is not None:
        cfg["n_trials"] = int(args.n_trials)
    if getattr(args, "out", None):
        cfg["output"] = str(Path(args.out).resolve())
    elif cfg.get("output") and not Path(cfg["output"]).is_absolute():
        cfg["output"] = str(ROOT / cfg["output"])  # config paths are relative to the repo root
    return cfg


def add_common_args(parser) -> None:
    parser.add_argument("--smoke", action="store_true",
                        help="Run on synthetic data with 3 seeds (no dependency on A/B).")
    parser.add_argument("--datasets", help="Comma-separated list, overrides the config.")
    parser.add_argument("--models", help="Comma-separated list, overrides the config.")
    parser.add_argument("--seeds", help="Comma-separated list, overrides the config.")
    parser.add_argument("--n-trials", dest="n_trials", type=int, help="Override the tuning budget.")
    parser.add_argument("--out", help="Output CSV path.")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Re-run rows that already exist in the output CSV.")
    parser.add_argument("--strict", action="store_true",
                        help="Fail instead of skipping models that are not implemented yet.")


def _split_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]
