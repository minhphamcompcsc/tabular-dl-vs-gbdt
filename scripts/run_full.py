"""Report sections 3.3, 3.4, 3.8: tune every model with the same budget N
(once, on the seed-0 split), then train and test on every seed.

Tuned params are cached in results/best_params/ and reused by the other
experiments. Baselines (family "baseline") are not tuned.

Usage:
    python scripts/run_full.py                      # needs src/data.py and src/tune.py
    python scripts/run_full.py --no-tune            # default params, n_trials recorded as 0
    python scripts/run_full.py --models xgboost --datasets adult
    python scripts/run_full.py --smoke
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from src.config import add_common_args, apply_cli_overrides, load_config
from src.models import family
from src.runner import get_or_tune, run_grid


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--no-tune", action="store_true", help="Skip tuning and use default params.")
    args = parser.parse_args(argv)
    cfg = apply_cli_overrides(load_config("full"), args)
    n_trials = 0 if args.no_tune else int(cfg["n_trials"])
    tuning_seed = int(cfg.get("tuning_seed", 0))

    baselines = [m for m in cfg["models"] if family(m) == "baseline"]
    tuned = [m for m in cfg["models"] if family(m) != "baseline"]
    total = run_grid(cfg["datasets"], baselines, cfg["seeds"], cfg["output"], n_trials=0,
                     resume=args.resume, strict=args.strict)
    total += run_grid(cfg["datasets"], tuned, cfg["seeds"], cfg["output"],
                      params_for=lambda d, m: get_or_tune(d, m, n_trials, tuning_seed),
                      n_trials=n_trials, resume=args.resume, strict=args.strict)
    print(f"Wrote {total} rows to {cfg['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
