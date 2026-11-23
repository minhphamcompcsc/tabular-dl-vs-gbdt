"""Report section 3.6 (Figure 3): performance vs number of training samples.

Reuses the params tuned by run_full.py (budget N); val and test parts stay the
same for each seed, only the train part is subsampled.

Usage:
    python scripts/run_size_curve.py
    python scripts/run_size_curve.py --smoke
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from src.config import add_common_args, apply_cli_overrides, load_config
from src.runner import get_or_tune, run_grid


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args(argv)
    cfg = apply_cli_overrides(load_config("size_curve"), args)
    n_trials = int(cfg["n_trials"])
    sizes = [None if s is None else int(s) for s in cfg["train_sizes"]]
    total = run_grid(cfg["datasets"], cfg["models"], cfg["seeds"], cfg["output"],
                     params_for=lambda d, m: get_or_tune(d, m, n_trials, cfg.get("tuning_seed", 0)),
                     n_trials=n_trials, train_sizes=sizes, resume=args.resume, strict=args.strict)
    print(f"Wrote {total} rows to {cfg['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
