"""Report section 3.2: baselines with default params (no tuning).

Usage:
    python scripts/run_baseline.py               # real datasets (needs src/data.py)
    python scripts/run_baseline.py --smoke       # synthetic data, 3 seeds
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from src.config import add_common_args, apply_cli_overrides, load_config
from src.runner import run_grid


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args(argv)
    cfg = apply_cli_overrides(load_config("baseline"), args)
    n = run_grid(cfg["datasets"], cfg["models"], cfg["seeds"], cfg["output"], n_trials=0,
                 resume=args.resume, strict=args.strict)
    print(f"Wrote {n} rows to {cfg['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
