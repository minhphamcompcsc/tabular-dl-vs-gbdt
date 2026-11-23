"""Report section 3.7: TabPFN (no tuning) vs tuned GBDT on small train sets.

Usage:
    python scripts/run_tabpfn.py
    python scripts/run_tabpfn.py --smoke --models dummy,linear   # pipeline check only
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
    args = parser.parse_args(argv)
    cfg = apply_cli_overrides(load_config("tabpfn"), args)
    n_trials = int(cfg["n_trials"])
    sizes = [None if s is None else int(s) for s in cfg["train_sizes"]]
    untuned = [m for m in cfg["models"] if family(m) in ("foundation", "baseline")]
    tuned = [m for m in cfg["models"] if m not in untuned]
    total = run_grid(cfg["datasets"], untuned, cfg["seeds"], cfg["output"], n_trials=0,
                     train_sizes=sizes, resume=args.resume, strict=args.strict)
    total += run_grid(cfg["datasets"], tuned, cfg["seeds"], cfg["output"],
                      params_for=lambda d, m: get_or_tune(d, m, n_trials, cfg.get("tuning_seed", 0)),
                      n_trials=n_trials, train_sizes=sizes, resume=args.resume, strict=args.strict)
    print(f"Wrote {total} rows to {cfg['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
