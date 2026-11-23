"""Report section 3.5 (Figure 2): same models tuned with budgets 5, 15 and N.

Usage:
    python scripts/run_ablation_budget.py
    python scripts/run_ablation_budget.py --smoke
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
    cfg = apply_cli_overrides(load_config("ablation_budget"), args)
    budgets = sorted({int(b) if b is not None else int(cfg["n_trials"]) for b in cfg["budgets"]})
    total = 0
    for budget in budgets:
        print(f"== budget {budget} trials ==")
        total += run_grid(cfg["datasets"], cfg["models"], cfg["seeds"], cfg["output"],
                          params_for=lambda d, m, b=budget: get_or_tune(d, m, b, cfg.get("tuning_seed", 0)),
                          n_trials=budget, resume=args.resume, strict=args.strict)
    print(f"Wrote {total} rows to {cfg['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
