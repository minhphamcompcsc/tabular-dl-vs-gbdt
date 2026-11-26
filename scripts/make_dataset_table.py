"""Build Table 1 (dataset description) by loading every dataset once.

Unlike make_report.py this script needs the data itself (src/data.py downloads
UCI and scikit-learn datasets), so run it where internet access is available.

Outputs: results/tables/table1_datasets.md|.tex (results/smoke/tables/ with --smoke)

Usage:
    python scripts/make_dataset_table.py
    python scripts/make_dataset_table.py --smoke                 # synthetic datasets
    python scripts/make_dataset_table.py --datasets adult,german_credit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src import plots
from src.config import ROOT, SMOKE_DATASETS, load_config
from src.runner import data_api


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--datasets", default=None, help="Comma-separated names (default: all in experiment.yaml).")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args(argv)

    if args.datasets:
        names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        names = list(SMOKE_DATASETS) if args.smoke else list(load_config("full")["datasets"])
    rdir = Path(args.results_dir) if args.results_dir else ROOT / ("results/smoke" if args.smoke else "results")

    entries = []
    for name in names:
        print(f"[load] {name}")
        entries.append(data_api(name).load_dataset(name))
    paths = plots.save_table(plots.table_datasets(entries), rdir / "tables", "table1_datasets",
                             "Mô tả các tập dữ liệu")
    for p in paths:
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
