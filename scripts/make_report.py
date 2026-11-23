"""Build Tables 2, 3, 4 and Figures 2, 3 from the results CSVs.

Inputs (whichever exist):  results/full.csv (or results/baseline.csv),
                           results/ablation_budget.csv, results/size_curve.csv
Outputs:                   results/tables/*.md|.tex, results/figures/*.png|.pdf

Usage:
    python scripts/make_report.py
    python scripts/make_report.py --smoke          # reads results/smoke/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src import plots
from src.config import ROOT
from src.results_io import read_results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args(argv)
    rdir = Path(args.results_dir) if args.results_dir else ROOT / ("results/smoke" if args.smoke else "results")
    tables, figures = rdir / "tables", rdir / "figures"
    made: list[Path] = []

    main_csv = next((p for p in (rdir / "full.csv", rdir / "baseline.csv") if p.exists()), None)
    if main_csv is not None:
        df = read_results(main_csv)
        print(f"Tables 2-4 from {main_csv} ({len(df)} rows)")
        made += plots.save_table(plots.table_main(df), tables, "table2_main",
                                 "Kết quả chính trên tập test, trung bình ± độ lệch chuẩn qua các seed")
        tests = plots.table_tests(df)
        if not tests.empty:
            made += plots.save_table(tests, tables, "table3_tests",
                                     "Kiểm định Wilcoxon theo cặp seed, hiệu chỉnh Holm theo từng dataset")
        made += plots.save_table(plots.table_ranks(df), tables, "table3_ranks",
                                 "Xếp hạng trung bình theo thước đo chính (1 là tốt nhất)")
        made += plots.save_table(plots.table_cost(df), tables, "table4_cost",
                                 "Thời gian huấn luyện và số tham số")
    else:
        print(f"[skip] no full.csv or baseline.csv in {rdir}")

    for name, csv, fn in (("fig2_ablation_budget", "ablation_budget.csv", plots.fig_ablation_budget),
                          ("fig3_size_curve", "size_curve.csv", plots.fig_size_curve)):
        if (rdir / csv).exists():
            made += plots.save_figure(fn(read_results(rdir / csv)), figures, name)
        else:
            print(f"[skip] {rdir / csv} not found")

    for p in made:
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
