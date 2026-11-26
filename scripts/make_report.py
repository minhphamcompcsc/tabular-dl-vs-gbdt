"""Build Tables 2a, 2, 3, 4, 5 and Figures 2, 3, 4 from the results CSVs.

Inputs (whichever exist):  results/baseline.csv (3.2), results/full.csv (3.3, 3.4, 3.8),
                           results/ablation_budget.csv, results/size_curve.csv,
                           results/tabpfn.csv (3.7)
Outputs:                   results/tables/*.md|.tex, results/figures/*.png|.pdf

Each CSV is read on its own: a missing file only skips its own tables.

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

    # 3.2 Baseline: its own table, also when full.csv exists.
    if (rdir / "baseline.csv").exists():
        df = read_results(rdir / "baseline.csv")
        print(f"Table 2a (baseline) from {rdir / 'baseline.csv'} ({len(df)} rows)")
        made += plots.save_table(plots.table_main(df), tables, "table2a_baseline",
                                 "Baseline với tham số mặc định (không tuning), trung bình ± độ lệch chuẩn qua các seed")
    else:
        print(f"[skip] {rdir / 'baseline.csv'} not found")

    # 3.3, 3.4, 3.8: main results, tests, ranks, cost.
    if (rdir / "full.csv").exists():
        df = read_results(rdir / "full.csv")
        print(f"Tables 2-4 from {rdir / 'full.csv'} ({len(df)} rows)")
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
        print(f"[skip] {rdir / 'full.csv'} not found")

    for name, csv, fn in (("fig2_ablation_budget", "ablation_budget.csv", plots.fig_ablation_budget),
                          ("fig3_size_curve", "size_curve.csv", plots.fig_size_curve)):
        if (rdir / csv).exists():
            made += plots.save_figure(fn(read_results(rdir / csv)), figures, name)
        else:
            print(f"[skip] {rdir / csv} not found")

    # 3.7 TabPFN: one column per dataset and training-set size, plus a curve.
    if (rdir / "tabpfn.csv").exists():
        df = read_results(rdir / "tabpfn.csv")
        print(f"Table 5 and Figure 4 (TabPFN) from {rdir / 'tabpfn.csv'} ({len(df)} rows)")
        by_size = df.assign(dataset=df["dataset"] + " (n=" + df["n_train"].astype(str) + ")")
        t5 = plots.table_main(by_size, metrics=["roc_auc", "rmse"])  # primary metric only
        size_cols = sorted(t5.columns[1:], key=lambda c: (c.split(" (n=")[0], int(c.split("(n=")[1].split(")")[0])))
        t5 = t5[["Model", *size_cols]]
        made += plots.save_table(t5, tables, "table5_tabpfn",
                                 "TabPFN (không tuning) so với GBDT đã tuning theo kích thước tập train")
        made += plots.save_figure(plots.fig_size_curve(df), figures, "fig4_tabpfn")
    else:
        print(f"[skip] {rdir / 'tabpfn.csv'} not found")

    for p in made:
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
