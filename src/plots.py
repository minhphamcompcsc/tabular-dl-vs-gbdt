"""Tables and figures for the report, built only from results CSVs.

OWNER: C (Minh).

Tables (Markdown + LaTeX):
    Table 1  dataset description (rows, columns, class ratio)  -> table1_datasets
    Table 2  main results, mean +/- std over seeds         -> table2_main
    Table 3  Holm-corrected tests vs best GBDT + avg rank  -> table3_tests, table3_ranks
    Table 4  training time and number of parameters        -> table4_cost
Figures (PNG + PDF):
    Figure 2 tuning-budget ablation                        -> fig2_ablation_budget
    Figure 3 performance vs training-set size              -> fig3_size_curve

Colors follow the model, never its rank, so a model has the same color in
every figure. GBDT lines are dashed with square markers, deep learning lines
solid with circles, so identity never relies on color alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import stats  # noqa: E402
from src.metrics import HIGHER_IS_BETTER  # noqa: E402
from src.models import family  # noqa: E402

# Fixed model -> color (validated categorical palette, light mode).
MODEL_COLORS = {
    "dummy": "#8a8a85",
    "linear": "#e34948",
    "xgboost": "#2a78d6",
    "lightgbm": "#eb6834",
    "catboost": "#1baf7a",
    "mlp": "#eda100",
    "resnet": "#e87ba4",
    "ft_transformer": "#008300",
    "tabpfn": "#6250d6",
    # smoke-test stand-ins borrow the color of the model they stand in for
    "sk_hgb": "#2a78d6",
    "sk_mlp": "#eda100",
}
TEXT = "#2b2b29"
MUTED = "#6b6b66"
GRID = "#e4e3df"
METRIC_LABEL = {"roc_auc": "ROC-AUC (cao hơn tốt hơn)", "accuracy": "Accuracy (cao hơn tốt hơn)",
                "rmse": "RMSE (thấp hơn tốt hơn)"}


# ------------------------------------------------------------------ tables

def _fmt(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    def esc(v) -> str:
        s = str(v).replace("±", "$\\pm$").replace("**", "")
        for ch in ("_", "%", "&", "#"):
            s = s.replace(ch, "\\" + ch)
        return s

    bold_cells = df.map(lambda v: str(v).startswith("**")) if hasattr(df, "map") else df.applymap(
        lambda v: str(v).startswith("**"))
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{esc(caption)}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{" + "l" * len(df.columns) + "}", "\\toprule",
             " & ".join(esc(c) for c in df.columns) + " \\\\", "\\midrule"]
    for i, row in enumerate(df.itertuples(index=False)):
        cells = [f"\\textbf{{{esc(v)}}}" if bold_cells.iat[i, j] else esc(v) for j, v in enumerate(row)]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def table_main(df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Table 2: rows = models, columns = dataset/metric, cells = mean ± std.

    The best mean in each column is wrapped in ** ** (bold in LaTeX).
    """
    s = stats.summarize(df)
    if metrics:
        s = s[s["metric_name"].isin(metrics)]
    s = s.assign(col=s["dataset"] + " / " + s["metric_name"])
    primary_first = {"roc_auc": 0, "rmse": 0, "accuracy": 1}
    cols = [c for _, c in sorted({(d, primary_first.get(m, 2), m): f"{d} / {m}"
                                  for d, m in zip(s["dataset"], s["metric_name"])}.items())]
    order = _model_order(s["model"].unique())
    out = pd.DataFrame({"Model": order})
    for col in cols:
        sub = s[s["col"] == col].set_index("model")
        metric = sub["metric_name"].iloc[0]
        best = sub["mean"].idxmax() if HIGHER_IS_BETTER[metric] else sub["mean"].idxmin()
        values = []
        for m in order:
            if m not in sub.index:
                values.append("")
                continue
            cell = _fmt(sub.at[m, "mean"], sub.at[m, "std"])
            values.append(f"**{cell}**" if m == best else cell)
        out[col] = values
    return out


def table_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Table 3a: each deep learning model vs the best GBDT, Holm per dataset."""
    t = stats.compare_to_reference(df)
    if t.empty:
        return t
    return pd.DataFrame({
        "Dataset": t["dataset"],
        "Metric": t["metric"],
        "DL model": t["model_a"],
        "Best GBDT": t["model_b"],
        "Mean diff (DL - GBDT)": t["mean_diff"].map(lambda v: f"{v:+.4f}"),
        "p (Holm)": t["p_holm"].map(lambda v: f"{v:.4f}"),
        "Seeds": t["n_seeds"],
        "Kết luận": t["winner"].map(lambda w: "khác biệt không có ý nghĩa" if w == "tie" else f"{w} tốt hơn"),
    })


def table_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Table 3b: average rank on the primary metric (1 = best)."""
    r = stats.average_rank(df)
    return pd.DataFrame({"Model": r["model"], "Avg rank": r["avg_rank"].map(lambda v: f"{v:.2f}"),
                         "Datasets": r["n_datasets"]})


def table_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Table 4: mean ± std training time (s) and number of parameters."""
    first_metric = df.groupby(["dataset", "model", "seed", "n_trials", "n_train"], as_index=False).first()
    g = first_metric.groupby(["dataset", "model"], as_index=False).agg(
        t_mean=("train_time_s", "mean"), t_std=("train_time_s", "std"),
        n_params=("n_params", "median"), hardware=("hardware", "first"))
    g["t_std"] = g["t_std"].fillna(0.0)
    g = g.assign(order=g["model"].map({m: i for i, m in enumerate(_model_order(g["model"].unique()))}))
    g = g.sort_values(["dataset", "order"])
    return pd.DataFrame({
        "Dataset": g["dataset"],
        "Model": g["model"],
        "Train time (s)": [_fmt(a, b, 3) for a, b in zip(g["t_mean"], g["t_std"])],
        "Params": g["n_params"].map(lambda v: f"{int(v):,}"),
        "Hardware": g["hardware"],
    })


def table_datasets(entries) -> pd.DataFrame:
    """Table 1: one row per dataset from ``(X, y, meta)`` as returned by ``load_dataset``.

    Columns: task, rows, numeric and categorical columns, share of missing
    cells, positive-class ratio (binary only), source and license.
    """
    rows = []
    for X, y, meta in entries:
        y = np.asarray(y)
        rows.append({
            "Dataset": meta.name,
            "Tác vụ": "Phân loại nhị phân" if meta.task == "binary" else "Hồi quy",
            "Số dòng": f"{len(X):,}",
            "Cột số": len(meta.num_cols),
            "Cột phân loại": len(meta.cat_cols),
            "Ô thiếu (%)": f"{100 * float(X.isna().to_numpy().mean()):.2f}",
            "Tỉ lệ lớp 1": f"{float((y == 1).mean()):.3f}" if meta.task == "binary" else "-",
            "Nguồn": meta.source or "-",
            "License": (meta.license or "").strip() or "-",
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- figures

def _model_order(models) -> list[str]:
    known = [m for m in MODEL_COLORS if m in set(models)]
    return known + sorted(set(models) - set(known))


def _style(model: str) -> dict:
    fam = family(model)
    return {
        "color": MODEL_COLORS.get(model, MUTED),
        "linestyle": "--" if fam == "gbdt" else "-",
        "marker": "s" if fam == "gbdt" else "o",
        "linewidth": 2,
        "markersize": 6,
    }


def _setup_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def _line_panel(ax, s: pd.DataFrame, x: str, log_x: bool) -> list:
    models = _model_order(s["model"].unique())
    ends = []
    for m in models:
        sub = s[s["model"] == m].sort_values(x)
        st = _style(m)
        ax.plot(sub[x], sub["mean"], label=m, **st)
        ax.fill_between(sub[x], sub["mean"] - sub["std"], sub["mean"] + sub["std"],
                        color=st["color"], alpha=0.12, linewidth=0)
        if not sub.empty:
            ends.append((m, sub[x].iloc[-1], sub["mean"].iloc[-1]))
    if log_x:
        ax.set_xscale("log")
    xs = sorted(s[x].unique())
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(v):,}" for v in xs])
    ax.minorticks_off()
    return ends


def _direct_labels(ax, ends, min_gap_pt: float = 11.0) -> None:
    """Label line ends; labels are pushed apart vertically so they never overlap.
    Call after the layout is final (tight_layout changes the transforms)."""
    ax.figure.canvas.draw()
    min_gap_px = min_gap_pt * ax.figure.dpi / 72.0
    pts = sorted(((ax.transData.transform((x, y)), m) for m, x, y in ends), key=lambda t: t[0][1])
    placed = []
    for (px, py), m in pts:
        if placed and py - placed[-1][1] < min_gap_px:
            py = placed[-1][1] + min_gap_px
        placed.append((px, py, m))
    inv = ax.transData.inverted()
    for px, py, m in placed:
        x, y = inv.transform((px, py))
        ax.annotate(m, (x, y), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=8, color=TEXT, annotation_clip=False)


def _facet_figure(df: pd.DataFrame, x: str, title: str, xlabel: str, log_x: bool):
    s = stats.summarize(df)
    datasets = sorted(s["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 3.6), squeeze=False)
    all_ends = []
    for ax, dataset in zip(axes[0], datasets):
        metric = stats.primary_metric_of(df, dataset)
        sub = s[(s["dataset"] == dataset) & (s["metric_name"] == metric)]
        all_ends.append(_line_panel(ax, sub, x, log_x))
        _setup_axes(ax, dataset, xlabel, METRIC_LABEL.get(metric, metric))
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False, fontsize=9,
               labelcolor=TEXT, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 0.95, 0.95), w_pad=6)  # room for direct labels
    for ax, ends in zip(axes[0], all_ends):
        if 0 < len(ends) <= 4:  # direct labels only when few series
            _direct_labels(ax, ends)
    return fig


def fig_ablation_budget(df: pd.DataFrame):
    """Figure 2: primary metric vs number of tuning trials."""
    return _facet_figure(df, "n_trials", "Hình 2. Ảnh hưởng của ngân sách tuning",
                         "Số lần thử tuning", log_x=False)


def fig_size_curve(df: pd.DataFrame):
    """Figure 3: primary metric vs number of training samples (log scale)."""
    return _facet_figure(df, "n_train", "Hình 3. Hiệu năng theo kích thước tập train",
                         "Số mẫu train (thang log)", log_x=True)


def save_figure(fig, out_dir: str | Path, name: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def save_table(table: pd.DataFrame, out_dir: str | Path, name: str, caption: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md, tex = out_dir / f"{name}.md", out_dir / f"{name}.tex"
    md.write_text(to_markdown(table), encoding="utf-8")
    tex.write_text(to_latex(table, caption, f"tab:{name}"), encoding="utf-8")
    return [md, tex]


__all__ = [
    "table_datasets", "table_main", "table_tests", "table_ranks", "table_cost",
    "fig_ablation_budget", "fig_size_curve", "save_figure", "save_table",
]
