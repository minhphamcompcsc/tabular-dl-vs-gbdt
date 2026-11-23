"""End-to-end: scripts run on synthetic data and write the agreed CSV format."""
import importlib.util
from pathlib import Path

import pandas as pd

from src.interfaces import RESULT_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_baseline_smoke_writes_contract_csv_and_resumes(tmp_path):
    out = tmp_path / "baseline.csv"
    args = ["--smoke", "--seeds", "0,1", "--models", "dummy,linear", "--out", str(out)]
    assert _script("run_baseline").main(args) == 0
    df = pd.read_csv(out)
    assert list(df.columns) == RESULT_COLUMNS
    # 2 datasets x 2 models x 2 seeds; binary has 2 metrics, regression 1
    assert len(df) == 2 * 2 * 2 + 2 * 2 * 1
    assert set(df["metric_name"]) == {"roc_auc", "accuracy", "rmse"}
    # the linear baseline must beat the constant predictor
    auc = df[df["metric_name"] == "roc_auc"].groupby("model")["metric_value"].mean()
    assert auc["linear"] > auc["dummy"] + 0.1
    # running again with resume adds nothing
    _script("run_baseline").main(args)
    assert len(pd.read_csv(out)) == len(df)


def test_stub_models_are_skipped_not_crashing(tmp_path):
    out = tmp_path / "full.csv"
    rc = _script("run_full").main(["--smoke", "--seeds", "0", "--models", "linear,xgboost,mlp", "--out", str(out)])
    assert rc == 0
    models = set(pd.read_csv(out)["model"])
    assert "linear" in models  # xgboost / mlp are skipped while they are stubs


def test_report_from_smoke_results(tmp_path):
    for name in ("run_full", "run_ablation_budget", "run_size_curve"):
        exp = name.removeprefix("run_")
        _script(name).main(["--smoke", "--seeds", "0,1,2", "--out", str(tmp_path / f"{exp}.csv")])
    assert _script("make_report").main(["--results-dir", str(tmp_path)]) == 0
    for f in ("tables/table2_main.md", "tables/table3_tests.tex", "tables/table4_cost.md",
              "figures/fig2_ablation_budget.png", "figures/fig3_size_curve.pdf"):
        assert (tmp_path / f).exists(), f
