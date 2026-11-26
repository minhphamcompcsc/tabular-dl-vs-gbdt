"""Table 1 (dataset description) on the synthetic datasets."""
import numpy as np

from src import fakes, plots


def test_table_datasets_on_fakes():
    entries = [fakes.load_dataset(n) for n in fakes.FAKE_DATASETS]
    t = plots.table_datasets(entries)
    assert list(t["Dataset"]) == list(fakes.FAKE_DATASETS)
    binary = t.set_index("Dataset").loc["fake_binary"]
    _, y, _ = fakes.load_dataset("fake_binary")
    assert binary["Tỉ lệ lớp 1"] == f"{np.mean(y == 1):.3f}"
    assert binary["Cột số"] == fakes.N_NUM and binary["Cột phân loại"] == 1
    assert binary["Số dòng"] == f"{fakes.N_SAMPLES:,}"
    assert t.set_index("Dataset").loc["fake_regression", "Tỉ lệ lớp 1"] == "-"


def test_table_datasets_counts_missing_and_blank_license():
    from dataclasses import replace

    X, y, meta = fakes.load_dataset("fake_binary")
    X.iloc[:300, 0] = np.nan                 # 300 of 3000 x 7 cells
    meta = replace(meta, license=" ")        # do not touch the cached meta
    row = plots.table_datasets([(X, y, meta)]).iloc[0]
    assert row["Ô thiếu (%)"] == f"{100 * 300 / (fakes.N_SAMPLES * 7):.2f}"
    assert row["License"] == "-"


def test_make_dataset_table_smoke(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "make_dataset_table.py"
    import sys
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("make_dataset_table", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--smoke", "--results-dir", str(tmp_path)]) == 0
    md = (tmp_path / "tables" / "table1_datasets.md").read_text(encoding="utf-8")
    assert "fake_binary" in md and "fake_regression" in md
    assert (tmp_path / "tables" / "table1_datasets.tex").exists()
