"""Known-answer tests for src/stats.py (the "Xong khi" criterion of part C)."""
import numpy as np
import pandas as pd
import pytest

from src import stats


def make_df(values: dict[str, list[float]], dataset="d", metric="roc_auc") -> pd.DataFrame:
    rows = []
    for model, vals in values.items():
        for seed, v in enumerate(vals):
            rows.append({"dataset": dataset, "model": model, "seed": seed, "n_trials": 0, "n_train": 100,
                         "metric_name": metric, "metric_value": v, "train_time_s": 1.0, "n_params": 10,
                         "hardware": "cpu", "git_commit": "x"})
    return pd.DataFrame(rows)


def test_holm_known_values():
    # sorted p: 0.01*3=0.03, 0.03*2=0.06, max(0.06, 0.04*1)=0.06
    assert np.allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(stats.holm([0.5, 0.9]), [1.0, 1.0])  # capped at 1
    assert len(stats.holm([])) == 0


def test_min_achievable_p():
    assert stats.min_achievable_p(3) == 0.25
    assert stats.min_achievable_p(5) == 0.0625
    assert stats.min_achievable_p(10) == pytest.approx(0.001953125)


def test_consistent_winner_over_10_seeds_is_significant():
    rng = np.random.default_rng(0)
    base = 0.80 + rng.normal(0, 0.01, 10)
    df = make_df({"mlp": base + 0.05 + rng.uniform(0, 0.01, 10), "xgboost": base})
    r = stats.compare(df, "mlp", "xgboost", "d", "roc_auc")
    assert r["n_seeds"] == 10
    assert r["p_raw"] == pytest.approx(2 / 2 ** 10)  # all 10 differences positive
    assert r["a_better"] and r["mean_diff"] > 0.05


def test_identical_models_give_p_one():
    v = [0.8, 0.81, 0.79, 0.8, 0.82]
    r = stats.compare(make_df({"mlp": v, "xgboost": v}), "mlp", "xgboost", "d", "roc_auc")
    assert r["p_raw"] == 1.0


def test_three_seeds_can_never_be_significant():
    df = make_df({"mlp": [0.9, 0.9, 0.9], "xgboost": [0.5, 0.5, 0.5]})
    r = stats.compare(df, "mlp", "xgboost", "d", "roc_auc")
    assert r["p_raw"] >= 0.25


def test_rmse_lower_is_better():
    df = make_df({"mlp": [1.0] * 10, "xgboost": [2.0] * 10}, metric="rmse")
    r = stats.compare(df, "mlp", "xgboost", "d", "rmse")
    assert r["mean_diff"] == pytest.approx(-1.0) and r["a_better"]


def test_compare_to_reference_picks_best_gbdt_and_applies_holm():
    rng = np.random.default_rng(1)
    noise = lambda: rng.normal(0, 0.002, 10)  # noqa: E731
    df = make_df({
        "xgboost": 0.85 + noise(),
        "catboost": 0.87 + noise(),        # best GBDT
        "mlp": 0.80 + noise(),             # clearly worse than catboost
        "ft_transformer": 0.87 + noise(),  # about equal to catboost
    })
    t = stats.compare_to_reference(df).set_index("model_a")
    assert set(t["model_b"]) == {"catboost"}
    assert t.at["mlp", "winner"] == "catboost"
    assert t.at["mlp", "p_holm"] == pytest.approx(min(1.0, 2 * t.at["mlp", "p_raw"]))  # smallest of 2 p-values
    assert t.at["ft_transformer", "p_holm"] >= t.at["ft_transformer", "p_raw"]


def test_missing_seeds_are_dropped_with_warning():
    df = make_df({"mlp": [0.9] * 10, "xgboost": [0.8] * 10})
    df = df[~((df["model"] == "xgboost") & (df["seed"] == 9))]
    with pytest.warns(UserWarning):
        r = stats.compare(df, "mlp", "xgboost", "d", "roc_auc")
    assert r["n_seeds"] == 9


def test_average_rank():
    df = pd.concat([
        make_df({"a": [0.9] * 3, "b": [0.8] * 3}, dataset="d1"),
        make_df({"a": [2.0] * 3, "b": [1.0] * 3}, dataset="d2", metric="rmse"),
    ])
    r = stats.average_rank(df).set_index("model")["avg_rank"]
    assert r["a"] == 1.5 and r["b"] == 1.5


def test_summarize_mean_std():
    s = stats.summarize(make_df({"a": [1.0, 2.0, 3.0]}))
    row = s.iloc[0]
    assert row["mean"] == 2.0 and row["std"] == pytest.approx(1.0) and row["n_seeds"] == 3
