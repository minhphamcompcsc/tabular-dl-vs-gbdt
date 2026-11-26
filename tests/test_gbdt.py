"""Exercise real GBDT early stopping without downloading datasets."""
import numpy as np
import pytest

from src.models import build_model
from src.runner import best_params_path


@pytest.mark.parametrize("name", ["xgboost", "lightgbm", "catboost"])
@pytest.mark.parametrize("task", ["binary", "regression"])
def test_validation_stops_training_and_selects_best_trees(name, task):
    pytest.importorskip(name)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(400, 4))
    # Validation deliberately contradicts the training relationship, so
    # additional boosting rounds make it worse and should trigger stopping.
    y = (X[:, 0] > 0).astype(int) if task == "binary" else X[:, 0] * 3
    y_val = 1 - y[300:] if task == "binary" else -y[300:]
    limit = 80
    params = {"early_stopping_rounds": 5, "learning_rate": 0.1}
    params["iterations" if name == "catboost" else "n_estimators"] = limit
    model = build_model(name, task, params, seed=0)
    model.fit(X[:300], y[:300], X[300:], y_val)
    estimator = model.estimator

    if name == "xgboost":
        best_rounds = estimator.best_iteration + 1
        assert estimator.get_booster().num_boosted_rounds() == best_rounds
        trees = estimator.get_booster().trees_to_dataframe()
        expected_nodes = len(trees)
        expected = (estimator.predict_proba(X[300:], iteration_range=(0, best_rounds))[:, 1]
                    if task == "binary" else
                    estimator.predict(X[300:], iteration_range=(0, best_rounds)))
    elif name == "lightgbm":
        best_rounds = estimator.best_iteration_
        trees = estimator.booster_.dump_model(num_iteration=best_rounds)["tree_info"]
        assert len(trees) == best_rounds
        expected_nodes = sum(2 * tree["num_leaves"] - 1 for tree in trees)
        expected = (estimator.predict_proba(X[300:], num_iteration=best_rounds)[:, 1]
                    if task == "binary" else
                    estimator.predict(X[300:], num_iteration=best_rounds))
    else:
        best_rounds = estimator.get_best_iteration() + 1
        assert estimator.tree_count_ == best_rounds
        expected_nodes = sum(2 * int(n) - 1 for n in estimator.get_tree_leaf_counts())
        expected = (estimator.predict_proba(X[300:], ntree_end=best_rounds)[:, 1]
                    if task == "binary" else
                    estimator.predict(X[300:], ntree_end=best_rounds))

    assert 1 <= best_rounds < limit
    assert model.n_params == expected_nodes > 0
    pred = model.predict(X[300:])
    assert pred.shape == (100,)
    assert np.isfinite(pred).all()
    np.testing.assert_allclose(pred, expected)
    if task == "binary":
        assert ((0 <= pred) & (pred <= 1)).all()


def test_gbdt_cache_is_separate_from_pre_early_stopping_cache():
    from src.config import gbdt_early_stopping_rounds

    path = best_params_path("adult", "xgboost", 30)
    assert path.name == f"adult__xgboost__n30__es{gbdt_early_stopping_rounds()}_v1.json"
    assert best_params_path("adult", "mlp", 30).name == "adult__mlp__n30.json"
