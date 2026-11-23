"""Contract tests. They run against the synthetic data now, and against A's
src/data.py and every registered model as soon as those are implemented
(until then they are reported as skipped, not failed)."""
import numpy as np
import pytest

from src import fakes
from src.interfaces import DatasetMeta, Split
from src.models import REGISTRY, build_model


def _data_modules():
    mods = [("fakes", fakes, "fake_binary"), ("fakes", fakes, "fake_regression")]
    from src import data
    mods += [("data", data, name) for name in data.DATASETS]
    return mods


@pytest.mark.parametrize("label,mod,name", _data_modules(), ids=lambda v: v if isinstance(v, str) else "")
def test_data_contract(label, mod, name):
    try:
        X, y, meta = mod.load_dataset(name)
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    assert isinstance(meta, DatasetMeta) and meta.task in ("binary", "regression")
    assert len(X) == len(y)
    if meta.task == "binary":
        assert set(np.unique(y)) <= {0, 1}
    split = mod.make_split(X, y, seed=0)
    assert isinstance(split, Split)
    n = len(y)
    assert abs(len(split.train.y) / n - 0.6) < 0.02
    assert abs(len(split.val.y) / n - 0.2) < 0.02
    assert abs(len(split.test.y) / n - 0.2) < 0.02
    again = mod.make_split(X, y, seed=0)
    assert np.array_equal(split.test.y, again.test.y), "make_split must be deterministic per seed"
    small = mod.make_split(X, y, seed=0, train_size=100)
    assert len(small.train.y) == 100
    assert np.array_equal(small.test.y, split.test.y), "train_size must not change the test part"


@pytest.mark.parametrize("model_name", sorted(REGISTRY))
@pytest.mark.parametrize("dataset", fakes.FAKE_DATASETS)
def test_model_contract(model_name, dataset):
    X, y, meta = fakes.load_dataset(dataset)
    split = fakes.make_split(X, y, seed=0, train_size=300)
    model = build_model(model_name, meta.task, seed=0)
    pre = fakes.build_preprocessor(model.preprocess_kind, meta).fit(split.train.X)
    Xtr, Xva, Xte = (pre.transform(p.X) for p in (split.train, split.val, split.test))
    try:
        model.fit(Xtr, split.train.y, Xva, split.val.y)
    except (NotImplementedError, ImportError) as exc:
        pytest.skip(f"{model_name}: {exc}")
    pred = np.asarray(model.predict(Xte))
    assert pred.shape == (len(split.test.y),)
    assert np.all(np.isfinite(pred))
    if meta.task == "binary":
        assert pred.min() >= 0 and pred.max() <= 1, "binary models must return P(y=1)"
    assert isinstance(model.n_params, int) and model.n_params > 0
    assert model.preprocess_kind in ("nn", "gbdt")
