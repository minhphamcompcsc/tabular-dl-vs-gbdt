"""XGBoost, LightGBM and CatBoost behind one interface.

OWNER: A (Huấn). Validation-based early stopping with AUC or RMSE.
Only the best validation iteration is used for prediction and node counts.
"""
from __future__ import annotations

import numpy as np

from src.config import gbdt_early_stopping_rounds


class _GBDTModel:
    preprocess_kind = "gbdt"
    device = "cpu"

    def __init__(self, task: str, seed: int = 0,
                 early_stopping_rounds: int | None = None, **params):
        if task not in {"binary", "regression"}:
            raise ValueError(f"Task không được hỗ trợ: {task}")
        self.task = task
        self.seed = seed
        self.params = params
        self.early_stopping_rounds = (
            gbdt_early_stopping_rounds() if early_stopping_rounds is None
            else early_stopping_rounds
        )
        if type(self.early_stopping_rounds) is not int or self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be a positive integer")
        self.n_params = 0
        self.estimator = self._build_estimator()

    def fit(self, X_tr, y_tr, X_val, y_val):
        self._fit(X_tr, y_tr, X_val, y_val)
        self.n_params = sum(
            2 * int(n_leaves) - 1 for n_leaves in self._leaf_counts()
        )
        return self

    def predict(self, X) -> np.ndarray:
        if self.task == "binary":
            return np.asarray(self.estimator.predict_proba(X)[:, 1])
        return np.asarray(self.estimator.predict(X))


class XGBoostModel(_GBDTModel):
    def _fit(self, X_tr, y_tr, X_val, y_val):
        from xgboost.callback import EarlyStopping

        # A fresh callback is needed for every fit, including repeated fits.
        self.estimator.set_params(callbacks=[EarlyStopping(
            rounds=self.early_stopping_rounds, save_best=True,
        )])
        self.estimator.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    def _build_estimator(self):
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBClassifier if self.task == "binary" else XGBRegressor
        options = {"n_jobs": 4, "tree_method": "hist", **self.params}
        options.update(random_state=self.seed, device="cpu",
                       eval_metric="auc" if self.task == "binary" else "rmse")
        return cls(**options)

    def _leaf_counts(self):
        trees = self.estimator.get_booster().trees_to_dataframe()
        leaves = trees.loc[trees["Feature"] == "Leaf"]
        return leaves.groupby("Tree").size()


class LightGBMModel(_GBDTModel):
    def _fit(self, X_tr, y_tr, X_val, y_val):
        from lightgbm import early_stopping

        self.estimator.fit(
            X_tr, y_tr, eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(self.early_stopping_rounds,
                                      first_metric_only=True, verbose=False)],
        )

    def _build_estimator(self):
        from lightgbm import LGBMClassifier, LGBMRegressor

        cls = LGBMClassifier if self.task == "binary" else LGBMRegressor
        options = {"n_jobs": 4, "verbosity": -1, **self.params}
        options.update(random_state=self.seed, device_type="cpu",
                       metric="auc" if self.task == "binary" else "rmse")
        return cls(**options)

    def _leaf_counts(self):
        model = self.estimator.booster_.dump_model(
            num_iteration=self.estimator.best_iteration_,
        )
        return [tree["num_leaves"] for tree in model["tree_info"]]


class CatBoostModel(_GBDTModel):
    def _fit(self, X_tr, y_tr, X_val, y_val):
        self.estimator.fit(
            X_tr, y_tr, eval_set=(X_val, y_val),
            early_stopping_rounds=self.early_stopping_rounds, use_best_model=True,
        )

    def _build_estimator(self):
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls = CatBoostClassifier if self.task == "binary" else CatBoostRegressor
        options = {"thread_count": 4, "verbose": False, **self.params}
        options.update(
            random_seed=self.seed,
            task_type="CPU",
            allow_writing_files=False,
            eval_metric="AUC" if self.task == "binary" else "RMSE",
        )
        return cls(**options)

    def _leaf_counts(self):
        return self.estimator.get_tree_leaf_counts()
