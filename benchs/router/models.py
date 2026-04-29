"""
CatBoost-based models for the ANN router.

CatBoostIndexSelector
    Given (meta-features + target_recall + index), predicts log(QPS).
    At inference, all candidate indexes are scored; argmax is returned.

ParameterRecommender
    One CatBoostRegressor per index type.
    Given (meta-features + target_recall), predicts the optimal search parameter.
    Trained only on rows where the index can actually reach the target recall.
"""

import numpy as np
import pandas as pd

from .features import engineer_features, selector_feature_cols, recommender_feature_cols

try:
    from catboost import CatBoostRegressor, Pool
    _CATBOOST = True
except ImportError:
    _CATBOOST = False

# Indexes whose parameter is a small ratio → log-transform target for better fit
_LOG_PARAM_INDEXES = {"SuCo"}

_SELECTOR_CATBOOST_PARAMS = dict(
    iterations=500,
    learning_rate=0.05,
    depth=4,
    l2_leaf_reg=5,
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
)

_RECOMMENDER_CATBOOST_PARAMS = dict(
    iterations=300,
    learning_rate=0.05,
    depth=4,
    l2_leaf_reg=5,
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
)


def _require_catboost():
    if not _CATBOOST:
        raise ImportError("catboost is required: pip install catboost")


class CatBoostIndexSelector:
    """
    Regressor that predicts log(QPS+1) for each (dataset, index) pair.

    The index column is treated as a CatBoost categorical feature — no manual
    one-hot encoding needed. At inference, pass all candidate indexes and
    return argmax over predicted QPS.
    """

    def __init__(self):
        _require_catboost()
        self._model = CatBoostRegressor(
            cat_features=["index"],
            **_SELECTOR_CATBOOST_PARAMS,
        )
        self._feature_cols = selector_feature_cols()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "CatBoostIndexSelector":
        """
        df must contain:
            meta columns, 'index' (categorical), 'target_recall', 'qps_at_target_recall'
        Rows with qps_at_target_recall == 0 (unreachable recall) are included so the
        model learns that some indexes fail on certain datasets.
        """
        df = engineer_features(df)
        X = df[self._feature_cols + ["index"]].copy()
        y = np.log1p(df["qps_at_target_recall"].values.astype(float))
        pool = Pool(X, y, cat_features=["index"])
        self._model.fit(pool)
        self._fitted = True
        return self

    def predict_qps(self, df: pd.DataFrame) -> np.ndarray:
        """Returns predicted QPS (un-log-transformed) for each row."""
        df = engineer_features(df)
        X = df[self._feature_cols + ["index"]].copy()
        pool = Pool(X, cat_features=["index"])
        log_qps = self._model.predict(pool)
        return np.expm1(log_qps)

    def select_index(
        self,
        meta_features: dict,
        target_recall: float,
        candidate_indexes: list[str],
    ) -> tuple[str, float]:
        """
        Given a single dataset's meta_features dict and a target_recall,
        returns (best_index_name, predicted_qps).
        """
        rows = []
        for idx in candidate_indexes:
            row = {**meta_features, "target_recall": target_recall, "index": idx}
            rows.append(row)
        df = pd.DataFrame(rows)
        qps = self.predict_qps(df)
        best_i = int(np.argmax(qps))
        return candidate_indexes[best_i], float(qps[best_i])

    def feature_importances(self) -> pd.Series:
        cols = self._feature_cols + ["index"]
        return pd.Series(
            self._model.get_feature_importance(),
            index=cols,
        ).sort_values(ascending=False)


class ParameterRecommender:
    """
    One CatBoostRegressor per index type.

    For SuCo the param (candidate_ratio) is log-transformed before fitting.
    For graph-based indexes (efSearch) it is used on the linear scale.
    """

    def __init__(self):
        _require_catboost()
        self._models: dict[str, CatBoostRegressor] = {}
        self._log_transform: dict[str, bool] = {}
        self._feature_cols = recommender_feature_cols()
        self._fitted_indexes: list[str] = []

    def fit(self, df: pd.DataFrame) -> "ParameterRecommender":
        """
        df must contain:
            meta columns, 'index', 'target_recall',
            'param_at_target_recall', 'qps_at_target_recall'
        Only rows where the index could reach the target recall (qps > 0) are used.
        """
        df = engineer_features(df)
        trainable = df[df["qps_at_target_recall"] > 0].copy()

        for index_name, grp in trainable.groupby("index"):
            if len(grp) < 3:
                # Not enough data to train a meaningful model for this index
                continue

            X = grp[self._feature_cols]
            use_log = index_name in _LOG_PARAM_INDEXES
            y_raw = grp["param_at_target_recall"].values.astype(float)
            y = np.log(y_raw + 1e-12) if use_log else y_raw

            model = CatBoostRegressor(**_RECOMMENDER_CATBOOST_PARAMS)
            model.fit(X, y, verbose=0)

            self._models[index_name] = model
            self._log_transform[index_name] = use_log
            self._fitted_indexes.append(index_name)

        return self

    def predict_param(
        self,
        index_name: str,
        meta_features: dict,
        target_recall: float,
    ) -> float | None:
        """
        Returns the recommended search parameter for index_name on a dataset
        described by meta_features, targeting target_recall.

        Returns None if no model is available for this index.
        """
        if index_name not in self._models:
            return None

        row = {**meta_features, "target_recall": target_recall}
        df = engineer_features(pd.DataFrame([row]))
        X = df[self._feature_cols]

        pred = float(self._models[index_name].predict(X)[0])
        if self._log_transform[index_name]:
            pred = float(np.exp(pred))
        return pred

    @property
    def fitted_indexes(self) -> list[str]:
        return list(self._fitted_indexes)
