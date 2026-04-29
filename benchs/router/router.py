"""
ANNRouter: fit once on benchmark results, predict at inference.

Usage:
    router = ANNRouter()
    router.fit("benchs/results_router")
    result = router.predict(
        meta_features={'n': 1_000_000, 'd': 128, 'lid_mle': 18.8,
                       'pdist_mean': 538.0, 'pdist_std': 107.0,
                       'kmeans_inertia_ratio_16': 0.51},
        target_recall=0.90,
    )
    # result = {'index': 'SuCo', 'param': 0.005, 'param_name': 'candidate_ratio',
    #           'predicted_qps': 2400.0}

    router.save("router_model.pkl")
    router2 = ANNRouter.load("router_model.pkl")
"""

import joblib
import numpy as np
import pandas as pd

from .data_loader import load_interpolated, RECALL_TARGETS
from .models import CatBoostIndexSelector, ParameterRecommender

_PARAM_NAMES = {
    "SuCo":   "candidate_ratio",
    "HNSW32": "efSearch",
    "HNSW48": "efSearch",
    "CSPG":   "efSearch",
    "SHG":    "efSearch",
}


class ANNRouter:
    """
    Two-stage router:
      1. CatBoostIndexSelector: pick the best index for a dataset + recall target.
      2. ParameterRecommender: recommend the optimal search parameter for that index.
    """

    def __init__(self):
        self._selector   = CatBoostIndexSelector()
        self._recommender = ParameterRecommender()
        self._candidate_indexes: list[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        results_dir: str,
        recall_targets: list[float] | None = None,
        k: int = 20,
    ) -> "ANNRouter":
        """
        Fit both sub-models on all JSON files found in results_dir.

        results_dir : path to benchs/results_router/
        recall_targets : list of recall thresholds to use (default: RECALL_TARGETS)
        k : recall@k used for the recall-QPS curves (default: 20)
        """
        if recall_targets is None:
            recall_targets = RECALL_TARGETS

        df = load_interpolated(results_dir, recall_targets=recall_targets, k=k)
        self._candidate_indexes = sorted(df["index"].unique().tolist())

        self._selector.fit(df)
        self._recommender.fit(df)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        meta_features: dict,
        target_recall: float,
        candidate_indexes: list[str] | None = None,
        data_sample: np.ndarray | None = None,
    ) -> dict:
        """
        Select the best ANN index and its optimal parameter for a new dataset.

        Parameters
        ----------
        meta_features : dict
            Must contain at minimum {'n', 'd'}.
            If lid_mle / pdist_mean / pdist_std / kmeans_inertia_ratio_16 are absent
            AND data_sample is provided, they are computed on the fly.
        target_recall : float
            Desired recall level (e.g. 0.9).
        candidate_indexes : list[str] | None
            Subset of indexes to consider. Defaults to all seen during fit().
        data_sample : np.ndarray | None
            Raw data matrix [m, d] for on-the-fly meta-feature computation.

        Returns
        -------
        dict with keys:
            index, param, param_name, predicted_qps
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        meta_features = dict(meta_features)
        _fill_missing_meta(meta_features, data_sample)

        if candidate_indexes is None:
            candidate_indexes = self._candidate_indexes

        best_index, predicted_qps = self._selector.select_index(
            meta_features, target_recall, candidate_indexes
        )

        param = self._recommender.predict_param(best_index, meta_features, target_recall)
        param_name = _PARAM_NAMES.get(best_index, "param")

        return {
            "index":         best_index,
            "param":         param,
            "param_name":    param_name,
            "predicted_qps": predicted_qps,
        }

    def predict_ranking(
        self,
        meta_features: dict,
        target_recall: float,
        candidate_indexes: list[str] | None = None,
        data_sample: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Return a ranked DataFrame of all candidate indexes with predicted QPS.

        Columns: index, predicted_qps, param, param_name
        Sorted by predicted_qps descending.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_ranking().")

        meta_features = dict(meta_features)
        _fill_missing_meta(meta_features, data_sample)

        if candidate_indexes is None:
            candidate_indexes = self._candidate_indexes

        rows = []
        for idx in candidate_indexes:
            row = {**meta_features, "target_recall": target_recall, "index": idx}
            rows.append(row)
        df_in = pd.DataFrame(rows)
        qps = self._selector.predict_qps(df_in)

        results = []
        for idx, q in zip(candidate_indexes, qps):
            param = self._recommender.predict_param(idx, meta_features, target_recall)
            results.append({
                "index":         idx,
                "predicted_qps": float(q),
                "param":         param,
                "param_name":    _PARAM_NAMES.get(idx, "param"),
            })
        return (
            pd.DataFrame(results)
            .sort_values("predicted_qps", ascending=False)
            .reset_index(drop=True)
        )

    @property
    def feature_importances(self) -> pd.Series:
        """SHAP-free feature importances from the index selector model."""
        return self._selector.feature_importances()

    @property
    def candidate_indexes(self) -> list[str]:
        return list(self._candidate_indexes)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "ANNRouter":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, expected ANNRouter")
        return obj


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _fill_missing_meta(meta: dict, data_sample) -> None:
    """Compute missing meta-features from a raw data sample if provided."""
    required = ["lid_mle", "pdist_mean", "pdist_std", "kmeans_inertia_ratio_16"]
    missing = [k for k in required if k not in meta or meta[k] is None]
    if not missing:
        return
    if data_sample is None:
        raise ValueError(
            f"meta_features is missing {missing} and no data_sample was provided. "
            "Either supply pre-computed features or pass a data_sample array."
        )
    from .features import compute_from_sample
    computed = compute_from_sample(data_sample)
    for k in missing:
        meta[k] = computed[k]
