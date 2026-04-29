"""
Leave-One-Dataset-Out (LODO) cross-validation for the ANN router.

Metrics per fold (held-out dataset):
  precision_at_1   : 1 if router selects the correct best index, else 0
  regret           : (QPS_oracle - QPS_router) / QPS_oracle
  param_error      : |recall_achieved(recommended_param) - target_recall|
                     (approximated from the raw recall-QPS curve)
  spearman_rho     : rank correlation between oracle and router index rankings

Also compares against a depth-3 DecisionTree classifier (sanity baseline).
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data_loader import load_interpolated, load_raw_curves, RECALL_TARGETS
from .features import engineer_features, selector_feature_cols
from .models import CatBoostIndexSelector, ParameterRecommender


# -----------------------------------------------------------------------
# Oracle helpers
# -----------------------------------------------------------------------

def _oracle_best(df_fold: pd.DataFrame, target_recall: float) -> tuple[str, float]:
    """Return (best_index, oracle_qps) for a single held-out dataset at target_recall."""
    sub = df_fold[np.isclose(df_fold["target_recall"], target_recall)]
    if sub.empty:
        return None, 0.0
    best_row = sub.loc[sub["qps_at_target_recall"].idxmax()]
    return str(best_row["index"]), float(best_row["qps_at_target_recall"])


def _param_recall_error(
    raw_curves: pd.DataFrame,
    dataset: str,
    index: str,
    param: float,
    target_recall: float,
    k: int = 20,
) -> float:
    """
    Estimate |actual_recall(param) - target_recall| by interpolating the raw curve.
    Returns NaN if the curve is unavailable.
    """
    curve = raw_curves[
        (raw_curves["dataset"] == dataset)
        & (raw_curves["index"]   == index)
        & (raw_curves["k"]       == k)
    ]
    if curve.empty or param is None or np.isnan(param):
        return float("nan")

    curve = curve.sort_values("param")
    actual_recall = float(np.interp(param, curve["param"].values, curve["recall"].values))
    return abs(actual_recall - target_recall)


# -----------------------------------------------------------------------
# Decision-tree baseline
# -----------------------------------------------------------------------

def _tree_select_index(
    train_df: pd.DataFrame,
    meta_features: dict,
    target_recall: float,
    candidate_indexes: list[str],
) -> str:
    """Depth-3 DecisionTreeClassifier sanity baseline."""
    from sklearn.tree import DecisionTreeClassifier

    feat_cols = selector_feature_cols()
    tr_eng = engineer_features(train_df)

    # Build label = best index per (dataset, target_recall)
    # Note: feat_cols already contains target_recall, so don't prepend it again
    best = (
        tr_eng.loc[tr_eng.groupby(["dataset", "target_recall"])["qps_at_target_recall"].idxmax()]
        [["dataset"] + feat_cols + ["index"]]
        .copy()
    )
    sub = best[np.isclose(best["target_recall"], target_recall)]
    if sub.empty:
        return candidate_indexes[0]

    X_train = sub[feat_cols].values
    y_train = sub["index"].values
    if len(np.unique(y_train)) < 2:
        return str(y_train[0])

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)

    meta_row = {**meta_features, "target_recall": target_recall}
    meta_df = engineer_features(pd.DataFrame([meta_row]))
    X_test = meta_df[feat_cols].values
    return str(clf.predict(X_test)[0])


# -----------------------------------------------------------------------
# LODO cross-validation
# -----------------------------------------------------------------------

def lodo_cv(
    results_dir: str,
    recall_targets: list[float] | None = None,
    k: int = 20,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run Leave-One-Dataset-Out cross-validation and return a DataFrame of metrics.

    Returns a DataFrame with columns:
        dataset, target_recall,
        catboost_selected, oracle_selected,
        precision_at_1, regret,
        param_error, spearman_rho,
        tree_selected, tree_precision_at_1, tree_regret
    """
    if recall_targets is None:
        recall_targets = RECALL_TARGETS

    df_all  = load_interpolated(results_dir, recall_targets=recall_targets, k=k)
    raw_all = load_raw_curves(results_dir)

    datasets = sorted(df_all["dataset"].unique().tolist())
    records  = []

    for held_out in datasets:
        train_df = df_all[df_all["dataset"] != held_out].copy()
        test_df  = df_all[df_all["dataset"] == held_out].copy()

        if train_df.empty or test_df.empty:
            continue

        # Train CatBoost sub-models on all other datasets
        selector   = CatBoostIndexSelector()
        recommender = ParameterRecommender()
        selector.fit(train_df)
        recommender.fit(train_df)

        candidate_indexes = sorted(df_all["index"].unique().tolist())
        meta_row = test_df.iloc[0]
        meta_feats = {
            "n":                         meta_row["n"],
            "d":                         meta_row["d"],
            "lid_mle":                   meta_row["lid_mle"],
            "pdist_mean":                meta_row["pdist_mean"],
            "pdist_std":                 meta_row["pdist_std"],
            "kmeans_inertia_ratio_16":   meta_row["kmeans_inertia_ratio_16"],
        }

        for tr in recall_targets:
            oracle_index, oracle_qps = _oracle_best(test_df, tr)
            if oracle_index is None:
                continue

            # CatBoost prediction
            cb_index, _ = selector.select_index(meta_feats, tr, candidate_indexes)
            cb_param = recommender.predict_param(cb_index, meta_feats, tr)

            # Actual (ground-truth) QPS of the router-selected index on the held-out dataset
            cb_actual_row = test_df[
                (np.isclose(test_df["target_recall"], tr)) & (test_df["index"] == cb_index)
            ]
            cb_actual_qps = float(cb_actual_row["qps_at_target_recall"].iloc[0]) \
                if not cb_actual_row.empty else 0.0

            precision = int(cb_index == oracle_index)
            regret    = (oracle_qps - cb_actual_qps) / (oracle_qps + 1e-10) if oracle_qps > 0 else 0.0
            regret    = max(0.0, float(regret))

            param_err = _param_recall_error(raw_all, held_out, cb_index, cb_param, tr, k)

            # Spearman rank correlation of full index ranking
            oracle_ranking = (
                test_df[np.isclose(test_df["target_recall"], tr)]
                .sort_values("qps_at_target_recall", ascending=False)["index"]
                .values
            )
            # router predicted scores for all indexes
            cb_rows = [
                {**meta_feats, "target_recall": tr, "index": idx}
                for idx in candidate_indexes
            ]
            cb_scores = selector.predict_qps(
                engineer_features(pd.DataFrame(cb_rows))
            )
            cb_ranking = [
                candidate_indexes[i] for i in np.argsort(cb_scores)[::-1]
            ]

            common = [idx for idx in oracle_ranking if idx in cb_ranking]
            if len(common) >= 2:
                o_ranks = [list(oracle_ranking).index(i) for i in common]
                c_ranks = [cb_ranking.index(i) for i in common]
                rho, _ = spearmanr(o_ranks, c_ranks)
            else:
                rho = float("nan")

            # Decision-tree baseline
            tree_index = _tree_select_index(train_df, meta_feats, tr, candidate_indexes)
            tree_row = test_df[
                (np.isclose(test_df["target_recall"], tr)) &
                (test_df["index"] == tree_index)
            ]
            tree_qps = float(tree_row["qps_at_target_recall"].iloc[0]) if not tree_row.empty else 0.0
            tree_precision = int(tree_index == oracle_index)
            tree_regret    = max(0.0, (oracle_qps - tree_qps) / (oracle_qps + 1e-10))

            records.append({
                "dataset":             held_out,
                "target_recall":       tr,
                "catboost_selected":   cb_index,
                "oracle_selected":     oracle_index,
                "precision_at_1":      precision,
                "regret":              regret,
                "param_error":         param_err,
                "spearman_rho":        rho,
                "tree_selected":       tree_index,
                "tree_precision_at_1": tree_precision,
                "tree_regret":         tree_regret,
            })

        if verbose:
            print(f"  [{held_out}] done")

    results = pd.DataFrame(records)

    if verbose:
        _print_summary(results)

    return results


def _print_summary(results: pd.DataFrame) -> None:
    print("\n=== LODO CV Summary ===")
    print(f"Datasets evaluated : {results['dataset'].nunique()}")
    print(f"Recall targets     : {sorted(results['target_recall'].unique())}")
    print()
    agg = results.groupby("target_recall").agg(
        cb_p1    =("precision_at_1",      "mean"),
        cb_regret=("regret",              "mean"),
        cb_rho   =("spearman_rho",        "mean"),
        tree_p1  =("tree_precision_at_1", "mean"),
        tree_reg =("tree_regret",         "mean"),
    ).round(3)
    print(agg.to_string())
    print()
    print("Overall CatBoost  — Precision@1: {:.3f}  Regret: {:.3f}  Spearman ρ: {:.3f}".format(
        results["precision_at_1"].mean(),
        results["regret"].mean(),
        results["spearman_rho"].mean(skipna=True),
    ))
    print("Overall Tree base — Precision@1: {:.3f}  Regret: {:.3f}".format(
        results["tree_precision_at_1"].mean(),
        results["tree_regret"].mean(),
    ))
