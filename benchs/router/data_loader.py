"""
Parse benchs/results_router/*.json into flat DataFrames for router training.

Two outputs:
  load_raw_curves()     -> one row per (dataset, index, k, param) with recall+qps
  load_interpolated()   -> one row per (dataset, index, k, target_recall)
                          with qps_at_target_recall and param_at_target_recall
"""

import json
import os
import numpy as np
import pandas as pd

# Recall thresholds used as targets during training and evaluation
RECALL_TARGETS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]

# Recall@k values stored in the JSON files
RECALL_KS = [1, 10, 20, 50, 100]

_PARAM_NAMES = {
    "SuCo":   "candidate_ratio",
    "HNSW32": "efSearch",
    "HNSW48": "efSearch",
    "CSPG":   "efSearch",
    "SHG":    "efSearch",
}


def _param_name(index: str) -> str:
    return _PARAM_NAMES.get(index, "param")


def _interpolate_qps_and_param(recalls, qps_vals, params, target_recall):
    """
    Linear interpolation of QPS and parameter at a target recall.

    recall-QPS curves are monotonically increasing in recall as param increases
    (more search effort). We interpolate in recall space.

    Returns (qps, param) at target_recall, or (0.0, None) if unreachable.
    """
    recalls = np.asarray(recalls, dtype=float)
    qps_vals = np.asarray(qps_vals, dtype=float)
    params = np.asarray(params, dtype=float)

    # Sort by recall ascending (should already be, but be safe)
    order = np.argsort(recalls)
    recalls, qps_vals, params = recalls[order], qps_vals[order], params[order]

    if target_recall < recalls[0]:
        # Even the cheapest setting exceeds the target: pick cheapest
        return float(qps_vals[0]), float(params[0])
    if target_recall > recalls[-1]:
        # Index cannot reach this recall
        return 0.0, None

    qps_interp = float(np.interp(target_recall, recalls, qps_vals))
    param_interp = float(np.interp(target_recall, recalls, params))
    return qps_interp, param_interp


def load_raw_curves(results_dir: str) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per (dataset, index, k, param_config).

    Columns:
        dataset, n, d, lid_mle, pdist_mean, pdist_std, kmeans_inertia_ratio_16,
        build_time_s, size_mb, memory_mb,
        index, param, param_name, k, recall, qps, ms_per_query
    """
    rows = []
    for fname in sorted(os.listdir(results_dir)):
        if not (fname.startswith("results_") and fname.endswith(".json")):
            continue
        path = os.path.join(results_dir, fname)
        with open(path) as f:
            data = json.load(f)

        dataset = data["dataset"]
        feats = data.get("features", {})
        construction = data.get("construction", {})

        meta = {
            "dataset":                   dataset,
            "n":                         feats.get("n",    data.get("n")),
            "d":                         feats.get("d",    data.get("d")),
            "lid_mle":                   feats.get("lid_mle"),
            "pdist_mean":                feats.get("pdist_mean"),
            "pdist_std":                 feats.get("pdist_std"),
            "kmeans_inertia_ratio_16":   feats.get("kmeans_inertia_ratio_16"),
        }

        for k in RECALL_KS:
            key = f"recall_k{k}"
            if key not in data:
                continue
            for index, configs in data[key].items():
                constr = construction.get(index, {})
                build_time = constr.get("build_time_s", np.nan)
                size_mb    = constr.get("size_mb",      np.nan)
                memory_mb  = constr.get("memory_mb",    np.nan)
                # -1 sentinel → NaN
                if build_time == -1: build_time = np.nan
                if size_mb    == -1: size_mb    = np.nan
                if memory_mb  == -1: memory_mb  = np.nan

                pname = _param_name(index)
                for cfg in configs:
                    rows.append({
                        **meta,
                        "build_time_s": build_time,
                        "size_mb":      size_mb,
                        "memory_mb":    memory_mb,
                        "index":        index,
                        "param":        cfg["param"],
                        "param_name":   pname,
                        "k":            k,
                        "recall":       cfg["recall"],
                        "qps":          cfg["qps"],
                        "ms_per_query": cfg["ms_per_query"],
                    })

    return pd.DataFrame(rows)


def load_interpolated(
    results_dir: str,
    recall_targets: list[float] | None = None,
    k: int = 20,
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per (dataset, index, target_recall).

    Columns:
        dataset, n, d, lid_mle, pdist_mean, pdist_std, kmeans_inertia_ratio_16,
        index, param_name, target_recall, qps_at_target_recall, param_at_target_recall

    Rows where the index cannot reach target_recall have qps_at_target_recall=0
    and param_at_target_recall=NaN.
    """
    if recall_targets is None:
        recall_targets = RECALL_TARGETS

    raw = load_raw_curves(results_dir)
    raw_k = raw[raw["k"] == k].copy()

    rows = []
    meta_cols = ["dataset", "n", "d", "lid_mle", "pdist_mean", "pdist_std",
                 "kmeans_inertia_ratio_16"]

    for (dataset, index), grp in raw_k.groupby(["dataset", "index"]):
        meta = {c: grp[c].iloc[0] for c in meta_cols}
        recalls  = grp["recall"].values
        qps_vals = grp["qps"].values
        params   = grp["param"].values
        pname    = grp["param_name"].iloc[0]

        for tr in recall_targets:
            qps_interp, param_interp = _interpolate_qps_and_param(
                recalls, qps_vals, params, tr
            )
            rows.append({
                **meta,
                "index":                index,
                "param_name":           pname,
                "target_recall":        tr,
                "qps_at_target_recall": qps_interp,
                "param_at_target_recall": param_interp,
            })

    df = pd.DataFrame(rows)
    df["param_at_target_recall"] = df["param_at_target_recall"].astype(float)
    return df
