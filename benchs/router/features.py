"""
Meta-feature engineering for the ANN router.

Raw features from bench_router_paper.py:
    n, d, lid_mle, pdist_mean, pdist_std, kmeans_inertia_ratio_16

Derived features added here:
    log_n       log(n) — log scale for tree splits on cardinality
    d_over_lid  d / lid_mle — how many dimensions are "wasted" vs. intrinsic dim
    pdist_cv    pdist_std / pdist_mean — normalised spread of distances

These are the features fed to Model A (index selector) and Model B (param recommender).
Construction features (build_time_s, size_mb) are NOT included — they are unknowns
for unseen datasets.
"""

import numpy as np
import pandas as pd

RAW_META = [
    "n", "d", "lid_mle", "pdist_mean", "pdist_std", "kmeans_inertia_ratio_16",
]

ENGINEERED_META = [
    "log_n", "d_over_lid", "pdist_cv",
]

ALL_META = RAW_META + ENGINEERED_META


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived meta-features to a DataFrame that already contains RAW_META columns.
    Returns a copy with additional columns; existing columns are unchanged.
    """
    df = df.copy()
    df["log_n"]      = np.log1p(df["n"].astype(float))
    df["d_over_lid"] = df["d"].astype(float) / df["lid_mle"].astype(float).clip(lower=1e-6)
    df["pdist_cv"]   = (
        df["pdist_std"].astype(float) / df["pdist_mean"].astype(float).clip(lower=1e-6)
    )
    return df


def selector_feature_cols() -> list[str]:
    """Feature columns for Model A (index selector)."""
    return [
        "log_n", "d_over_lid", "pdist_cv",   # derived
        "d", "lid_mle", "pdist_mean",          # raw
        "kmeans_inertia_ratio_16",
        "target_recall",
    ]


def recommender_feature_cols() -> list[str]:
    """Feature columns for Model B (parameter recommender)."""
    return [
        "log_n", "d_over_lid", "pdist_cv",
        "d", "lid_mle", "pdist_mean",
        "kmeans_inertia_ratio_16",
        "target_recall",
    ]


def compute_from_sample(
    X: np.ndarray,
    sample_size: int = 10_000,
    k_lid: int = 20,
    n_pairs: int = 10_000,
    n_clusters: int = 16,
    rng: int = 42,
) -> dict:
    """
    Compute meta-features from a raw data matrix X (shape [n, d]).

    Used at inference time when the caller provides a data sample instead of
    pre-computed features.

    Returns a dict with keys matching RAW_META (excluding n, d which come from X.shape).
    """
    import faiss

    rng = np.random.default_rng(rng)
    n, d = X.shape

    # Sample for LID and pairwise distances
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    sample = X[idx].astype(np.float32)

    # LID via MLE using k-NN distances
    index_flat = faiss.IndexFlatL2(d)
    index_flat.add(sample)
    D2, _ = index_flat.search(sample, k_lid + 1)
    D2 = D2[:, 1:]  # exclude self
    D = np.sqrt(np.maximum(D2, 0))
    D_k = D[:, -1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        lid_vals = -1.0 / np.mean(np.log(D / (D_k + 1e-10) + 1e-10), axis=1)
    lid_mle = float(np.median(lid_vals[np.isfinite(lid_vals)]))

    # Pairwise distances on random pairs
    i1 = rng.choice(len(sample), size=n_pairs, replace=True)
    i2 = rng.choice(len(sample), size=n_pairs, replace=True)
    diff = sample[i1].astype(float) - sample[i2].astype(float)
    pdists = np.sqrt((diff ** 2).sum(axis=1))
    pdist_mean = float(pdists.mean())
    pdist_std  = float(pdists.std())

    # K-means clusterability (inertia ratio vs single cluster)
    kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=False)
    kmeans.train(sample)
    _, assignments = kmeans.index.search(sample, 1)
    centroids = kmeans.centroids
    assigned_centroids = centroids[assignments.ravel()]
    inertia_k = float(((sample - assigned_centroids) ** 2).sum())
    mean_vec = sample.mean(axis=0, keepdims=True)
    inertia_1 = float(((sample - mean_vec) ** 2).sum())
    kmeans_ratio = inertia_k / (inertia_1 + 1e-10)

    return {
        "lid_mle":                 lid_mle,
        "pdist_mean":              pdist_mean,
        "pdist_std":               pdist_std,
        "kmeans_inertia_ratio_16": kmeans_ratio,
    }
