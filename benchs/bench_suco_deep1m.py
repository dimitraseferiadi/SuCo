#!/usr/bin/env python3
"""
benchs/bench_suco_deep1m.py

Benchmark and end-to-end accuracy test for faiss.IndexSuCo on the Deep1M
dataset (1-million-vector subset of Deep10B, d=96).

Usage
-----
# Basic run – single configuration, full output:
    python benchs/bench_suco_deep1m.py

# Override the dataset base directory (if data is not in ./data/):
    python benchs/bench_suco_deep1m.py --data-dir /path/to/datasets/

# Run a parameter sweep (nsubspaces, collision_ratio, candidate_ratio):
    python benchs/bench_suco_deep1m.py --sweep

# Save/load a pre-built index to speed up repeated query benchmarks:
    python benchs/bench_suco_deep1m.py --index-path /tmp/suco_deep1m.idx
    python benchs/bench_suco_deep1m.py --index-path /tmp/suco_deep1m.idx  # reuses saved

Dataset
-------
Deep1M is part of the Deep1B collection.  Download instructions:
  https://github.com/facebookresearch/faiss/tree/main/benchs#getting-deep1b

Expected files (relative to --data-dir, default ./data/):
  deep1b/base.fvecs              – 1B base vectors (only the first 1M are used)
  deep1b/deep1B_queries.fvecs    – 10 000 query vectors
  deep1b/deep1M_groundtruth.ivecs – ground-truth top-100 for the 1M subset
  deep1b/learn.fvecs             – 358M training vectors (first 500K used)
"""

import argparse
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# FAISS import
# ---------------------------------------------------------------------------
try:
    import faiss
    from faiss.contrib.datasets import DatasetDeep1B, set_dataset_basedir
except ImportError as e:
    sys.exit(f"Cannot import faiss: {e}\n"
             "Build FAISS with IndexSuCo and run from the repo root.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def recall_at_k(I: np.ndarray, gt: np.ndarray, k: int) -> float:
    """
    Fraction of queries for which the true nearest neighbour is among the
    top-k returned results.  gt[:, 0] is the true NN for each query.
    """
    nq = I.shape[0]
    hits = (I[:, :k] == gt[:, :1]).any(axis=1).sum()
    return float(hits) / nq


def recall_at_k_topR(I: np.ndarray, gt: np.ndarray, k: int, r: int) -> float:
    """Fraction of the true top-r neighbours found in the top-k results."""
    nq = I.shape[0]
    hits = sum(
        len(set(I[i, :k]) & set(gt[i, :r]))
        for i in range(nq)
    )
    return hits / (nq * r)


def approx_ratio(
    xb: np.ndarray,
    xq: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    gt: np.ndarray,
    k: int,
) -> float:
    """
    Mean distance approximation ratio (paper metric):
        mean_{q, j in [k]}  L2(q, approx_j) / L2(q, true_j)
    D[q, j] is the squared L2 returned by FAISS search.
    """
    k_use = min(k, D.shape[1], gt.shape[1])
    nq    = len(xq)
    approx_l2 = np.sqrt(np.maximum(D[:, :k_use], 0.0)).astype(np.float64)
    true_l2   = np.zeros((nq, k_use), dtype=np.float64)
    for j in range(k_use):
        diff = xq.astype(np.float64) - xb[gt[:, j]].astype(np.float64)
        true_l2[:, j] = np.sqrt((diff * diff).sum(axis=1))
    mask   = true_l2 > 0
    ratios = np.where(mask, approx_l2 / np.where(mask, true_l2, 1.0), 1.0)
    return float(ratios.mean())


def fmt_time(seconds: float) -> str:
    if seconds >= 60:
        return f"{seconds/60:.1f}min"
    return f"{seconds:.2f}s"


def print_header(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


# ---------------------------------------------------------------------------
# Build + search  (core routine, reused by sweep)
# ---------------------------------------------------------------------------

def run_benchmark(
    xb: np.ndarray,
    xq: np.ndarray,
    xt: np.ndarray,
    gt: np.ndarray,
    *,
    nsubspaces: int      = 8,
    ncentroids_half: int = 50,
    collision_ratio: float = 0.05,
    candidate_ratio: float = 0.005,
    niter: int           = 10,
    k: int               = 10,
    index_path: str      = "",
    verbose: bool        = True,
) -> dict:
    """
    Build IndexSuCo (or load from index_path if it exists), search, return
    a results dict with timing and recall metrics.
    """
    nb, d = xb.shape
    nq    = xq.shape[0]

    # ------------------------------------------------------------------
    # Build / load
    # ------------------------------------------------------------------
    index = faiss.IndexSuCo(
        d,
        nsubspaces,
        ncentroids_half,
        collision_ratio,
        candidate_ratio,
        niter,
    )
    index.verbose = verbose

    if index_path and os.path.exists(index_path):
        if verbose:
            print(f"  Loading index from {index_path} …")
        t0 = time.perf_counter()
        index.read_index(index_path)
        t_load = time.perf_counter() - t0
        t_train = t_add = 0.0
        if verbose:
            print(f"  Loaded in {fmt_time(t_load)}")
    else:
        # Train
        if verbose:
            print(f"  Training on {len(xt):,} vectors …")
        t0 = time.perf_counter()
        index.train(xt)
        t_train = time.perf_counter() - t0
        if verbose:
            print(f"  Training done in {fmt_time(t_train)}")

        # Add
        if verbose:
            print(f"  Adding {nb:,} vectors …")
        t0 = time.perf_counter()
        index.add(xb)
        t_add = time.perf_counter() - t0
        if verbose:
            print(f"  Add done in {fmt_time(t_add)}")

        if index_path:
            if verbose:
                print(f"  Saving index to {index_path} …")
            index.write_index(index_path)

    # ------------------------------------------------------------------
    # Warm-up  (3 × 5-query batches for stable cache state)
    # ------------------------------------------------------------------
    for _ in range(3):
        index.search(xq[:min(5, nq)], k)

    # ------------------------------------------------------------------
    # Timed search
    # ------------------------------------------------------------------
    if verbose:
        print(f"  Searching {nq:,} queries (k={k}) …")
    t0 = time.perf_counter()
    D, I = index.search(xq, k)
    t_search = time.perf_counter() - t0
    ms_per_q = t_search / nq * 1000.0
    qps      = nq / t_search

    # ------------------------------------------------------------------
    # Recall metrics
    # ------------------------------------------------------------------
    r1     = recall_at_k(I, gt, 1)
    k_r    = min(10, k, gt.shape[1])
    r10    = recall_at_k(I, gt, k_r)
    r10r10 = recall_at_k_topR(I, gt, k_r, k_r)
    ratio  = approx_ratio(xb, xq, D, I, gt, k_r)
    r20  = recall_at_k(I, gt, min(20,  k, gt.shape[1])) if min(k, gt.shape[1]) >= 20  else None
    r30  = recall_at_k(I, gt, min(30,  k, gt.shape[1])) if min(k, gt.shape[1]) >= 30  else None
    r40  = recall_at_k(I, gt, min(40,  k, gt.shape[1])) if min(k, gt.shape[1]) >= 40  else None
    r50  = recall_at_k(I, gt, min(50,  k, gt.shape[1])) if min(k, gt.shape[1]) >= 50  else None
    r100 = recall_at_k(I, gt, min(100, k, gt.shape[1])) if min(k, gt.shape[1]) >= 100 else None

    if verbose:
        print(f"\n  Results:")
        print(f"    ntotal          = {index.ntotal:,}")
        print(f"    OMP threads     = {faiss.omp_get_max_threads()}")
        print(f"    train time      = {fmt_time(t_train)}")
        print(f"    add time        = {fmt_time(t_add)}")
        print(f"    search time     = {t_search*1000:.1f}ms total")
        print(f"    ms / query      = {ms_per_q:.3f}")
        print(f"    QPS             = {qps:.0f}")
        print(f"    Recall@1        = {r1:.4f}")
        print(f"    Recall@10       = {r10:.4f}")
        print(f"    10-Recall@10    = {r10r10:.4f}")
        print(f"    Dist ratio      = {ratio:.4f}")
        if r20  is not None: print(f"    Recall@20       = {r20:.4f}")
        if r30  is not None: print(f"    Recall@30       = {r30:.4f}")
        if r40  is not None: print(f"    Recall@40       = {r40:.4f}")
        if r50  is not None: print(f"    Recall@50       = {r50:.4f}")
        if r100 is not None: print(f"    Recall@100      = {r100:.4f}")

    return dict(
        nsubspaces=nsubspaces,
        ncentroids_half=ncentroids_half,
        collision_ratio=collision_ratio,
        candidate_ratio=candidate_ratio,
        t_train=t_train,
        t_add=t_add,
        t_search_ms=t_search * 1000,
        ms_per_query=ms_per_q,
        qps=qps,
        recall_at_1=r1,
        recall_at_10=r10,
        recall_10r10=r10r10,
        dist_ratio=ratio,
        recall_at_20=r20,
        recall_at_30=r30,
        recall_at_40=r40,
        recall_at_50=r50,
        recall_at_100=r100,
    )


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

SWEEP_CONFIGS = [
    # (nsubspaces, ncentroids_half, collision_ratio, candidate_ratio)
    # ---- varying collision_ratio ---
    (8, 50, 0.02, 0.002),
    (8, 50, 0.05, 0.005),   # paper default
    (8, 50, 0.10, 0.010),
    (8, 50, 0.20, 0.020),
    # ---- varying nsubspaces --------
    (4, 50, 0.05, 0.005),
    (12, 50, 0.05, 0.005),
    # ---- varying ncentroids_half ---
    (8, 25, 0.05, 0.005),
    (8, 100, 0.05, 0.005),
]


def run_sweep(xb, xq, xt, gt, k: int = 10) -> None:
    print_header("Parameter Sweep")
    header = (
        f"{'Ns':>3}  {'hd':>3}  {'nc':>4}  {'alpha':>6}  {'beta':>7}  "
        f"{'ms/q':>7}  {'QPS':>7}  {'R@1':>6}  {'R@10':>6}  "
        f"{'10R@10':>8}  {'ratio':>8}"
    )
    print(header)
    print("-" * len(header))

    d  = xb.shape[1]
    nq = len(xq)
    cur_ns = cur_nc = None
    index  = None

    for ns, nc, alpha, beta in SWEEP_CONFIGS:
        if (ns, nc) != (cur_ns, cur_nc):
            index = faiss.IndexSuCo(d, ns, nc, alpha, beta, 10)
            index.train(xt)
            index.add(xb)
            cur_ns, cur_nc = ns, nc

        index.collision_ratio = alpha
        index.candidate_ratio = beta
        for _ in range(3):  # warm-up (3 batches for stable cache)
            index.search(xq[:min(5, len(xq))], k)
        t0 = time.perf_counter()
        D, I = index.search(xq, k)
        t_s = time.perf_counter() - t0
        k_r    = min(10, k, gt.shape[1])
        r1     = recall_at_k(I, gt, 1)
        r10    = recall_at_k(I, gt, k_r)
        r10r10 = recall_at_k_topR(I, gt, k_r, k_r)
        ratio  = approx_ratio(xb, xq, D, I, gt, k_r)
        print(
            f"{ns:>3}  {d // ns // 2:>3}  {nc:>4}  "
            f"{alpha:>6.3f}  {beta:>7.4f}  "
            f"{t_s/nq*1000:>7.3f}  {nq/t_s:>7.0f}  "
            f"{r1:>6.4f}  {r10:>6.4f}  "
            f"{r10r10:>8.4f}  {ratio:>8.4f}"
        )


# ---------------------------------------------------------------------------
# Comparison vs flat (brute force) baseline
# ---------------------------------------------------------------------------

def compare_vs_flat(xb: np.ndarray, xq: np.ndarray, gt: np.ndarray, k: int = 10) -> None:
    print_header("Baseline: IndexFlatL2")
    flat = faiss.IndexFlatL2(xb.shape[1])
    flat.add(xb)

    t0 = time.perf_counter()
    _, I_flat = flat.search(xq, k)
    t_flat = time.perf_counter() - t0

    r1  = recall_at_k(I_flat, gt, 1)
    r10 = recall_at_k(I_flat, gt, min(10, k))
    print(f"  ms/query  = {t_flat/xq.shape[0]*1000:.3f}")
    print(f"  QPS       = {xq.shape[0]/t_flat:.0f}")
    print(f"  Recall@1  = {r1:.4f}  (upper bound for SuCo)")
    print(f"  Recall@10 = {r10:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark IndexSuCo on the Deep1M dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir", default="data/",
        help="Root directory that contains the deep1b/ subdirectory.",
    )
    p.add_argument(
        "--maxtrain", type=int, default=500_000,
        help="Number of training vectors to use (max 358M).",
    )
    p.add_argument(
        "--k", type=int, default=10,
        help="Number of nearest neighbours to retrieve.",
    )
    p.add_argument(
        "--nsubspaces", type=int, default=8,
        help="Number of subspaces (Ns).  Deep1M has d=96, so nsubspaces must "
             "divide 96 and 96/nsubspaces must be even.",
    )
    p.add_argument(
        "--ncentroids-half", type=int, default=50,
        help="K-means centroids per half-subspace (sqrt(K)).",
    )
    p.add_argument(
        "--collision-ratio", type=float, default=0.05,
        help="alpha: fraction of dataset retrieved per subspace.",
    )
    p.add_argument(
        "--candidate-ratio", type=float, default=0.005,
        help="beta: fraction of dataset in the re-rank pool.",
    )
    p.add_argument(
        "--niter", type=int, default=10,
        help="K-means iterations during training.",
    )
    p.add_argument(
        "--index-path", default="",
        help="If given, save the index here after building (or load it if it "
             "already exists, skipping build).",
    )
    p.add_argument(
        "--sweep", action="store_true",
        help="Run a parameter sweep over several (Ns, nc, alpha, beta) configs.",
    )
    p.add_argument(
        "--flat-baseline", action="store_true",
        help="Also run a brute-force flat index as an upper-bound baseline.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print_header("Loading Deep1M dataset")
    set_dataset_basedir(args.data_dir)
    ds = DatasetDeep1B(nb=10**6)

    print(f"  data_dir  : {args.data_dir}")
    print(f"  d         : {ds.d}")
    print(f"  nb (base) : {ds.nb:,}")
    print(f"  nq        : {ds.nq:,}")
    print(f"  maxtrain  : {args.maxtrain:,}")
    print(f"  OMP threads: {faiss.omp_get_max_threads()}")

    print("  Loading queries …", end=" ", flush=True)
    xq = ds.get_queries()                              # (10000, 96)
    print(f"shape={xq.shape}")

    print("  Loading base …", end=" ", flush=True)
    xb = ds.get_database()                             # (1000000, 96)
    print(f"shape={xb.shape}")

    print("  Loading train …", end=" ", flush=True)
    xt = ds.get_train(maxtrain=args.maxtrain)          # (maxtrain, 96)
    print(f"shape={xt.shape}")

    print("  Loading ground truth …", end=" ", flush=True)
    gt = ds.get_groundtruth(k=100)                     # (10000, 100)
    print(f"shape={gt.shape}")

    # ------------------------------------------------------------------
    # Optional flat baseline
    # ------------------------------------------------------------------
    if args.flat_baseline:
        compare_vs_flat(xb, xq, gt, k=args.k)

    # ------------------------------------------------------------------
    # Main benchmark
    # ------------------------------------------------------------------
    print_header("IndexSuCo benchmark  (Deep1M, d=96)")
    print(f"  nsubspaces      = {args.nsubspaces}")
    print(f"  ncentroids_half = {args.ncentroids_half}")
    print(f"  collision_ratio = {args.collision_ratio}")
    print(f"  candidate_ratio = {args.candidate_ratio}")
    print(f"  niter           = {args.niter}")
    print(f"  k               = {args.k}")

    run_benchmark(
        xb, xq, xt, gt,
        nsubspaces=args.nsubspaces,
        ncentroids_half=args.ncentroids_half,
        collision_ratio=args.collision_ratio,
        candidate_ratio=args.candidate_ratio,
        niter=args.niter,
        k=args.k,
        index_path=args.index_path,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Optional sweep
    # ------------------------------------------------------------------
    if args.sweep:
        run_sweep(xb, xq, xt, gt, k=args.k)


if __name__ == "__main__":
    main()
