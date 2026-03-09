#!/usr/bin/env python3
"""
benchs/bench_suco_deep1m.py

Benchmark and end-to-end accuracy test for faiss.IndexSuCo on the Deep1M,
Deep10M, and Deep100M subsets of Deep1B (d=96).

Usage
-----
# Deep1M (default):
    python benchs/bench_suco_deep1m.py --data-dir /path/to/data/

# Deep10M:
    python benchs/bench_suco_deep1m.py --data-dir /path/to/data/ --nb 10000000

# Deep100M:
    python benchs/bench_suco_deep1m.py --data-dir /path/to/data/ --nb 100000000

# Parameter sweep:
    python benchs/bench_suco_deep1m.py --nb 10000000 --sweep

# Save/load a pre-built index to speed up repeated query benchmarks:
    python benchs/bench_suco_deep1m.py --nb 1000000 --index-path /tmp/suco.idx

Dataset preparation
-------------------
Run prepare_deep1m.py once to build base.fvecs / learn.fvecs:
  # Deep1M / Deep10M  (base00 + learn00 already sufficient):
    python benchs/prepare_deep1m.py --data-dir /home/dhm/data/ --nb 10000000

  # Deep100M  (download base01-03 first, then prepare):
    python benchs/downloadDeep1B.py --data-dir /home/dhm/data/ --base-chunks 1 2 3
    python benchs/prepare_deep1m.py --data-dir /home/dhm/data/ --nb 100000000 --nt 5000000

Expected files (relative to --data-dir):
  deep1b/base.fvecs                  – prepared base vectors (≥ --nb vectors)
  deep1b/learn.fvecs                 – prepared training vectors
  deep1b/deep1B_queries.fvecs        – 10 000 query vectors
  deep1b/deep1M_groundtruth.ivecs    – ground-truth for 1M subset
  deep1b/deep10M_groundtruth.ivecs   – ground-truth for 10M subset
  deep1b/deep100M_groundtruth.ivecs  – ground-truth for 100M subset
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
    from faiss.contrib.vecs_io import fvecs_mmap
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


def _index_size_mb(index) -> float:
    """Return serialized size of the index in MiB (disk / RAM footprint proxy)."""
    try:
        buf = faiss.serialize_index(index)
        return buf.nbytes / (1024.0 * 1024.0)
    except Exception:
        # Fallback for wrappers (e.g. IndexSuCo) that expose write_index(path).
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".idx", delete=False) as f:
            tmppath = f.name
        try:
            index.write_index(tmppath)
            return os.path.getsize(tmppath) / (1024.0 * 1024.0)
        finally:
            try:
                os.unlink(tmppath)
            except OSError:
                pass


def print_header(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


# ---------------------------------------------------------------------------
# Batched add helper  (avoids allocating the full base matrix in RAM)
# ---------------------------------------------------------------------------

# Number of vectors added to the index in one batch.
# 100 K × 96 × 4 bytes ≈ 37 MB – small enough for any machine.
ADD_BATCH_SIZE = 100_000


def add_batched(index, xb, verbose: bool = False) -> float:
    """
    Add all vectors in `xb` to `index` in chunks of ADD_BATCH_SIZE.
    `xb` can be any object that supports len() and integer slicing
    (np.ndarray, np.memmap, etc.).
    Returns the total wall-clock time in seconds.
    """
    nb  = len(xb)
    t0  = time.perf_counter()
    for i0 in range(0, nb, ADD_BATCH_SIZE):
        batch = np.ascontiguousarray(xb[i0:i0 + ADD_BATCH_SIZE], dtype='float32')
        index.add(batch)
        if verbose:
            done = min(i0 + ADD_BATCH_SIZE, nb)
            print(f"\r    {done:>12,} / {nb:,}  ({done/nb*100:.1f}%)",
                  end="", flush=True)
    if verbose:
        print()   # newline after progress
    return time.perf_counter() - t0


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
        t_add = add_batched(index, xb, verbose=verbose)
        if verbose:
            print(f"  Add done in {fmt_time(t_add)}")

        if index_path:
            if verbose:
                print(f"  Saving index to {index_path} …")
            index.write_index(index_path)

    size_mb = _index_size_mb(index)

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
        print(f"    build time      = {fmt_time(t_train + t_add)}")
        print(f"    index size      = {size_mb:.1f} MiB")
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
        t_build=t_train + t_add,
        index_size_mb=size_mb,
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
            add_batched(index, xb)
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
# HNSW benchmark
# ---------------------------------------------------------------------------

# efSearch values used for the sweep (accuracy vs speed trade-off curve)
HNSW_EF_SWEEP = [8, 16, 32, 64, 128, 256, 512]


def run_hnsw_benchmark(
    xb: np.ndarray,
    xq: np.ndarray,
    gt: np.ndarray,
    *,
    M: int              = 32,
    ef_construction: int = 200,
    ef_search: int       = 128,
    k: int               = 10,
    index_path: str      = "",
    verbose: bool        = True,
) -> dict:
    """
    Build IndexHNSWFlat (or load from index_path) on xb, then search xq.
    Returns a results dict with timing and recall metrics.
    """
    nb, d = xb.shape
    nq    = xq.shape[0]

    # ------------------------------------------------------------------
    # Build / load
    # ------------------------------------------------------------------
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction

    hnsw_path = (index_path.replace(".idx", "") + f"_hnsw_M{M}_efc{ef_construction}.idx"
                 if index_path else "")

    if hnsw_path and os.path.exists(hnsw_path):
        if verbose:
            print(f"  Loading HNSW index from {hnsw_path} …")
        t0 = time.perf_counter()
        index = faiss.read_index(hnsw_path)
        t_load = time.perf_counter() - t0
        t_add  = 0.0
        if verbose:
            print(f"  Loaded in {fmt_time(t_load)}")
    else:
        if verbose:
            print(f"  Building HNSW (M={M}, efConstruction={ef_construction}) "
                  f"on {nb:,} vectors …")
        t_add = add_batched(index, xb, verbose=verbose)
        if verbose:
            print(f"  Build done in {fmt_time(t_add)}")
        if hnsw_path:
            if verbose:
                print(f"  Saving HNSW index to {hnsw_path} …")
            faiss.write_index(index, hnsw_path)

    # ------------------------------------------------------------------
    # Set efSearch and search
    # ------------------------------------------------------------------
    size_mb = _index_size_mb(index)
    index.hnsw.efSearch = ef_search

    # Warm-up
    index.search(xq[:1], k)

    if verbose:
        print(f"  Searching {nq:,} queries (k={k}, efSearch={ef_search}) …")
    t0 = time.perf_counter()
    D, I = index.search(xq, k)
    t_search = time.perf_counter() - t0
    ms_per_q = t_search / nq * 1000.0
    qps      = nq / t_search

    r1  = recall_at_k(I, gt, 1)
    r10 = recall_at_k(I, gt, min(10, k))

    if verbose:
        print(f"\n  Results:")
        print(f"    ntotal          = {index.ntotal:,}")
        print(f"    M               = {M}")
        print(f"    efConstruction  = {ef_construction}")
        print(f"    efSearch        = {ef_search}")
        print(f"    build time      = {fmt_time(t_add)}")
        print(f"    index size      = {size_mb:.1f} MiB")
        print(f"    search time     = {t_search*1000:.1f}ms total")
        print(f"    ms / query      = {ms_per_q:.3f}")
        print(f"    QPS             = {qps:.0f}")
        print(f"    Recall@1        = {r1:.4f}")
        print(f"    Recall@10       = {r10:.4f}")

    return dict(
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
        t_build=t_add,
        index_size_mb=size_mb,
        t_search_ms=t_search * 1000,
        ms_per_query=ms_per_q,
        qps=qps,
        recall_at_1=r1,
        recall_at_10=r10,
    )


def run_hnsw_ef_sweep(
    xb: np.ndarray,
    xq: np.ndarray,
    gt: np.ndarray,
    *,
    M: int               = 32,
    ef_construction: int = 200,
    k: int               = 10,
    index_path: str      = "",
) -> None:
    """Build HNSW once, then sweep efSearch to show the recall/speed trade-off."""
    print_header(f"HNSW efSearch sweep  (M={M}, efConstruction={ef_construction})")
    header = (
        f"{'efSearch':>8}  {'ms/q':>7}  {'QPS':>7}  {'R@1':>6}  {'R@10':>6}"
    )
    print(header)
    print("-" * len(header))

    # Build the index once (reuse saved if available)
    nb, d = xb.shape
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction

    hnsw_path = (index_path.replace(".idx", "") + f"_hnsw_M{M}_efc{ef_construction}.idx"
                 if index_path else "")

    if hnsw_path and os.path.exists(hnsw_path):
        print(f"  Loading HNSW index from {hnsw_path} …")
        index = faiss.read_index(hnsw_path)
    else:
        print(f"  Building HNSW (M={M}, efConstruction={ef_construction}) on {nb:,} vectors …")
        t_build = add_batched(index, xb, verbose=True)
        print(f"  Build done in {fmt_time(t_build)}")
        if hnsw_path:
            faiss.write_index(index, hnsw_path)

    nq = xq.shape[0]
    for efs in HNSW_EF_SWEEP:
        index.hnsw.efSearch = efs
        for _ in range(3):  # warm-up (3 batches for stable cache)
            index.search(xq[:min(5, len(xq))], k)
        t0 = time.perf_counter()
        _, I = index.search(xq, k)
        t_s = time.perf_counter() - t0
        r1  = recall_at_k(I, gt, 1)
        r10 = recall_at_k(I, gt, min(10, k))
        print(
            f"{efs:>8}  {t_s/nq*1000:>7.3f}  {nq/t_s:>7.0f}  "
            f"{r1:>6.4f}  {r10:>6.4f}"
        )


# ---------------------------------------------------------------------------
# IVFFlat benchmark
# ---------------------------------------------------------------------------

IVF_NPROBE_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def run_ivfflat_benchmark(
    xb: np.ndarray,
    xq: np.ndarray,
    xt: np.ndarray,
    gt: np.ndarray,
    *,
    nlist: int      = 1024,
    nprobe: int     = 64,
    k: int          = 10,
    index_path: str = "",
    verbose: bool   = True,
) -> dict:
    """Build IndexIVFFlat (or load from index_path), search xq, return metrics dict."""
    nb, d = xb.shape
    nq    = xq.shape[0]

    quantizer = faiss.IndexFlatL2(d)
    index     = faiss.IndexIVFFlat(quantizer, d, nlist)

    ivf_path = (index_path.replace(".idx", "") + f"_ivfflat_nlist{nlist}.idx"
                if index_path else "")

    if ivf_path and os.path.exists(ivf_path):
        if verbose:
            print(f"  Loading IVFFlat index from {ivf_path} …")
        t0 = time.perf_counter()
        index = faiss.read_index(ivf_path)
        t_load = time.perf_counter() - t0
        t_train = t_add = 0.0
        if verbose:
            print(f"  Loaded in {fmt_time(t_load)}")
    else:
        if verbose:
            print(f"  Training IVFFlat (nlist={nlist}) on {len(xt):,} vectors …")
        t0 = time.perf_counter()
        index.train(xt)
        t_train = time.perf_counter() - t0
        if verbose:
            print(f"  Training done in {fmt_time(t_train)}")

        if verbose:
            print(f"  Adding {nb:,} vectors …")
        t_add = add_batched(index, xb, verbose=verbose)
        if verbose:
            print(f"  Add done in {fmt_time(t_add)}")

        if ivf_path:
            if verbose:
                print(f"  Saving IVFFlat index to {ivf_path} …")
            faiss.write_index(index, ivf_path)

    size_mb = _index_size_mb(index)
    index.nprobe = nprobe
    for _ in range(3):  # warm-up (3 batches for stable cache)
            index.search(xq[:min(5, len(xq))], k)

    if verbose:
        print(f"  Searching {nq:,} queries (k={k}, nprobe={nprobe}) …")
    t0 = time.perf_counter()
    D, I = index.search(xq, k)
    t_search = time.perf_counter() - t0
    ms_per_q = t_search / nq * 1000.0
    qps      = nq / t_search

    r1  = recall_at_k(I, gt, 1)
    r10 = recall_at_k(I, gt, min(10, k))

    if verbose:
        print(f"\n  Results:")
        print(f"    ntotal          = {index.ntotal:,}")
        print(f"    nlist           = {nlist}")
        print(f"    nprobe          = {nprobe}")
        print(f"    train time      = {fmt_time(t_train)}")
        print(f"    add time        = {fmt_time(t_add)}")
        print(f"    build time      = {fmt_time(t_train + t_add)}")
        print(f"    index size      = {size_mb:.1f} MiB")
        print(f"    search time     = {t_search*1000:.1f}ms total")
        print(f"    ms / query      = {ms_per_q:.3f}")
        print(f"    QPS             = {qps:.0f}")
        print(f"    Recall@1        = {r1:.4f}")
        print(f"    Recall@10       = {r10:.4f}")

    return dict(
        nlist=nlist, nprobe=nprobe,
        t_train=t_train, t_add=t_add, t_build=t_train + t_add,
        index_size_mb=size_mb, t_search_ms=t_search * 1000,
        ms_per_query=ms_per_q, qps=qps, recall_at_1=r1, recall_at_10=r10,
    )


def run_ivfflat_nprobe_sweep(
    xb: np.ndarray,
    xq: np.ndarray,
    xt: np.ndarray,
    gt: np.ndarray,
    *,
    nlist: int       = 1024,
    k: int           = 10,
    index_path: str  = "",
    dataset_tag: str = "",
) -> None:
    """Build IVFFlat once, sweep nprobe to show the recall/QPS trade-off."""
    tag = f", {dataset_tag}" if dataset_tag else ""
    print_header(f"IVFFlat nprobe sweep  (nlist={nlist}{tag})")
    header = f"{'nprobe':>7}  {'ms/q':>7}  {'QPS':>7}  {'R@1':>6}  {'R@10':>6}"
    print(header)
    print("-" * len(header))

    nb, d = xb.shape
    quantizer = faiss.IndexFlatL2(d)
    index     = faiss.IndexIVFFlat(quantizer, d, nlist)

    ivf_path = (index_path.replace(".idx", "") + f"_ivfflat_nlist{nlist}.idx"
                if index_path else "")

    if ivf_path and os.path.exists(ivf_path):
        print(f"  Loading IVFFlat index from {ivf_path} …")
        index = faiss.read_index(ivf_path)
    else:
        print(f"  Training IVFFlat (nlist={nlist}) on {len(xt):,} vectors …")
        index.train(xt)
        print(f"  Adding {nb:,} vectors …")
        add_batched(index, xb, verbose=True)
        if ivf_path:
            faiss.write_index(index, ivf_path)

    nq = xq.shape[0]
    for nprobe in IVF_NPROBE_SWEEP:
        if nprobe > nlist:
            break
        index.nprobe = nprobe
        for _ in range(3):  # warm-up (3 batches for stable cache)
            index.search(xq[:min(5, len(xq))], k)
        t0 = time.perf_counter()
        _, I = index.search(xq, k)
        t_s = time.perf_counter() - t0
        r1  = recall_at_k(I, gt, 1)
        r10 = recall_at_k(I, gt, min(10, k))
        print(f"{nprobe:>7}  {t_s/nq*1000:>7.3f}  {nq/t_s:>7.0f}  "
              f"{r1:>6.4f}  {r10:>6.4f}")


# ---------------------------------------------------------------------------
# IVFPQ benchmark
# ---------------------------------------------------------------------------

def run_ivfpq_benchmark(
    xb: np.ndarray,
    xq: np.ndarray,
    xt: np.ndarray,
    gt: np.ndarray,
    *,
    nlist: int      = 1024,
    nprobe: int     = 64,
    pq_m: int       = 8,
    pq_nbits: int   = 8,
    k: int          = 10,
    index_path: str = "",
    verbose: bool   = True,
) -> dict:
    """Build IndexIVFPQ (or load from index_path), search xq, return metrics dict."""
    nb, d = xb.shape
    nq    = xq.shape[0]

    if d % pq_m != 0:
        raise ValueError(f"pq_m={pq_m} must divide d={d}")

    quantizer = faiss.IndexFlatL2(d)
    index     = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, pq_nbits)

    ivfpq_path = (index_path.replace(".idx", "") + f"_ivfpq_nlist{nlist}_m{pq_m}_b{pq_nbits}.idx"
                  if index_path else "")

    if ivfpq_path and os.path.exists(ivfpq_path):
        if verbose:
            print(f"  Loading IVFPQ index from {ivfpq_path} …")
        t0 = time.perf_counter()
        index = faiss.read_index(ivfpq_path)
        t_load = time.perf_counter() - t0
        t_train = t_add = 0.0
        if verbose:
            print(f"  Loaded in {fmt_time(t_load)}")
    else:
        if verbose:
            print(f"  Training IVFPQ (nlist={nlist}, m={pq_m}, nbits={pq_nbits}) "
                  f"on {len(xt):,} vectors …")
        t0 = time.perf_counter()
        index.train(xt)
        t_train = time.perf_counter() - t0
        if verbose:
            print(f"  Training done in {fmt_time(t_train)}")

        if verbose:
            print(f"  Adding {nb:,} vectors …")
        t_add = add_batched(index, xb, verbose=verbose)
        if verbose:
            print(f"  Add done in {fmt_time(t_add)}")

        if ivfpq_path:
            if verbose:
                print(f"  Saving IVFPQ index to {ivfpq_path} …")
            faiss.write_index(index, ivfpq_path)

    size_mb = _index_size_mb(index)
    index.nprobe = nprobe
    for _ in range(3):  # warm-up (3 batches for stable cache)
            index.search(xq[:min(5, len(xq))], k)

    if verbose:
        print(f"  Searching {nq:,} queries (k={k}, nprobe={nprobe}) …")
    t0 = time.perf_counter()
    D, I = index.search(xq, k)
    t_search = time.perf_counter() - t0
    ms_per_q = t_search / nq * 1000.0
    qps      = nq / t_search

    r1  = recall_at_k(I, gt, 1)
    r10 = recall_at_k(I, gt, min(10, k))

    if verbose:
        print(f"\n  Results:")
        print(f"    ntotal          = {index.ntotal:,}")
        print(f"    nlist           = {nlist}")
        print(f"    nprobe          = {nprobe}")
        print(f"    pq_m            = {pq_m}  (code_size={pq_m * pq_nbits}b = {pq_m * pq_nbits // 8}B/vec)")
        print(f"    pq_nbits        = {pq_nbits}")
        print(f"    train time      = {fmt_time(t_train)}")
        print(f"    add time        = {fmt_time(t_add)}")
        print(f"    build time      = {fmt_time(t_train + t_add)}")
        print(f"    index size      = {size_mb:.1f} MiB")
        print(f"    search time     = {t_search*1000:.1f}ms total")
        print(f"    ms / query      = {ms_per_q:.3f}")
        print(f"    QPS             = {qps:.0f}")
        print(f"    Recall@1        = {r1:.4f}")
        print(f"    Recall@10       = {r10:.4f}")

    return dict(
        nlist=nlist, nprobe=nprobe, pq_m=pq_m, pq_nbits=pq_nbits,
        t_train=t_train, t_add=t_add, t_build=t_train + t_add,
        index_size_mb=size_mb, t_search_ms=t_search * 1000,
        ms_per_query=ms_per_q, qps=qps, recall_at_1=r1, recall_at_10=r10,
    )


def run_ivfpq_nprobe_sweep(
    xb: np.ndarray,
    xq: np.ndarray,
    xt: np.ndarray,
    gt: np.ndarray,
    *,
    nlist: int       = 1024,
    pq_m: int        = 8,
    pq_nbits: int    = 8,
    k: int           = 10,
    index_path: str  = "",
    dataset_tag: str = "",
) -> None:
    """Build IVFPQ once, sweep nprobe to show the recall/QPS trade-off."""
    tag = f", {dataset_tag}" if dataset_tag else ""
    print_header(f"IVFPQ nprobe sweep  (nlist={nlist}, m={pq_m}, nbits={pq_nbits}{tag})")
    header = f"{'nprobe':>7}  {'ms/q':>7}  {'QPS':>7}  {'R@1':>6}  {'R@10':>6}"
    print(header)
    print("-" * len(header))

    nb, d = xb.shape
    if d % pq_m != 0:
        raise ValueError(f"pq_m={pq_m} must divide d={d}")

    quantizer = faiss.IndexFlatL2(d)
    index     = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, pq_nbits)

    ivfpq_path = (index_path.replace(".idx", "") + f"_ivfpq_nlist{nlist}_m{pq_m}_b{pq_nbits}.idx"
                  if index_path else "")

    if ivfpq_path and os.path.exists(ivfpq_path):
        print(f"  Loading IVFPQ index from {ivfpq_path} …")
        index = faiss.read_index(ivfpq_path)
    else:
        print(f"  Training IVFPQ (nlist={nlist}, m={pq_m}, nbits={pq_nbits}) "
              f"on {len(xt):,} vectors …")
        index.train(xt)
        print(f"  Adding {nb:,} vectors …")
        add_batched(index, xb, verbose=True)
        if ivfpq_path:
            faiss.write_index(index, ivfpq_path)

    nq = xq.shape[0]
    for nprobe in IVF_NPROBE_SWEEP:
        if nprobe > nlist:
            break
        index.nprobe = nprobe
        for _ in range(3):  # warm-up (3 batches for stable cache)
            index.search(xq[:min(5, len(xq))], k)
        t0 = time.perf_counter()
        _, I = index.search(xq, k)
        t_s = time.perf_counter() - t0
        r1  = recall_at_k(I, gt, 1)
        r10 = recall_at_k(I, gt, min(10, k))
        print(f"{nprobe:>7}  {t_s/nq*1000:>7.3f}  {nq/t_s:>7.0f}  "
              f"{r1:>6.4f}  {r10:>6.4f}")


# ---------------------------------------------------------------------------
# OPQ + IVFPQ benchmark functions
# ---------------------------------------------------------------------------

def run_opqpq_benchmark(
    xb: np.ndarray,
    xq: np.ndarray,
    xt: np.ndarray,
    gt: np.ndarray,
    *,
    nlist: int       = 1024,
    pq_m: int        = 8,
    pq_nbits: int    = 8,
    nprobe: int      = 32,
    opq_niter: int   = 25,
    k: int           = 10,
    index_path: str  = "",
    verbose: bool    = True,
    dataset_tag: str = "",
) -> dict:
    """OPQ pre-rotation + IVFPQ: reduces IVFPQ's recall ceiling."""
    tag = f", {dataset_tag}" if dataset_tag else ""
    if verbose:
        print_header(f"OPQ+IVFPQ benchmark  "
                     f"(nlist={nlist}, m={pq_m}, nbits={pq_nbits}, "
                     f"nprobe={nprobe}{tag})")

    nb, d = xb.shape
    nq    = xq.shape[0]
    if d % pq_m != 0:
        raise ValueError(f"pq_m={pq_m} must divide d={d}")

    opqpq_path = (index_path.replace(".idx", "")
                  + f"_opqpq_nlist{nlist}_m{pq_m}_b{pq_nbits}.idx"
                  if index_path else "")

    if opqpq_path and os.path.exists(opqpq_path):
        if verbose:
            print(f"  Loading OPQ+IVFPQ index from {opqpq_path} \u2026")
        index = faiss.read_index(opqpq_path)
        t_train, t_add = 0.0, 0.0
    else:
        opq       = faiss.OPQMatrix(d, pq_m)
        opq.niter = opq_niter
        quantizer = faiss.IndexFlatL2(d)
        sub       = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, pq_nbits)
        index     = faiss.IndexPreTransform(opq, sub)
        index.verbose = False
        if verbose:
            print(f"  Training OPQ+IVFPQ "
                  f"(nlist={nlist}, m={pq_m}, nbits={pq_nbits}, "
                  f"opq_niter={opq_niter}) on {len(xt):,} vectors \u2026")
        t0 = time.perf_counter()
        index.train(xt)
        t_train = time.perf_counter() - t0
        if verbose:
            print(f"  Train done in {fmt_time(t_train)}")
            print(f"  Adding {nb:,} vectors \u2026")
        t_add = add_batched(index, xb, verbose=verbose)
        if verbose:
            print(f"  Add done in {fmt_time(t_add)}")
        if opqpq_path:
            if verbose:
                print(f"  Saving OPQ+IVFPQ index to {opqpq_path} \u2026")
            faiss.write_index(index, opqpq_path)

    size_mb = _index_size_mb(index)
    ivf     = faiss.extract_index_ivf(index)
    ivf.nprobe = nprobe
    for _ in range(3):  # warm-up (3 batches for stable cache)
            index.search(xq[:min(5, len(xq))], k)

    if verbose:
        print(f"  Searching {nq:,} queries (k={k}, nprobe={nprobe}) \u2026")
    t0 = time.perf_counter()
    D, I = index.search(xq, k)
    t_search = time.perf_counter() - t0
    ms_per_q = t_search / nq * 1000.0
    qps      = nq / t_search

    r1  = recall_at_k(I, gt, 1)
    r10 = recall_at_k(I, gt, min(10, k))

    if verbose:
        print(f"\n  Results:")
        print(f"    ntotal          = {index.ntotal:,}")
        print(f"    nlist           = {nlist}")
        print(f"    nprobe          = {nprobe}")
        print(f"    pq_m            = {pq_m}  (code_size={pq_m * pq_nbits}b = {pq_m * pq_nbits // 8}B/vec)")
        print(f"    pq_nbits        = {pq_nbits}")
        print(f"    opq_niter       = {opq_niter}")
        print(f"    train time      = {fmt_time(t_train)}")
        print(f"    add time        = {fmt_time(t_add)}")
        print(f"    build time      = {fmt_time(t_train + t_add)}")
        print(f"    index size      = {size_mb:.1f} MiB")
        print(f"    search time     = {t_search*1000:.1f}ms total")
        print(f"    ms / query      = {ms_per_q:.3f}")
        print(f"    QPS             = {qps:.0f}")
        print(f"    Recall@1        = {r1:.4f}")
        print(f"    Recall@10       = {r10:.4f}")

    return dict(
        nlist=nlist, nprobe=nprobe, pq_m=pq_m, pq_nbits=pq_nbits,
        opq_niter=opq_niter,
        t_train=t_train, t_add=t_add, t_build=t_train + t_add,
        index_size_mb=size_mb, t_search_ms=t_search * 1000,
        ms_per_query=ms_per_q, qps=qps, recall_at_1=r1, recall_at_10=r10,
    )


def run_opqpq_nprobe_sweep(
    xb: np.ndarray,
    xq: np.ndarray,
    xt: np.ndarray,
    gt: np.ndarray,
    *,
    nlist: int       = 1024,
    pq_m: int        = 8,
    pq_nbits: int    = 8,
    opq_niter: int   = 25,
    k: int           = 10,
    index_path: str  = "",
    dataset_tag: str = "",
) -> None:
    """Build OPQ+IVFPQ once, sweep nprobe to show the recall/QPS trade-off."""
    tag = f", {dataset_tag}" if dataset_tag else ""
    print_header(f"OPQ+IVFPQ nprobe sweep  "
                 f"(nlist={nlist}, m={pq_m}, nbits={pq_nbits}{tag})")
    header = f"{'nprobe':>7}  {'ms/q':>7}  {'QPS':>7}  {'R@1':>6}  {'R@10':>6}"
    print(header)
    print("-" * len(header))

    nb, d = xb.shape
    if d % pq_m != 0:
        raise ValueError(f"pq_m={pq_m} must divide d={d}")

    opqpq_path = (index_path.replace(".idx", "")
                  + f"_opqpq_nlist{nlist}_m{pq_m}_b{pq_nbits}.idx"
                  if index_path else "")

    if opqpq_path and os.path.exists(opqpq_path):
        print(f"  Loading OPQ+IVFPQ index from {opqpq_path} \u2026")
        index = faiss.read_index(opqpq_path)
    else:
        opq       = faiss.OPQMatrix(d, pq_m)
        opq.niter = opq_niter
        quantizer = faiss.IndexFlatL2(d)
        sub       = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, pq_nbits)
        index     = faiss.IndexPreTransform(opq, sub)
        index.verbose = False
        print(f"  Training OPQ+IVFPQ (nlist={nlist}, m={pq_m}, nbits={pq_nbits}, "
              f"opq_niter={opq_niter}) on {len(xt):,} vectors \u2026")
        index.train(xt)
        print(f"  Adding {nb:,} vectors \u2026")
        add_batched(index, xb, verbose=True)
        if opqpq_path:
            faiss.write_index(index, opqpq_path)

    ivf = faiss.extract_index_ivf(index)
    nq  = xq.shape[0]
    for nprobe in IVF_NPROBE_SWEEP:
        if nprobe > nlist:
            break
        ivf.nprobe = nprobe
        for _ in range(3):  # warm-up (3 batches for stable cache)
            index.search(xq[:min(5, len(xq))], k)
        t0 = time.perf_counter()
        _, I = index.search(xq, k)
        t_s = time.perf_counter() - t0
        r1  = recall_at_k(I, gt, 1)
        r10 = recall_at_k(I, gt, min(10, k))
        print(f"{nprobe:>7}  {t_s/nq*1000:>7.3f}  {nq/t_s:>7.0f}  "
              f"{r1:>6.4f}  {r10:>6.4f}")


# ---------------------------------------------------------------------------
# Comparison vs flat (brute force) baseline
# ---------------------------------------------------------------------------

def compare_vs_flat(xb, xq: np.ndarray, gt: np.ndarray, k: int = 10) -> None:
    print_header("Baseline: IndexFlatL2")
    nb = len(xb)
    if nb > 10_000_000:
        print(f"  SKIP: flat baseline is O(nb) memory; nb={nb:,} is too large.")
        print(f"  Re-run with --nb 1000000 or --nb 10000000 to get a flat baseline.")
        return
    flat = faiss.IndexFlatL2(xb.shape[1])
    add_batched(flat, xb, verbose=False)

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

# -------- dataset size helpers --------
_NB_TO_NAME = {
    1_000_000:   "Deep1M",
    10_000_000:  "Deep10M",
    100_000_000: "Deep100M",
}
# Sensible default maxtrain per dataset size
_NB_TO_MAXTRAIN = {
    1_000_000:     500_000,
    10_000_000:  1_000_000,
    100_000_000: 5_000_000,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark IndexSuCo on Deep1M / Deep10M / Deep100M.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir", default="data/",
        help="Root directory that contains the deep1b/ subdirectory.",
    )
    p.add_argument(
        "--nb", type=int, default=1_000_000,
        choices=[1_000_000, 10_000_000, 100_000_000],
        metavar="NB",
        help=("Dataset size: 1000000 (Deep1M), 10000000 (Deep10M), "
              "100000000 (Deep100M).  Default: 1000000."),
    )
    p.add_argument(
        "--maxtrain", type=int, default=None,
        help=("Number of training vectors to use. "
              "Default: 500K for 1M, 1M for 10M, 5M for 100M."),
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
    # ---- HNSW options ----
    p.add_argument(
        "--hnsw", action="store_true",
        help="Run an IndexHNSWFlat benchmark (single efSearch value).",
    )
    p.add_argument(
        "--hnsw-sweep", action="store_true",
        help="Build HNSW once, then sweep efSearch to show recall/QPS trade-off.",
    )
    p.add_argument(
        "--hnsw-M", type=int, default=32,
        help="HNSW M parameter (connections per node per layer).",
    )
    p.add_argument(
        "--hnsw-ef-construction", type=int, default=200,
        help="HNSW efConstruction (build-time quality).",
    )
    p.add_argument(
        "--hnsw-ef-search", type=int, default=128,
        help="HNSW efSearch (query-time beam width, used with --hnsw).",
    )
    # ---- IVFFlat options ----
    p.add_argument(
        "--ivfflat", action="store_true",
        help="Run a single IndexIVFFlat configuration.",
    )
    p.add_argument(
        "--ivfflat-sweep", action="store_true",
        help="Build IVFFlat once, then sweep nprobe to show recall/QPS trade-off.",
    )
    # ---- IVFPQ options ----
    p.add_argument(
        "--ivfpq", action="store_true",
        help="Run a single IndexIVFPQ configuration.",
    )
    p.add_argument(
        "--ivfpq-sweep", action="store_true",
        help="Build IVFPQ once, then sweep nprobe to show recall/QPS trade-off.",
    )
    # ---- OPQ+IVFPQ options ----
    p.add_argument(
        "--opqpq", action="store_true",
        help="Run a single OPQ+IVFPQ configuration.",
    )
    p.add_argument(
        "--opqpq-sweep", action="store_true",
        help="Build OPQ+IVFPQ once, then sweep nprobe to show recall/QPS trade-off.",
    )
    p.add_argument(
        "--opq-niter", type=int, default=25,
        help="Number of OPQMatrix training iterations.",
    )
    # ---- shared IVF parameters ----
    p.add_argument(
        "--nlist", type=int, default=1024,
        help="Number of IVF Voronoi cells.",
    )
    p.add_argument(
        "--nprobe", type=int, default=64,
        help="Number of IVF cells to probe at query time.",
    )
    p.add_argument(
        "--pq-m", type=int, default=8,
        help=(
            "Number of IVFPQ sub-quantizers (must divide d=96). "
            "Valid choices: 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96."
        ),
    )
    p.add_argument(
        "--pq-nbits", type=int, default=8,
        help="Bits per IVFPQ code per sub-quantizer (typically 8).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    dataset_name = _NB_TO_NAME.get(args.nb, f"Deep{args.nb//1_000_000}M")
    if args.maxtrain is None:
        args.maxtrain = _NB_TO_MAXTRAIN.get(args.nb, 500_000)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print_header(f"Loading {dataset_name} dataset  (nb={args.nb:,})")
    set_dataset_basedir(args.data_dir)
    ds = DatasetDeep1B(nb=args.nb)

    print(f"  data_dir  : {args.data_dir}")
    print(f"  dataset   : {dataset_name}")
    print(f"  d         : {ds.d}")
    print(f"  nb (base) : {ds.nb:,}")
    print(f"  nq        : {ds.nq:,}")
    print(f"  maxtrain  : {args.maxtrain:,}")
    print(f"  OMP threads: {faiss.omp_get_max_threads()}")

    print("  Loading queries …", end=" ", flush=True)
    xq = ds.get_queries()                              # (10000, 96)
    print(f"shape={xq.shape}")

    # For large datasets, keep base as a memory-mapped view to avoid a
    # 35+ GiB contiguous RAM allocation.  add_batched() slices it in chunks.
    base_fvecs = os.path.join(args.data_dir.rstrip("/"), "deep1b", "base.fvecs")
    print("  Mapping base (mmap) …", end=" ", flush=True)
    xb = fvecs_mmap(base_fvecs)[:args.nb]             # (nb, 96), pages load on demand
    print(f"shape={xb.shape}  dtype={xb.dtype}  (not loaded into RAM)")

    print("  Loading train …", end=" ", flush=True)
    xt = ds.get_train(maxtrain=args.maxtrain)
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
    print_header(f"IndexSuCo benchmark  ({dataset_name}, d=96)")
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
    # Optional SuCo parameter sweep
    # ------------------------------------------------------------------
    if args.sweep:
        run_sweep(xb, xq, xt, gt, k=args.k)

    # ------------------------------------------------------------------
    # HNSW single-configuration benchmark
    # ------------------------------------------------------------------
    if args.hnsw:
        print_header(f"IndexHNSWFlat benchmark  ({dataset_name}, d=96)")
        print(f"  M               = {args.hnsw_M}")
        print(f"  efConstruction  = {args.hnsw_ef_construction}")
        print(f"  efSearch        = {args.hnsw_ef_search}")
        print(f"  k               = {args.k}")
        run_hnsw_benchmark(
            xb, xq, gt,
            M=args.hnsw_M,
            ef_construction=args.hnsw_ef_construction,
            ef_search=args.hnsw_ef_search,
            k=args.k,
            index_path=args.index_path,
            verbose=True,
        )

    # ------------------------------------------------------------------
    # HNSW efSearch sweep  (recall vs QPS trade-off curve)
    # ------------------------------------------------------------------
    if args.hnsw_sweep:
        run_hnsw_ef_sweep(
            xb, xq, gt,
            M=args.hnsw_M,
            ef_construction=args.hnsw_ef_construction,
            k=args.k,
            index_path=args.index_path,
        )

    # ------------------------------------------------------------------
    # IVFFlat single-configuration benchmark
    # ------------------------------------------------------------------
    if args.ivfflat:
        print_header(f"IndexIVFFlat benchmark  ({dataset_name}, d=96)")
        print(f"  nlist           = {args.nlist}")
        print(f"  nprobe          = {args.nprobe}")
        print(f"  k               = {args.k}")
        run_ivfflat_benchmark(
            xb, xq, xt, gt,
            nlist=args.nlist,
            nprobe=args.nprobe,
            k=args.k,
            index_path=args.index_path,
            verbose=True,
        )

    # ------------------------------------------------------------------
    # IVFFlat nprobe sweep  (recall vs QPS trade-off curve)
    # ------------------------------------------------------------------
    if args.ivfflat_sweep:
        run_ivfflat_nprobe_sweep(
            xb, xq, xt, gt,
            nlist=args.nlist,
            k=args.k,
            index_path=args.index_path,
            dataset_tag=f"{dataset_name}, d=96",
        )

    # ------------------------------------------------------------------
    # IVFPQ single-configuration benchmark
    # ------------------------------------------------------------------
    if args.ivfpq:
        print_header(f"IndexIVFPQ benchmark  ({dataset_name}, d=96)")
        print(f"  nlist           = {args.nlist}")
        print(f"  nprobe          = {args.nprobe}")
        print(f"  pq_m            = {args.pq_m}")
        print(f"  pq_nbits        = {args.pq_nbits}")
        print(f"  k               = {args.k}")
        run_ivfpq_benchmark(
            xb, xq, xt, gt,
            nlist=args.nlist,
            nprobe=args.nprobe,
            pq_m=args.pq_m,
            pq_nbits=args.pq_nbits,
            k=args.k,
            index_path=args.index_path,
            verbose=True,
        )

    # ------------------------------------------------------------------
    # IVFPQ nprobe sweep  (recall vs QPS trade-off curve)
    # ------------------------------------------------------------------
    if args.ivfpq_sweep:
        run_ivfpq_nprobe_sweep(
            xb, xq, xt, gt,
            nlist=args.nlist,
            pq_m=args.pq_m,
            pq_nbits=args.pq_nbits,
            k=args.k,
            index_path=args.index_path,
            dataset_tag=f"{dataset_name}, d=96",
        )

    # ------------------------------------------------------------------
    # OPQ+IVFPQ single-configuration benchmark
    # ------------------------------------------------------------------
    if args.opqpq:
        print_header(f"OPQ+IVFPQ benchmark  ({dataset_name}, d=96)")
        print(f"  nlist           = {args.nlist}")
        print(f"  nprobe          = {args.nprobe}")
        print(f"  pq_m            = {args.pq_m}")
        print(f"  pq_nbits        = {args.pq_nbits}")
        print(f"  opq_niter       = {args.opq_niter}")
        print(f"  k               = {args.k}")
        run_opqpq_benchmark(
            xb, xq, xt, gt,
            nlist=args.nlist,
            nprobe=args.nprobe,
            pq_m=args.pq_m,
            pq_nbits=args.pq_nbits,
            opq_niter=args.opq_niter,
            k=args.k,
            index_path=args.index_path,
            verbose=True,
        )

    # ------------------------------------------------------------------
    # OPQ+IVFPQ nprobe sweep  (recall vs QPS trade-off curve)
    # ------------------------------------------------------------------
    if args.opqpq_sweep:
        run_opqpq_nprobe_sweep(
            xb, xq, xt, gt,
            nlist=args.nlist,
            pq_m=args.pq_m,
            pq_nbits=args.pq_nbits,
            opq_niter=args.opq_niter,
            k=args.k,
            index_path=args.index_path,
            dataset_tag=f"{dataset_name}, d=96",
        )


if __name__ == "__main__":
    main()
