#!/usr/bin/env python3
"""
benchs/bench_suco_gist1m.py

Benchmark faiss.IndexSuCo (and optionally IndexHNSWFlat / IndexFlatL2) on
the GIST1M dataset  (d=960, nb=1 000 000, nq=1 000).

Usage
-----
# Default run (nsubspaces=40, matches paper's half_dim≈12):
    python benchs/bench_suco_gist1m.py --data-dir /path/to/data/

# Parameter sweep over nsubspaces and collision_ratio:
    python benchs/bench_suco_gist1m.py --data-dir /path/to/data/ --sweep

# HNSW efSearch sweep for recall/QPS trade-off:
    python benchs/bench_suco_gist1m.py --data-dir /path/to/data/ --hnsw-sweep

# Save/reload index to skip rebuild:
    python benchs/bench_suco_gist1m.py --data-dir /path/to/data/ --index-path /tmp/suco_gist.idx

Dataset preparation
-------------------
Download from http://corpus-texmex.irisa.fr/ (ANN_GIST1M):

    mkdir -p /path/to/data/gist1M && cd /path/to/data/gist1M
    wget http://corpus-texmex.irisa.fr/gist.tar.gz
    tar -xzf gist.tar.gz
    mv gist/* . && rmdir gist && rm gist.tar.gz

Expected files (relative to --data-dir):
    gist1M/gist_base.fvecs          – 1 000 000 × 960  (~3.8 GB)
    gist1M/gist_learn.fvecs         –   500 000 × 960  (~1.9 GB)
    gist1M/gist_query.fvecs         –     1 000 × 960
    gist1M/gist_groundtruth.ivecs   –     1 000 × 100

Notes on nsubspaces for d=960
------------------------------
SuCo requires  d % nsubspaces == 0  AND  (d // nsubspaces) % 2 == 0.
Valid choices for d=960 and their half-subspace dimensions:

    nsubspaces   half_dim   comment
    ----------   --------   -------
         8          60      very high-d halves — likely weak recall
        16          30
        20          24
        24          20
        30          16
        32          15
        40          12      ← default here; matches paper's Deep1B setup
        48          10
        60           8
        80           6
        96           5      minimal halves

The default (nsubspaces=40) is chosen so that half_dim=12, the same ratio
used in the paper for Deep1B (d=96, nsubspaces=8, half_dim=6).  The sweep
covers several choices so you can observe the effect experimentally.
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
    from faiss.contrib.datasets import DatasetGIST1M, set_dataset_basedir
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
# Batched add helper
# ---------------------------------------------------------------------------

# 50K × 960 × 4 bytes ≈ 192 MB per batch
ADD_BATCH_SIZE = 50_000


def add_batched(index, xb, verbose: bool = False) -> float:
    """
    Add vectors in `xb` to `index` in chunks of ADD_BATCH_SIZE.
    Returns total wall-clock time in seconds.
    """
    nb = len(xb)
    t0 = time.perf_counter()
    for i0 in range(0, nb, ADD_BATCH_SIZE):
        batch = np.ascontiguousarray(xb[i0:i0 + ADD_BATCH_SIZE], dtype='float32')
        index.add(batch)
        if verbose:
            done = min(i0 + ADD_BATCH_SIZE, nb)
            print(f"\r    {done:>10,} / {nb:,}  ({done/nb*100:.1f}%)",
                  end="", flush=True)
    if verbose:
        print()
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Build + search
# ---------------------------------------------------------------------------

def run_benchmark(
    xb: np.ndarray,
    xq: np.ndarray,
    xt: np.ndarray,
    gt: np.ndarray,
    *,
    nsubspaces: int       = 40,
    ncentroids_half: int  = 50,
    collision_ratio: float = 0.05,
    candidate_ratio: float = 0.005,
    niter: int            = 10,
    k: int                = 10,
    index_path: str       = "",
    verbose: bool         = True,
) -> dict:
    """
    Build IndexSuCo (or load from index_path), search xq, return metrics dict.
    """
    nb, d = xb.shape
    nq    = xq.shape[0]

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
        if verbose:
            print(f"  Training on {len(xt):,} vectors  (d={d}) …")
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

        if index_path:
            if verbose:
                print(f"  Saving index to {index_path} …")
            index.write_index(index_path)

    size_mb = _index_size_mb(index)

    # Warm-up
    index.search(xq[:1], k)

    if verbose:
        print(f"  Searching {nq:,} queries (k={k}) …")
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
        print(f"    nsubspaces      = {nsubspaces}  (half_dim={d // nsubspaces // 2})")
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
        nsubspaces=nsubspaces,
        ncentroids_half=ncentroids_half,
        collision_ratio=collision_ratio,
        candidate_ratio=candidate_ratio,
        half_dim=d // nsubspaces // 2,
        t_train=t_train,
        t_add=t_add,
        t_build=t_train + t_add,
        index_size_mb=size_mb,
        t_search_ms=t_search * 1000,
        ms_per_query=ms_per_q,
        qps=qps,
        recall_at_1=r1,
        recall_at_10=r10,
    )


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

# All entries must satisfy: 960 % nsubspaces == 0 and (960 // nsubspaces) % 2 == 0
SWEEP_CONFIGS = [
    # (nsubspaces, ncentroids_half, collision_ratio, candidate_ratio)
    # ---- varying nsubspaces (half_dim shown in comment) ----
    ( 8, 50, 0.05, 0.005),   # half_dim=60  – low-subspace, very high-d halves
    (16, 50, 0.05, 0.005),   # half_dim=30
    (24, 50, 0.05, 0.005),   # half_dim=20
    (40, 50, 0.05, 0.005),   # half_dim=12  ← paper-equivalent default
    (60, 50, 0.05, 0.005),   # half_dim=8
    # ---- varying collision_ratio at the default nsubspaces=40 ----
    (40, 50, 0.02, 0.002),
    (40, 50, 0.10, 0.010),
    (40, 50, 0.20, 0.020),
    # ---- varying ncentroids_half at the default nsubspaces=40 ----
    (40, 25, 0.05, 0.005),
    (40, 100, 0.05, 0.005),
]


def run_sweep(xb, xq, xt, gt, k: int = 10) -> None:
    print_header("Parameter Sweep  (GIST1M, d=960)")
    header = (
        f"{'Ns':>3}  {'hd':>3}  {'nc':>4}  {'alpha':>6}  {'beta':>7}  "
        f"{'ms/q':>7}  {'QPS':>7}  {'R@1':>6}  {'R@10':>6}"
    )
    print(header)
    print("-" * len(header))

    for ns, nc, alpha, beta in SWEEP_CONFIGS:
        r = run_benchmark(
            xb, xq, xt, gt,
            nsubspaces=ns,
            ncentroids_half=nc,
            collision_ratio=alpha,
            candidate_ratio=beta,
            k=k,
            verbose=False,
        )
        print(
            f"{r['nsubspaces']:>3}  {r['half_dim']:>3}  {r['ncentroids_half']:>4}  "
            f"{r['collision_ratio']:>6.3f}  {r['candidate_ratio']:>7.4f}  "
            f"{r['ms_per_query']:>7.3f}  {r['qps']:>7.0f}  "
            f"{r['recall_at_1']:>6.4f}  {r['recall_at_10']:>6.4f}"
        )


# ---------------------------------------------------------------------------
# HNSW benchmark
# ---------------------------------------------------------------------------

HNSW_EF_SWEEP = [8, 16, 32, 64, 128, 256, 512]


def run_hnsw_benchmark(
    xb: np.ndarray,
    xq: np.ndarray,
    gt: np.ndarray,
    *,
    M: int               = 32,
    ef_construction: int = 200,
    ef_search: int       = 128,
    k: int               = 10,
    index_path: str      = "",
    verbose: bool        = True,
) -> dict:
    """Build IndexHNSWFlat (or load), search, return metrics dict."""
    nb, d = xb.shape
    nq    = xq.shape[0]

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

    size_mb = _index_size_mb(index)
    index.hnsw.efSearch = ef_search
    index.search(xq[:1], k)  # warm-up

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
        M=M, ef_construction=ef_construction, ef_search=ef_search,
        t_build=t_add, index_size_mb=size_mb, t_search_ms=t_search * 1000,
        ms_per_query=ms_per_q, qps=qps, recall_at_1=r1, recall_at_10=r10,
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
    """Build HNSW once, sweep efSearch to trace the recall/QPS curve."""
    print_header(f"HNSW efSearch sweep  (M={M}, efConstruction={ef_construction})")
    header = f"{'efSearch':>8}  {'ms/q':>7}  {'QPS':>7}  {'R@1':>6}  {'R@10':>6}"
    print(header)
    print("-" * len(header))

    nb, d = xb.shape
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction

    hnsw_path = (index_path.replace(".idx", "") + f"_hnsw_M{M}_efc{ef_construction}.idx"
                 if index_path else "")

    if hnsw_path and os.path.exists(hnsw_path):
        print(f"  Loading HNSW index from {hnsw_path} …")
        index = faiss.read_index(hnsw_path)
    else:
        print(f"  Building HNSW (M={M}, efConstruction={ef_construction}) "
              f"on {nb:,} vectors …")
        t_build = add_batched(index, xb, verbose=True)
        print(f"  Build done in {fmt_time(t_build)}")
        if hnsw_path:
            faiss.write_index(index, hnsw_path)

    nq = xq.shape[0]
    for efs in HNSW_EF_SWEEP:
        index.hnsw.efSearch = efs
        index.search(xq[:1], k)  # warm-up
        t0 = time.perf_counter()
        _, I = index.search(xq, k)
        t_s = time.perf_counter() - t0
        r1  = recall_at_k(I, gt, 1)
        r10 = recall_at_k(I, gt, min(10, k))
        print(f"{efs:>8}  {t_s/nq*1000:>7.3f}  {nq/t_s:>7.0f}  "
              f"{r1:>6.4f}  {r10:>6.4f}")


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
    index.search(xq[:1], k)  # warm-up

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
        index.search(xq[:1], k)  # warm-up
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
    pq_m: int       = 48,
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
    index.search(xq[:1], k)  # warm-up

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
    pq_m: int        = 48,
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
        index.search(xq[:1], k)  # warm-up
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
    pq_m: int        = 48,
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
    index.search(xq[:1], k)  # warm-up

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
    pq_m: int        = 48,
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
        index.search(xq[:1], k)  # warm-up
        t0 = time.perf_counter()
        _, I = index.search(xq, k)
        t_s = time.perf_counter() - t0
        r1  = recall_at_k(I, gt, 1)
        r10 = recall_at_k(I, gt, min(10, k))
        print(f"{nprobe:>7}  {t_s/nq*1000:>7.3f}  {nq/t_s:>7.0f}  "
              f"{r1:>6.4f}  {r10:>6.4f}")


# ---------------------------------------------------------------------------
# Flat baseline
# ---------------------------------------------------------------------------

def compare_vs_flat(xb: np.ndarray, xq: np.ndarray, gt: np.ndarray,
                    k: int = 10) -> None:
    print_header("Baseline: IndexFlatL2  (GIST1M)")
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

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark IndexSuCo on GIST1M (d=960, nb=1M, nq=1000).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir", default="data/",
        help="Root directory containing the gist1M/ subdirectory.",
    )
    p.add_argument(
        "--maxtrain", type=int, default=100_000,
        help="Number of training vectors (max 500K available).",
    )
    p.add_argument(
        "--k", type=int, default=10,
        help="Number of nearest neighbours to retrieve.",
    )
    p.add_argument(
        "--nsubspaces", type=int, default=40,
        help=(
            "Number of subspaces. Must divide 960 and 960/nsubspaces must be even. "
            "Default 40 gives half_dim=12 (paper-equivalent ratio). "
            "Other good choices: 8(hd=60), 16(hd=30), 24(hd=20), 60(hd=8)."
        ),
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
        help="Save/load the SuCo index at this path.",
    )
    p.add_argument(
        "--sweep", action="store_true",
        help="Sweep nsubspaces, collision_ratio, and ncentroids_half.",
    )
    p.add_argument(
        "--flat-baseline", action="store_true",
        help="Run brute-force IndexFlatL2 as an upper-bound baseline.",
    )
    p.add_argument(
        "--hnsw", action="store_true",
        help="Run a single IndexHNSWFlat configuration.",
    )
    p.add_argument(
        "--hnsw-sweep", action="store_true",
        help="Build HNSW once, then sweep efSearch.",
    )
    p.add_argument("--hnsw-M", type=int, default=32)
    p.add_argument("--hnsw-ef-construction", type=int, default=200)
    p.add_argument("--hnsw-ef-search", type=int, default=128)
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
        "--pq-m", type=int, default=48,
        help=(
            "Number of IVFPQ sub-quantizers (must divide d=960). "
            "Valid choices: 8, 16, 20, 24, 32, 40, 48, 60, 80, 96, 120, 160, 192."
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

    # Validate nsubspaces early so the error is clear
    d = 960
    if d % args.nsubspaces != 0 or (d // args.nsubspaces) % 2 != 0:
        sys.exit(
            f"Error: nsubspaces={args.nsubspaces} is invalid for d={d}.\n"
            f"  Require: {d} % nsubspaces == 0  AND  ({d} // nsubspaces) % 2 == 0.\n"
            f"  Valid choices: 8, 16, 20, 24, 30, 32, 40, 48, 60, 80, 96."
        )

    print_header("Loading GIST1M dataset  (d=960, nb=1 000 000)")
    set_dataset_basedir(args.data_dir)
    ds = DatasetGIST1M()

    print(f"  data_dir  : {args.data_dir}")
    print(f"  d         : {ds.d}")
    print(f"  nb (base) : {ds.nb:,}")
    print(f"  maxtrain  : {args.maxtrain:,}")

    print("  Loading base …", end=" ", flush=True)
    xb = ds.get_database()                          # (1_000_000, 960) ≈ 3.8 GB
    print(f"shape={xb.shape}")

    print("  Loading queries …", end=" ", flush=True)
    xq = ds.get_queries()                           # (1_000, 960) — actual file has 1K
    print(f"shape={xq.shape}")

    print("  Loading train …", end=" ", flush=True)
    xt = ds.get_train(maxtrain=args.maxtrain)       # up to 500K available
    print(f"shape={xt.shape}")

    print("  Loading ground truth …", end=" ", flush=True)
    gt = ds.get_groundtruth(k=100)                  # (1_000, 100)
    print(f"shape={gt.shape}")

    # Actual nq may differ from ds.nq (the class declares 10000, file has 1000)
    nq_actual = xq.shape[0]
    if nq_actual != ds.nq:
        print(f"  Note: actual nq={nq_actual:,} (DatasetGIST1M.nq={ds.nq:,} — file has {nq_actual:,} queries)")

    # ------------------------------------------------------------------
    # Optional flat baseline
    # ------------------------------------------------------------------
    if args.flat_baseline:
        compare_vs_flat(xb, xq, gt, k=args.k)

    # ------------------------------------------------------------------
    # Main SuCo benchmark
    # ------------------------------------------------------------------
    half_dim = d // args.nsubspaces // 2
    print_header(f"IndexSuCo benchmark  (GIST1M, d=960)")
    print(f"  nsubspaces      = {args.nsubspaces}  (half_dim={half_dim})")
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
    # Optional parameter sweep
    # ------------------------------------------------------------------
    if args.sweep:
        run_sweep(xb, xq, xt, gt, k=args.k)

    # ------------------------------------------------------------------
    # HNSW single-configuration
    # ------------------------------------------------------------------
    if args.hnsw:
        print_header("IndexHNSWFlat benchmark  (GIST1M, d=960)")
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
    # HNSW efSearch sweep
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
        print_header("IndexIVFFlat benchmark  (GIST1M, d=960)")
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
            dataset_tag="GIST1M, d=960",
        )

    # ------------------------------------------------------------------
    # IVFPQ single-configuration benchmark
    # ------------------------------------------------------------------
    if args.ivfpq:
        print_header("IndexIVFPQ benchmark  (GIST1M, d=960)")
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
            dataset_tag="GIST1M, d=960",
        )

    # ------------------------------------------------------------------
    # OPQ+IVFPQ single-configuration benchmark
    # ------------------------------------------------------------------
    if args.opqpq:
        print_header("OPQ+IVFPQ benchmark  (GIST1M, d=960)")
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
            dataset_tag="GIST1M, d=960",
        )


if __name__ == "__main__":
    main()
