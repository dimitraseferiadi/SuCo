#!/usr/bin/env python3
"""
benchs/bench_suco_spacev10m.py

Benchmark IndexSuCo (and HNSW / IVFFlat baselines) on a 10-million-vector
subset of the SpaceV dataset  (d=100, int8 vectors).

Usage
-----
# Basic run – single SuCo configuration:
    python benchs/bench_suco_spacev10m.py \\
        --data-dir /path/to/spacev10m/

# Save the SuCo index so subsequent runs skip the build step:
    python benchs/bench_suco_spacev10m.py \\
        --data-dir /path/to/spacev10m/ \\
        --index-path /path/to/spacev10m.idx

# Parameter sweep over (Ns, nc, α, β):
    python benchs/bench_suco_spacev10m.py --data-dir ... --sweep

# All comparisons in one shot:
    python benchs/bench_suco_spacev10m.py --data-dir ... \\
        --index-path /path/to/spacev10m.idx \\
        --flat-baseline --sweep --hnsw-sweep --ivfflat-sweep

Dataset files expected under --data-dir
    base.100M.i8bin          – 100M × 100 int8 base vectors
    query.30K.i8bin          – 29 316 × 100 int8 query vectors
    groundtruth.30K.i32bin   – 29 316 × 100 int32 GT against the *full* 100M

Notes
-----
*  Vectors are stored as int8 and are cast to float32 before indexing.
   SpaceV int8 coords are in [-128, 127]; the cast is lossless for L2 search.

*  The distributed GT file is computed against 100M vectors.  For an Nb-vector
   sub-index the GT is recomputed here via IndexFlatL2 on the loaded subset.
   Save the result with --gt-path to avoid recomputing on future runs.

*  Valid nsubspaces for d=100  (subspace_dim = d/Ns must be even):
   2, 5, 10, 25, 50  →  default: 10  (10 subspaces × 10 dims each)
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
except ImportError as e:
    sys.exit(f"Cannot import faiss: {e}\n"
             "Build FAISS with IndexSuCo and run from the repo root.")

# ---------------------------------------------------------------------------
# SpaceV binary format helpers
# ---------------------------------------------------------------------------

def read_i8bin(path: str, max_n: int | None = None) -> np.ndarray:
    """
    Read a .i8bin file  (4-byte n, 4-byte d header, then n*d int8 values).
    Returns float32 array of shape (min(n, max_n), d).
    """
    with open(path, "rb") as f:
        n, d = np.frombuffer(f.read(8), dtype=np.uint32)
        n = int(n); d = int(d)
        if max_n is not None:
            n = min(n, max_n)
        data = np.frombuffer(f.read(n * d), dtype=np.int8)
    return data.reshape(n, d).astype(np.float32)


def read_i32bin(path: str, max_n: int | None = None) -> np.ndarray:
    """
    Read a .i32bin file  (4-byte n, 4-byte d header, then n*d int32 values).
    Returns int32 array of shape (min(n, max_n), d).
    """
    with open(path, "rb") as f:
        n, d = np.frombuffer(f.read(8), dtype=np.uint32)
        n = int(n); d = int(d)
        if max_n is not None:
            n = min(n, max_n)
        data = np.frombuffer(f.read(n * d * 4), dtype=np.int32)
    return data.reshape(n, d)


# ---------------------------------------------------------------------------
# Groundtruth helpers
# ---------------------------------------------------------------------------

def compute_gt(xb: np.ndarray, xq: np.ndarray, k: int = 100,
               batch_size: int = 1000) -> np.ndarray:
    """
    Compute exact k-NN ground truth for xq against xb using IndexFlatL2.
    Batches the search to bound peak memory usage.
    """
    d   = xb.shape[1]
    nq  = xq.shape[0]
    gt  = np.empty((nq, k), dtype=np.int64)

    flat = faiss.IndexFlatL2(d)
    flat.add(xb)

    for start in range(0, nq, batch_size):
        end = min(start + batch_size, nq)
        _, I = flat.search(xq[start:end], k)
        gt[start:end] = I

    return gt


# ---------------------------------------------------------------------------
# Recall helpers
# ---------------------------------------------------------------------------

def recall_at_k(I: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Fraction of queries whose true NN is in the top-k results."""
    hits = (I[:, :k] == gt[:, :1]).any(axis=1).sum()
    return float(hits) / I.shape[0]


def recall_at_k_topR(I: np.ndarray, gt: np.ndarray, k: int, r: int) -> float:
    """Fraction of the true top-r neighbours found in top-k results."""
    nq = I.shape[0]
    hits = sum(len(set(I[i, :k]) & set(gt[i, :r])) for i in range(nq))
    return hits / (nq * r)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def fmt_time(seconds: float) -> str:
    return f"{seconds/60:.1f}min" if seconds >= 60 else f"{seconds:.2f}s"


def print_header(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


# ---------------------------------------------------------------------------
# Core benchmark routine
# ---------------------------------------------------------------------------

def run_benchmark(
    xb: np.ndarray,
    xq: np.ndarray,
    xt: np.ndarray,
    gt: np.ndarray,
    *,
    nsubspaces: int        = 10,
    ncentroids_half: int   = 50,
    collision_ratio: float = 0.05,
    candidate_ratio: float = 0.005,
    niter: int             = 10,
    k: int                 = 10,
    index_path: str        = "",
    verbose: bool          = True,
) -> dict:
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
            print(f"  Training on {len(xt):,} vectors …")
        t0 = time.perf_counter()
        index.train(xt)
        t_train = time.perf_counter() - t0
        if verbose:
            print(f"  Training done in {fmt_time(t_train)}")

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

    # Warm-up (not timed)
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
        index_size_mib = sum(os.path.getsize(index_path)
                             for p in [index_path] if os.path.exists(p)) / 1024**2
        print(f"\n  Results:")
        print(f"    ntotal          = {index.ntotal:,}")
        print(f"    train time      = {fmt_time(t_train)}")
        print(f"    add time        = {fmt_time(t_add)}")
        print(f"    search time     = {t_search*1000:.1f}ms total")
        print(f"    ms / query      = {ms_per_q:.3f}")
        print(f"    QPS             = {qps:.0f}")
        print(f"    Recall@1        = {r1:.4f}")
        print(f"    Recall@10       = {r10:.4f}")
        if index_path and os.path.exists(index_path):
            print(f"    index size      = {index_size_mib:.1f} MiB")

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
    )


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

# d=100: valid nsubspaces (d/Ns must be even): 2, 5, 10, 25, 50
SWEEP_CONFIGS = [
    # (nsubspaces, ncentroids_half, collision_ratio, candidate_ratio)
    # ---- varying collision_ratio (Ns=10 default) ----
    (10, 50, 0.02, 0.002),
    (10, 50, 0.05, 0.005),   # default
    (10, 50, 0.10, 0.010),
    (10, 50, 0.20, 0.020),
    # ---- varying nsubspaces ----
    (5,  50, 0.05, 0.005),
    (25, 50, 0.05, 0.005),
    # ---- varying ncentroids_half ----
    (10, 25,  0.05, 0.005),
    (10, 100, 0.05, 0.005),
]


def run_sweep(xb, xq, xt, gt, k: int = 10) -> None:
    print_header("Parameter Sweep  (SpaceV10M, d=100)")
    d = xb.shape[1]
    header = (
        f"{'Ns':>3}  {'hd':>3}  {'nc':>4}  {'alpha':>6}  {'beta':>7}  "
        f"{'ms/q':>7}  {'QPS':>7}  {'R@1':>6}  {'R@10':>6}"
    )
    print(header)
    print("-" * len(header))

    for ns, nc, alpha, beta in SWEEP_CONFIGS:
        hd = d // (2 * ns)
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
            f"{r['nsubspaces']:>3}  {hd:>3}  {r['ncentroids_half']:>4}  "
            f"{r['collision_ratio']:>6.3f}  {r['candidate_ratio']:>7.4f}  "
            f"{r['ms_per_query']:>7.3f}  {r['qps']:>7.0f}  "
            f"{r['recall_at_1']:>6.4f}  {r['recall_at_10']:>6.4f}"
        )


# ---------------------------------------------------------------------------
# Flat baseline
# ---------------------------------------------------------------------------

def compare_vs_flat(xb: np.ndarray, xq: np.ndarray, gt: np.ndarray,
                    k: int = 10) -> None:
    print_header("Baseline: IndexFlatL2  (SpaceV10M)")
    flat = faiss.IndexFlatL2(xb.shape[1])
    flat.add(xb)

    flat.search(xq[:1], k)  # warm-up

    t0 = time.perf_counter()
    _, I_flat = flat.search(xq, k)
    t_flat = time.perf_counter() - t0

    r1  = recall_at_k(I_flat, gt, 1)
    r10 = recall_at_k(I_flat, gt, min(10, k))
    print(f"  ms/query  = {t_flat/xq.shape[0]*1000:.3f}")
    print(f"  QPS       = {xq.shape[0]/t_flat:.0f}")
    print(f"  Recall@1  = {r1:.4f}  (upper bound for approximate methods)")
    print(f"  Recall@10 = {r10:.4f}")


# ---------------------------------------------------------------------------
# HNSW sweep
# ---------------------------------------------------------------------------

def run_hnsw_sweep(xb: np.ndarray, xq: np.ndarray, gt: np.ndarray,
                   k: int = 10, M: int = 32, ef_construction: int = 200) -> None:
    print_header(f"HNSW efSearch sweep  (M={M}, efConstruction={ef_construction})")
    ef_values = [8, 16, 32, 64, 128, 256, 512]

    header = f"{'efSearch':>8}  {'ms/q':>7}  {'QPS':>7}  {'R@1':>6}  {'R@10':>6}"
    print(header)
    print("-" * len(header))

    d  = xb.shape[1]
    nq = xq.shape[0]

    print(f"  Building HNSW index (M={M}, efConstruction={ef_construction}) …",
          flush=True)
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.add(xb)
    t_build = time.perf_counter() - t0
    print(f"  Build done in {fmt_time(t_build)}\n")

    index.search(xq[:1], k)  # warm-up

    for ef in ef_values:
        index.hnsw.efSearch = ef
        t0 = time.perf_counter()
        _, I = index.search(xq, k)
        t = time.perf_counter() - t0
        ms_q = t / nq * 1000
        qps  = nq / t
        r1   = recall_at_k(I, gt, 1)
        r10  = recall_at_k(I, gt, min(10, k))
        print(f"{ef:>8}  {ms_q:>7.3f}  {qps:>7.0f}  {r1:>6.4f}  {r10:>6.4f}")


# ---------------------------------------------------------------------------
# IVFFlat sweep
# ---------------------------------------------------------------------------

def run_ivfflat_sweep(xb: np.ndarray, xq: np.ndarray, gt: np.ndarray,
                      k: int = 10, nlist: int = 4096) -> None:
    print_header(f"IVFFlat nprobe sweep  (nlist={nlist}, SpaceV10M, d={xb.shape[1]})")
    nprobe_values = [1, 4, 16, 64, 128, 256, 512, 1024]

    header = f"{'nprobe':>7}  {'ms/q':>7}  {'QPS':>7}  {'R@1':>6}  {'R@10':>6}"
    print(header)
    print("-" * len(header))

    d  = xb.shape[1]
    nq = xq.shape[0]

    print(f"  Building IVFFlat index (nlist={nlist}) …", flush=True)
    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    # Use a random sample for training if xb is very large
    xt_ivf = xb if len(xb) <= 500_000 else xb[
        np.random.default_rng(42).choice(len(xb), 500_000, replace=False)
    ]
    index.train(xt_ivf)
    index.add(xb)
    t_build = time.perf_counter() - t0
    print(f"  Build done in {fmt_time(t_build)}\n")

    index.search(xq[:1], k)  # warm-up

    for nprobe in nprobe_values:
        if nprobe > nlist:
            continue
        index.nprobe = nprobe
        t0 = time.perf_counter()
        _, I = index.search(xq, k)
        t = time.perf_counter() - t0
        ms_q = t / nq * 1000
        qps  = nq / t
        r1   = recall_at_k(I, gt, 1)
        r10  = recall_at_k(I, gt, min(10, k))
        print(f"{nprobe:>7}  {ms_q:>7.3f}  {qps:>7.0f}  {r1:>6.4f}  {r10:>6.4f}")


# ---------------------------------------------------------------------------
# IVFPQ sweep
# ---------------------------------------------------------------------------

def run_ivfpq_sweep(xb: np.ndarray, xq: np.ndarray, gt: np.ndarray,
                   k: int = 10, nlist: int = 4096,
                   m: int = 10, nbits: int = 8) -> None:
    """
    IVFPQ nprobe sweep.  m PQ sub-quantisers, nbits bits each.
    For d=100: m must divide d → valid values: 2, 4, 5, 10, 20, 25, 50, 100.
    Default m=10 → 10 bytes / vector (compression ratio 40×).
    """
    print_header(
        f"IVFPQ nprobe sweep  (nlist={nlist}, m={m}, nbits={nbits}, "
        f"SpaceV10M, d={xb.shape[1]})"
    )
    nprobe_values = [1, 4, 16, 64, 128, 256, 512, 1024]

    header = f"{'nprobe':>7}  {'ms/q':>7}  {'QPS':>7}  {'R@1':>6}  {'R@10':>6}"
    print(header)
    print("-" * len(header))

    d  = xb.shape[1]
    nq = xq.shape[0]

    print(f"  Building IVFPQ index (nlist={nlist}, m={m}, nbits={nbits}) …",
          flush=True)
    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    xt_pq = xb if len(xb) <= 500_000 else xb[
        np.random.default_rng(42).choice(len(xb), 500_000, replace=False)
    ]
    index.train(xt_pq)
    index.add(xb)
    t_build = time.perf_counter() - t0
    print(f"  Build done in {fmt_time(t_build)}\n")

    index.search(xq[:1], k)  # warm-up

    for nprobe in nprobe_values:
        if nprobe > nlist:
            continue
        index.nprobe = nprobe
        t0 = time.perf_counter()
        _, I = index.search(xq, k)
        t = time.perf_counter() - t0
        ms_q = t / nq * 1000
        qps  = nq / t
        r1   = recall_at_k(I, gt, 1)
        r10  = recall_at_k(I, gt, min(10, k))
        print(f"{nprobe:>7}  {ms_q:>7.3f}  {qps:>7.0f}  {r1:>6.4f}  {r10:>6.4f}")


# ---------------------------------------------------------------------------
# OPQ + IVFPQ sweep
# ---------------------------------------------------------------------------

def run_opqpq_sweep(xb: np.ndarray, xq: np.ndarray, gt: np.ndarray,
                   k: int = 10, nlist: int = 4096,
                   m: int = 10, nbits: int = 8) -> None:
    """
    OPQ pre-rotation + IVFPQ nprobe sweep.
    OPQ rotates the space to equalise variance across PQ sub-spaces, usually
    improving recall by a few percentage points over plain IVFPQ.
    Built via faiss.index_factory using the string "OPQ{m},IVF{nlist},PQ{m}x{nbits}".
    """
    factory_str = f"OPQ{m},IVF{nlist},PQ{m}x{nbits}"
    print_header(
        f"OPQ+IVFPQ nprobe sweep  (factory='{factory_str}', SpaceV10M, d={xb.shape[1]})"
    )
    nprobe_values = [1, 4, 16, 64, 128, 256, 512, 1024]

    header = f"{'nprobe':>7}  {'ms/q':>7}  {'QPS':>7}  {'R@1':>6}  {'R@10':>6}"
    print(header)
    print("-" * len(header))

    d  = xb.shape[1]
    nq = xq.shape[0]

    print(f"  Building OPQ+IVFPQ index via index_factory('{factory_str}') …",
          flush=True)
    t0 = time.perf_counter()
    index = faiss.index_factory(d, factory_str)
    xt_opq = xb if len(xb) <= 500_000 else xb[
        np.random.default_rng(42).choice(len(xb), 500_000, replace=False)
    ]
    index.train(xt_opq)
    index.add(xb)
    t_build = time.perf_counter() - t0
    print(f"  Build done in {fmt_time(t_build)}\n")

    index.search(xq[:1], k)  # warm-up

    ivf = faiss.extract_index_ivf(index)
    for nprobe in nprobe_values:
        if nprobe > nlist:
            continue
        ivf.nprobe = nprobe
        t0 = time.perf_counter()
        _, I = index.search(xq, k)
        t = time.perf_counter() - t0
        ms_q = t / nq * 1000
        qps  = nq / t
        r1   = recall_at_k(I, gt, 1)
        r10  = recall_at_k(I, gt, min(10, k))
        print(f"{nprobe:>7}  {ms_q:>7.3f}  {qps:>7.0f}  {r1:>6.4f}  {r10:>6.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark IndexSuCo on SpaceV10M (d=100, int8 → float32).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir", default="data/spacev10m/",
        help="Directory containing base.100M.i8bin, query.30K.i8bin, "
             "groundtruth.30K.i32bin.",
    )
    p.add_argument(
        "--nb", type=int, default=10_000_000,
        help="Number of base vectors to index (at most 100M).",
    )
    p.add_argument(
        "--nq", type=int, default=None,
        help="Number of query vectors to use (default: all, ~29 316).",
    )
    p.add_argument(
        "--maxtrain", type=int, default=500_000,
        help="Number of training vectors for SuCo (sampled from xb).",
    )
    p.add_argument(
        "--k", type=int, default=10,
        help="Number of nearest neighbours to retrieve.",
    )
    # ---- ground truth ----
    p.add_argument(
        "--gt-path", default="",
        help="Path to a precomputed .npy GT file (shape [nq, k_gt]).  "
             "If the file does not exist it will be computed and saved here.",
    )
    p.add_argument(
        "--k-gt", type=int, default=100,
        help="Number of ground-truth neighbours to store when recomputing.",
    )
    # ---- SuCo parameters ----
    p.add_argument(
        "--nsubspaces", type=int, default=10,
        help="Number of subspaces Ns (d/Ns must be even; valid: 2,5,10,25,50).",
    )
    p.add_argument(
        "--ncentroids-half", type=int, default=50,
        help="K-means centroids per half-subspace (√K).",
    )
    p.add_argument(
        "--collision-ratio", type=float, default=0.05,
        help="α: fraction of dataset retrieved per subspace.",
    )
    p.add_argument(
        "--candidate-ratio", type=float, default=0.005,
        help="β: fraction of dataset in the re-rank pool.",
    )
    p.add_argument(
        "--niter", type=int, default=10,
        help="K-means iterations during training.",
    )
    p.add_argument(
        "--index-path", default="",
        help="Save/load the pre-built SuCo index here.",
    )
    # ---- which sections to run ----
    p.add_argument("--sweep",        action="store_true",
                   help="Run parameter sweep over (Ns, nc, α, β).")
    p.add_argument("--flat-baseline", action="store_true",
                   help="Run brute-force FlatL2 as upper-bound baseline.")
    p.add_argument("--hnsw-sweep",   action="store_true",
                   help="Run HNSW efSearch sweep.")
    p.add_argument("--ivfflat-sweep", action="store_true",
                   help="Run IVFFlat nprobe sweep.")
    p.add_argument("--hnsw-m",       type=int, default=32,
                   help="HNSW M parameter.")
    p.add_argument("--hnsw-ef-construction", type=int, default=200,
                   help="HNSW efConstruction parameter.")
    p.add_argument("--ivfflat-nlist", type=int, default=4096,
                   help="IVFFlat number of centroids.")
    p.add_argument("--ivfpq-sweep",   action="store_true",
                   help="Run IVFPQ nprobe sweep.")
    p.add_argument("--opqpq-sweep",   action="store_true",
                   help="Run OPQ+IVFPQ nprobe sweep.")
    p.add_argument("--pq-nlist",      type=int, default=4096,
                   help="nlist for IVFPQ and OPQ+IVFPQ indexes.")
    p.add_argument("--pq-m",          type=int, default=10,
                   help="Number of PQ sub-quantisers (must divide d=100; "
                        "valid: 2,4,5,10,20,25,50,100).")
    p.add_argument("--pq-nbits",      type=int, default=8,
                   help="Bits per PQ code (typically 8).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print_header(f"Loading SpaceV10M dataset  (d=100, nb={args.nb:,})")
    data_dir = args.data_dir

    base_path  = os.path.join(data_dir, "base.100M.i8bin")
    query_path = os.path.join(data_dir, "query.30K.i8bin")
    gt100m_path = os.path.join(data_dir, "groundtruth.30K.i32bin")

    for path in [base_path, query_path]:
        if not os.path.exists(path):
            sys.exit(f"Missing file: {path}")

    print(f"  data_dir  : {data_dir}")
    print(f"  nb (load) : {args.nb:,}")

    print("  Loading base vectors …", end=" ", flush=True)
    xb = read_i8bin(base_path, max_n=args.nb)
    print(f"shape={xb.shape}  dtype={xb.dtype}")

    print("  Loading queries …", end=" ", flush=True)
    xq = read_i8bin(query_path, max_n=args.nq)
    print(f"shape={xq.shape}")

    nq = xq.shape[0]

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------
    gt = None

    if args.gt_path and os.path.exists(args.gt_path):
        print(f"  Loading precomputed GT from {args.gt_path} …", end=" ", flush=True)
        gt = np.load(args.gt_path)
        print(f"shape={gt.shape}")
        if gt.shape[0] != nq:
            sys.exit(f"GT file has {gt.shape[0]} queries but xq has {nq}; "
                     "delete the GT file and rerun to recompute.")
    else:
        print(f"\n  Recomputing {args.k_gt}-NN ground truth with IndexFlatL2 …")
        print(f"  (xb: {xb.shape}, xq: {xq.shape})")
        print(f"  This may take several minutes and ~{xb.nbytes // 1024**3 + 1} GiB "
              "of RAM for the flat index.")
        t0 = time.perf_counter()
        gt = compute_gt(xb, xq, k=args.k_gt)
        t_gt = time.perf_counter() - t0
        print(f"  GT computed in {fmt_time(t_gt)}  shape={gt.shape}")
        if args.gt_path:
            np.save(args.gt_path, gt)
            print(f"  GT saved to {args.gt_path}")

    # ------------------------------------------------------------------
    # Training sample
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    maxtrain = min(args.maxtrain, len(xb))
    xt = xb[rng.choice(len(xb), maxtrain, replace=False)]
    print(f"\n  Training sample ready: {xt.shape}")

    # ------------------------------------------------------------------
    # Optional flat baseline
    # ------------------------------------------------------------------
    if args.flat_baseline:
        compare_vs_flat(xb, xq, gt, k=args.k)

    # ------------------------------------------------------------------
    # Main SuCo benchmark
    # ------------------------------------------------------------------
    d = xb.shape[1]
    hd = d // (2 * args.nsubspaces)
    print_header(f"IndexSuCo benchmark  (SpaceV10M, d={d})")
    print(f"  nsubspaces      = {args.nsubspaces}  (half_dim = {hd})")
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
    # Optional HNSW sweep
    # ------------------------------------------------------------------
    if args.hnsw_sweep:
        run_hnsw_sweep(
            xb, xq, gt,
            k=args.k,
            M=args.hnsw_m,
            ef_construction=args.hnsw_ef_construction,
        )

    # ------------------------------------------------------------------
    # Optional IVFFlat sweep
    # ------------------------------------------------------------------
    if args.ivfflat_sweep:
        run_ivfflat_sweep(xb, xq, gt, k=args.k, nlist=args.ivfflat_nlist)

    # ------------------------------------------------------------------
    # Optional IVFPQ sweep
    # ------------------------------------------------------------------
    if args.ivfpq_sweep:
        run_ivfpq_sweep(
            xb, xq, gt,
            k=args.k,
            nlist=args.pq_nlist,
            m=args.pq_m,
            nbits=args.pq_nbits,
        )

    # ------------------------------------------------------------------
    # Optional OPQ+IVFPQ sweep
    # ------------------------------------------------------------------
    if args.opqpq_sweep:
        run_opqpq_sweep(
            xb, xq, gt,
            k=args.k,
            nlist=args.pq_nlist,
            m=args.pq_m,
            nbits=args.pq_nbits,
        )


if __name__ == "__main__":
    main()
