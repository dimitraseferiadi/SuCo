#!/usr/bin/env python3
"""
bench_fig678_paper.py
─────────────────────
Generates benchmark data for paper Figures 2 (Table 2), 6, 7, and 8.

  --mode table2   Table 2 – Recall of SC-search returning k=50 NNs
                  under different datasets and (Ns, nc, α, β) configs.

  --mode fig6     Figure 6 – Query efficiency: Dynamic Activation
                  vs Multi-sequence algorithm on SIFT10M.
                  Sweeps collision_ratio from very small to large and
                  records QPS + R@1 for both algorithms.

  --mode fig7     Figure 7 – Effect of K (nc²) and Ns on QPS/R@1.
                  Varies K = {10²,25²,50²,75²,100²} with Ns fixed,
                  and Ns = {2,4,8,16,32} with nc fixed at 50.

  --mode fig8     Figure 8 – Effect of α and β independently.
                  Panel a: fix β=0.005, vary α ∈ {0.01…0.20}.
                  Panel b: fix α=0.05, vary β ∈ {0.001…0.020}.

All modes write a tab-separated TSV to --out.  A run script section at the
bottom shows how to call each mode for each dataset.

Usage examples
--------------
# Table 2 on SIFT10M
python benchs/bench_fig678_paper.py \\
    --mode table2 --dataset sift10m \\
    --mat-path /path/SIFT10M/SIFT10Mfeatures.mat \\
    --index-path /path/indices/sift10m.idx \\
    --gt-path /path/indices/sift10m_gt.npy \\
    --out benchs/table2_sift10m.tsv

# Figure 6 on SIFT10M (uses search_multisequence)
python benchs/bench_fig678_paper.py \\
    --mode fig6 --dataset sift10m \\
    --mat-path /path/SIFT10M/SIFT10Mfeatures.mat \\
    --index-path /path/indices/sift10m.idx \\
    --gt-path /path/indices/sift10m_gt.npy \\
    --out benchs/fig6_sift10m.tsv

# Figure 7 on SIFT10M (builds multiple index configs)
python benchs/bench_fig678_paper.py \\
    --mode fig7 --dataset sift10m \\
    --mat-path /path/SIFT10M/SIFT10Mfeatures.mat \\
    --index-path /path/indices/sift10m.idx \\
    --out benchs/fig7_sift10m.tsv

# Figure 8 on SIFT10M
python benchs/bench_fig678_paper.py \\
    --mode fig8 --dataset sift10m \\
    --mat-path /path/SIFT10M/SIFT10Mfeatures.mat \\
    --index-path /path/indices/sift10m.idx \\
    --gt-path /path/indices/sift10m_gt.npy \\
    --out benchs/fig8_sift10m.tsv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import tempfile

import faiss
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Shared data helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fvecs_read(path: str, n: int | None = None) -> np.ndarray:
    from faiss.contrib.vecs_io import fvecs_mmap

    x = fvecs_mmap(path)
    if n is None:
        return np.ascontiguousarray(x.astype("float32", copy=False))
    if n > x.shape[0]:
        raise ValueError(
            f"Requested {n} vectors from {path}, but file contains only {x.shape[0]}"
        )
    return np.ascontiguousarray(x[:n].astype("float32", copy=False))


def _load_sift10m_mat(mat_path: str, nb: int, nq_cap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SIFT10M .mat file, supporting both classic MAT and MATLAB v7.3 (HDF5)."""
    from scipy.io import loadmat

    try:
        data = loadmat(mat_path)
        key = [k for k in data if not k.startswith("_")][0]
        xraw = data[key]
    except NotImplementedError:
        # MATLAB v7.3 files are HDF5-based and require h5py.
        try:
            import h5py
        except ImportError as e:
            raise RuntimeError(
                "SIFT10M .mat appears to be MATLAB v7.3 (HDF5). "
                "Install h5py in your environment: pip install h5py"
            ) from e

        with h5py.File(mat_path, "r") as f:
            keys = [k for k in f.keys()]
            if not keys:
                raise RuntimeError(f"No datasets found in {mat_path}")

            preferred = ["fea", "features", "X", "data"]
            key = next((k for k in preferred if k in f), None)
            if key is None:
                for k in keys:
                    if isinstance(f[k], h5py.Dataset) and f[k].ndim == 2:
                        key = k
                        break
            if key is None:
                raise RuntimeError(
                    f"Could not locate a 2D feature dataset in {mat_path}. "
                    f"Available keys: {keys}"
                )
            xraw = f[key][()]

    xraw = np.asarray(xraw)
    if xraw.ndim != 2:
        raise RuntimeError(f"Expected 2D matrix in {mat_path}, got shape {xraw.shape}")

    # Normalize to (N, d)
    if xraw.shape[1] == 128:
        X = np.ascontiguousarray(xraw.astype("float32"))
    elif xraw.shape[0] == 128:
        X = np.ascontiguousarray(xraw.T.astype("float32"))
    else:
        raise RuntimeError(
            f"Could not infer feature orientation from shape {xraw.shape}; expected one axis to be 128"
        )

    xb = X[:nb]
    xq = X[nb: nb + nq_cap]
    xt = xb[:500_000]
    return xb, xt, xq


def _ivecs_read(path: str, n: int | None = None) -> np.ndarray:
    with open(path, "rb") as f:
        d = np.frombuffer(f.read(4), dtype=np.int32)[0]
        f.seek(0)
        if n is None:
            data = np.fromfile(f, dtype=np.int32)
        else:
            data = np.frombuffer(f.read(n * (4 + d * 4)), dtype=np.int32)
    return data.reshape(-1, d + 1)[:, 1:].copy()


def _auto_discover_gt_path(args, ds: str, data_dir: str, index_path: str) -> str:
    candidates = []

    if ds == "deep10m":
        candidates.extend([
            os.path.join(data_dir, "deep1b", "deep10M_groundtruth.ivecs"),
            os.path.join(data_dir, "deep1b", "deep10M_groundtruth.npy"),
            os.path.join(os.path.dirname(index_path), "deep10m_gt.npy") if index_path else "",
            os.path.join(os.path.dirname(index_path), "deep_gt.npy") if index_path else "",
        ])
    elif ds == "sift10m":
        candidates.extend([
            os.path.join(os.path.dirname(index_path), "sift10m_gt.npy") if index_path else "",
            os.path.join(data_dir, "sift10m_gt.npy"),
            os.path.join(data_dir, "SIFT10M", "sift10m_gt.npy"),
        ])
    elif ds == "sift1m":
        for cand in ("sift1M", "sift1m", "sift"):
            p = os.path.join(data_dir, cand)
            candidates.extend([
                os.path.join(p, "sift_groundtruth.ivecs"),
                os.path.join(p, "groundtruth.ivecs"),
            ])

    for p in candidates:
        if p and os.path.exists(p):
            return p
    return ""


def load_dataset(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return xb, xt (train), xq, gt."""
    ds = args.dataset
    nb = getattr(args, "nb", 10_000_000)
    nq_cap = 10_000

    if ds == "sift10m":
        xb, xt, xq = _load_sift10m_mat(args.mat_path, nb, nq_cap)
    elif ds == "sift1m":
        dd = args.data_dir
        sift_dir = None
        for cand in ("sift1M", "sift1m", "sift"):
            p = os.path.join(dd, cand)
            if os.path.exists(os.path.join(p, "sift_base.fvecs")):
                sift_dir = p
                break
        if sift_dir is None:
            raise FileNotFoundError(
                "Could not find SIFT1M files under --data-dir. "
                "Expected one of: sift1M/, sift1m/, sift/ containing sift_base.fvecs"
            )

        xb = _fvecs_read(os.path.join(sift_dir, "sift_base.fvecs"))
        xq = _fvecs_read(os.path.join(sift_dir, "sift_query.fvecs"))
        xt = _fvecs_read(os.path.join(sift_dir, "sift_learn.fvecs"))
    elif ds == "deep10m":
        from scipy.io import loadmat
        d96  = 96
        base = os.path.join(args.data_dir, "deep1b", "base.fvecs")
        qry  = os.path.join(args.data_dir, "deep1b", "deep1B_queries.fvecs")
        xb   = _fvecs_read(base, 10_000_000)
        xq   = _fvecs_read(qry,  nq_cap)
        xt   = xb[:1_000_000]
    else:
        sys.exit(f"Dataset '{ds}' not yet supported in this script for this mode. "
                 f"Use sift10m, sift1m, or deep10m.")

    gt = None
    gt_path = args.gt_path if (args.gt_path and os.path.exists(args.gt_path)) else ""
    if not gt_path:
        gt_path = _auto_discover_gt_path(args, ds, args.data_dir, args.index_path)

    if gt_path:
        # Accept both .npy and raw ivecs files.
        if gt_path.endswith(".npy"):
            gt = np.load(gt_path)
        else:
            gt = _ivecs_read(gt_path)
        print(f"  Loaded ground truth: {gt_path}")

    return xb, xt, xq[:nq_cap], gt


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def recall_at_1(I: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(I[:, 0] == gt[:, 0]))


def recall_at_k(I: np.ndarray, gt: np.ndarray, k: int) -> float:
    k = min(k, I.shape[1], gt.shape[1])
    hits = np.array([
        len(np.intersect1d(I[i, :k], gt[i, :k]))
        for i in range(len(I))
    ])
    return float(np.mean(hits) / k)


def recall_topk_in_r(I: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Fraction of true k-NNs found in returned k results (k-Recall@k)."""
    k = min(k, I.shape[1], gt.shape[1])
    hits = np.array([len(np.intersect1d(I[i, :k], gt[i, :k])) for i in range(len(I))])
    return float(np.mean(hits) / k)


def fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    return f"{s/60:.1f}min"


def add_batched(index, xb: np.ndarray, batch: int = 100_000) -> float:
    t0 = time.perf_counter()
    for i in range(0, len(xb), batch):
        index.add(xb[i: i + batch])
    return time.perf_counter() - t0


def build_index(d, ns, nc, alpha, beta, xt: np.ndarray, xb: np.ndarray,
                save_path: str = ""):
    idx = faiss.IndexSuCo(d, ns, nc, alpha, beta, 10)
    idx.train(xt)
    add_batched(idx, xb)
    if save_path:
        idx.write_index(save_path)
    return idx


def load_or_build(args, ns, nc, alpha=0.05, beta=0.005,
                  xt=None, xb=None) -> "faiss.IndexSuCo":
    """Load index from disk if compatible, otherwise build."""
    # For Fig6/8 we can reuse the pre-built default index since the IMI
    # structure doesn't change when only α/β change.
    if args.index_path and os.path.exists(args.index_path):
        try:
            idx = faiss.read_index(args.index_path)
        except RuntimeError as e:
            # Backward compatibility: some SuCo indices were serialized via
            # IndexSuCo.write_index(path) and carry the custom 'SuCo' header.
            # faiss.read_index() expects FAISS global dispatch headers (e.g. IxSC).
            msg = str(e)
            if "oCuS" in msg or "SuCo" in msg:
                if xb is None:
                    raise RuntimeError(
                        "Legacy SuCo index requires xb (to infer d) for fallback read"
                    ) from e
                idx = faiss.IndexSuCo(int(xb.shape[1]))
                idx.read_index(args.index_path)
            else:
                raise
        if idx.nsubspaces == ns and idx.ncentroids_half == nc:
            idx.collision_ratio = alpha
            idx.candidate_ratio = beta
            return idx
    # Build fresh
    d = xb.shape[1]
    print(f"  Building index (Ns={ns}, nc={nc}, α={alpha}, β={beta}) …",
          flush=True)
    idx = build_index(d, ns, nc, alpha, beta, xt, xb)
    return idx


def warmup(idx, xq: np.ndarray, k: int = 10, n: int = 5):
    idx.search(xq[:n], k)


def timed_search(idx, xq: np.ndarray, k: int, repeats: int = 1):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        D, I = idx.search(xq, k)
        best = min(best, time.perf_counter() - t0)
    return D, I, best


def timed_search_ms(idx, xq: np.ndarray, k: int,
                    cr_override: float = -1.0,
                    cdr_override: float = -1.0):
    """Time search_multisequence."""
    D = np.empty((len(xq), k), np.float32)
    I = np.empty((len(xq), k), np.int64)
    t0 = time.perf_counter()
    idx.search_multisequence(
        len(xq), faiss.swig_ptr(xq), k,
        faiss.swig_ptr(D), faiss.swig_ptr(I),
        cr_override, cdr_override)
    return D, I, time.perf_counter() - t0


# ──────────────────────────────────────────────────────────────────────────────
# TABLE 2 – SC-Linear recall when returning k=50 NNs
# ──────────────────────────────────────────────────────────────────────────────
# "SC-Linear" = the SuCo index used as-is (full-precision re-ranking) with k=50.
# The "recall" is the k-Recall@k metric: what fraction of the true 50 NNs are
# found in the returned 50 results.

TABLE2_CONFIGS = [
    # (nsubspaces, ncentroids_half, collision_ratio, candidate_ratio,  label)
    (8,  50, 0.02, 0.002,  "Ns=8  nc=50  α=0.02"),
    (8,  50, 0.05, 0.005,  "Ns=8  nc=50  α=0.05  [default]"),
    (8,  50, 0.10, 0.010,  "Ns=8  nc=50  α=0.10"),
    (8,  50, 0.20, 0.020,  "Ns=8  nc=50  α=0.20"),
    (4,  50, 0.05, 0.005,  "Ns=4  nc=50  α=0.05"),
    (16, 50, 0.05, 0.005,  "Ns=16 nc=50  α=0.05"),
    (8,  25, 0.05, 0.005,  "Ns=8  nc=25  α=0.05"),
    (8, 100, 0.05, 0.005,  "Ns=8  nc=100 α=0.05"),
]


def run_table2(args, xb, xt, xq, gt, writer):
    """Table 2: k-Recall@50 for various configs."""
    d  = xb.shape[1]
    nq = xq.shape[0]
    k  = 50

    print("\n  Table 2: SC-Linear recall (k=50)")
    hdr = ("label", "Ns", "nc", "alpha", "beta",
           "QPS", "R@1", "R@10", "50-R@50")
    writer.writerow(hdr)

    # Group configs by (Ns, nc) to avoid re-building unnecessarily
    built: dict[tuple, faiss.IndexSuCo] = {}

    for ns, nc, alpha, beta, label in TABLE2_CONFIGS:
        key = (ns, nc)
        if key not in built:
            built[key] = load_or_build(args, ns, nc, alpha, beta, xt, xb)

        idx = built[key]
        idx.collision_ratio = alpha
        idx.candidate_ratio = beta

        warmup(idx, xq, k)
        D, I, t_s = timed_search(idx, xq, k)
        qps = nq / t_s
        r1  = recall_at_1(I, gt)  if gt is not None else float("nan")
        r10 = recall_at_k(I, gt, 10) if gt is not None else float("nan")
        r50 = recall_topk_in_r(I, gt, 50) if gt is not None else float("nan")

        row = (label, ns, nc, alpha, beta,
               f"{qps:.0f}", f"{r1:.4f}", f"{r10:.4f}", f"{r50:.4f}")
        writer.writerow(row)
        print(f"  {label:40s}  QPS={qps:6.0f}  R@1={r1:.4f}  "
              f"R@10={r10:.4f}  50-R@50={r50:.4f}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 6 – Dynamic Activation vs Multi-sequence
# ──────────────────────────────────────────────────────────────────────────────

# Sweep collision_ratio values for the comparison.
# We hold candidate_ratio proportional (β = α/10) so the rerank pool scales
# with the collision pool.
FIG6_ALPHAS = [0.005, 0.010, 0.020, 0.030, 0.050, 0.070, 0.100, 0.150, 0.200]
FIG6_BETA_RATIO = 0.10   # β = α × β_ratio


def run_fig6(args, xb, xt, xq, gt, writer):
    """Figure 6: Dynamic Activation vs Multi-sequence efficiency."""
    d  = xb.shape[1]
    nq = xq.shape[0]
    k  = 10

    print("\n  Figure 6: Dynamic Activation vs Multi-sequence  (SIFT10M)")
    writer.writerow(("algorithm", "alpha", "beta", "candidates",
                     "QPS", "ms_per_q", "R@1", "R@10"))

    # Load or build the default index (Ns=8, nc=50)
    idx = load_or_build(args, 8, 50, 0.05, 0.005, xt, xb)

    ntotal = idx.ntotal
    print(f"  ntotal = {ntotal:,}  Ns={idx.nsubspaces}  nc={idx.ncentroids_half}")

    for alpha in FIG6_ALPHAS:
        beta = round(alpha * FIG6_BETA_RATIO, 6)
        candidates = int(alpha * ntotal)

        # ---- Dynamic Activation ----
        idx.collision_ratio = alpha
        idx.candidate_ratio = beta
        warmup(idx, xq, k)
        D_da, I_da, t_da = timed_search(idx, xq, k)
        qps_da = nq / t_da
        r1_da  = recall_at_1(I_da, gt) if gt is not None else float("nan")
        r10_da = recall_at_k(I_da, gt, 10) if gt is not None else float("nan")

        writer.writerow(("DynamicActivation", alpha, beta, candidates,
                         f"{qps_da:.0f}", f"{t_da/nq*1000:.3f}",
                         f"{r1_da:.4f}", f"{r10_da:.4f}"))

        # ---- Multi-sequence ----
        warmup(idx, xq, k)   # warm-up using regular search
        D_ms, I_ms, t_ms = timed_search_ms(idx, xq, k, alpha, beta)
        qps_ms = nq / t_ms
        r1_ms  = recall_at_1(I_ms, gt) if gt is not None else float("nan")
        r10_ms = recall_at_k(I_ms, gt, 10) if gt is not None else float("nan")

        writer.writerow(("MultiSequence", alpha, beta, candidates,
                         f"{qps_ms:.0f}", f"{t_ms/nq*1000:.3f}",
                         f"{r1_ms:.4f}", f"{r10_ms:.4f}"))

        print(f"  α={alpha:.3f}  cands≈{candidates//1000}K  "
              f"DynAct: {qps_da:6.0f} QPS  R@1={r1_da:.4f}  |  "
              f"MultiSeq: {qps_ms:6.0f} QPS  R@1={r1_ms:.4f}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 7 – Varying K (= nc²) and Ns
# ──────────────────────────────────────────────────────────────────────────────

# Valid Ns values for each dataset dimension:
#   d=128 (SIFT): subspace_dim = d/Ns must be even → Ns ∈ {1,2,4,8,16,32,64}
#   d=96  (Deep): Ns ∈ {1,2,3,4,6,8,12,16,24,32,48}
#   d=960 (GIST): many options
FIG7_NS_SIFT   = [2, 4, 8, 16, 32]
FIG7_NS_DEEP   = [2, 4, 6, 8, 12, 16]
FIG7_NC_VALS   = [10, 25, 50, 75, 100]   # √K values; K = nc²


def _valid_ns(d: int) -> list[int]:
    return [n for n in range(1, d+1) if d % n == 0 and (d // n) % 2 == 0]


def _is_valid_suco_ns(d: int, ns: int) -> bool:
    return ns > 0 and d % ns == 0 and (d // ns) % 2 == 0


def run_fig7(args, xb, xt, xq, gt, writer):
    """Figure 7: Effect of K and Ns on QPS/R@1."""
    d  = xb.shape[1]
    nq = xq.shape[0]
    k  = 10

    valid_all = _valid_ns(d)
    if d == 96:
        preferred_ns = FIG7_NS_DEEP
    elif d == 128:
        preferred_ns = FIG7_NS_SIFT
    else:
        preferred_ns = [n for n in valid_all if n <= 32]

    ns_vals = [n for n in preferred_ns if _is_valid_suco_ns(d, n)]
    if not ns_vals:
        ns_vals = [n for n in valid_all if n <= 32][:6] or valid_all[:6]

    print(f"  Ns sweep candidates: {ns_vals}")

    writer.writerow(("sweep_var", "x_val", "Ns", "nc", "K",
                     "alpha", "beta", "QPS", "R@1", "R@10",
                     "build_s", "index_mb"))

    # ---- Panel a: vary Ns with nc=50 fixed ----
    print("\n  Figure 7a: Vary Ns  (nc=50 fixed, α=0.05, β=0.005)")
    built_ns: dict[int, faiss.IndexSuCo] = {}
    for ns in ns_vals:
        if not _is_valid_suco_ns(d, ns):
            print(f"    Skipping invalid Ns={ns} for d={d}")
            continue
        if ns not in built_ns:
            print(f"    Building Ns={ns} …", flush=True)
            t0 = time.perf_counter()
            built_ns[ns] = build_index(d, ns, 50, 0.05, 0.005, xt, xb)
            t_build = time.perf_counter() - t0
        else:
            t_build = 0.0

        idx = built_ns[ns]
        warmup(idx, xq, k)
        D, I, t_s = timed_search(idx, xq, k)
        qps  = nq / t_s
        r1   = recall_at_1(I, gt) if gt is not None else float("nan")
        r10  = recall_at_k(I, gt, 10) if gt is not None else float("nan")
        mb   = idx.ntotal * d * 4 / 1024**2   # rough (just the vectors)

        writer.writerow(("Ns", ns, ns, 50, 50*50, 0.05, 0.005,
                         f"{qps:.0f}", f"{r1:.4f}", f"{r10:.4f}",
                         f"{t_build:.1f}", f"{mb:.0f}"))
        print(f"    Ns={ns:3d}  QPS={qps:6.0f}  R@1={r1:.4f}  R@10={r10:.4f}")

    # ---- Panel b: vary nc (=√K) with Ns=8 fixed ----
    print("\n  Figure 7b: Vary nc  (Ns=8 fixed, α=0.05, β=0.005)")
    ns_fixed = 8 if 8 in _valid_ns(d) else _valid_ns(d)[min(3, len(_valid_ns(d))-1)]
    built_nc: dict[int, faiss.IndexSuCo] = {}
    for nc in FIG7_NC_VALS:
        if nc not in built_nc:
            print(f"    Building nc={nc}  (K={nc*nc}) …", flush=True)
            t0 = time.perf_counter()
            built_nc[nc] = build_index(d, ns_fixed, nc, 0.05, 0.005, xt, xb)
            t_build = time.perf_counter() - t0
        else:
            t_build = 0.0

        idx = built_nc[nc]
        warmup(idx, xq, k)
        D, I, t_s = timed_search(idx, xq, k)
        qps  = nq / t_s
        r1   = recall_at_1(I, gt) if gt is not None else float("nan")
        r10  = recall_at_k(I, gt, 10) if gt is not None else float("nan")
        mb   = idx.ntotal * d * 4 / 1024**2

        writer.writerow(("nc", nc, ns_fixed, nc, nc*nc, 0.05, 0.005,
                         f"{qps:.0f}", f"{r1:.4f}", f"{r10:.4f}",
                         f"{t_build:.1f}", f"{mb:.0f}"))
        print(f"    nc={nc:3d}  K={nc*nc:6d}  QPS={qps:6.0f}  "
              f"R@1={r1:.4f}  R@10={r10:.4f}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 8 – Varying α and β independently
# ──────────────────────────────────────────────────────────────────────────────

FIG8_ALPHA_VALS  = [0.010, 0.020, 0.030, 0.050, 0.070, 0.100, 0.150, 0.200]
FIG8_BETA_FIXED  = 0.005   # β held constant for α sweep (panel a)
FIG8_ALPHA_FIXED = 0.050   # α held constant for β sweep (panel b)
FIG8_BETA_VALS   = [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020]


def run_fig8(args, xb, xt, xq, gt, writer):
    """Figure 8: QPS / R@1 as α and β vary independently."""
    nq = xq.shape[0]
    k  = 10

    writer.writerow(("sweep_var", "x_val", "alpha", "beta",
                     "QPS", "ms_per_q", "R@1", "R@10"))

    # Load base index (Ns=8, nc=50) – changing α/β doesn't require rebuild
    idx = load_or_build(args, 8, 50, 0.05, 0.005, xt, xb)

    # ---- Panel a: vary α, β fixed ----
    print(f"\n  Figure 8a: Vary α  (β={FIG8_BETA_FIXED} fixed, Ns=8, nc=50)")
    for alpha in FIG8_ALPHA_VALS:
        idx.collision_ratio = alpha
        idx.candidate_ratio = FIG8_BETA_FIXED
        warmup(idx, xq, k)
        D, I, t_s = timed_search(idx, xq, k)
        qps = nq / t_s
        r1  = recall_at_1(I, gt) if gt is not None else float("nan")
        r10 = recall_at_k(I, gt, 10) if gt is not None else float("nan")
        writer.writerow(("alpha", alpha, alpha, FIG8_BETA_FIXED,
                         f"{qps:.0f}", f"{t_s/nq*1000:.3f}",
                         f"{r1:.4f}", f"{r10:.4f}"))
        print(f"  α={alpha:.3f}  β={FIG8_BETA_FIXED:.3f}  "
              f"QPS={qps:6.0f}  R@1={r1:.4f}")

    # ---- Panel b: vary β, α fixed ----
    print(f"\n  Figure 8b: Vary β  (α={FIG8_ALPHA_FIXED} fixed, Ns=8, nc=50)")
    for beta in FIG8_BETA_VALS:
        idx.collision_ratio = FIG8_ALPHA_FIXED
        idx.candidate_ratio = beta
        warmup(idx, xq, k)
        D, I, t_s = timed_search(idx, xq, k)
        qps = nq / t_s
        r1  = recall_at_1(I, gt) if gt is not None else float("nan")
        r10 = recall_at_k(I, gt, 10) if gt is not None else float("nan")
        writer.writerow(("beta", beta, FIG8_ALPHA_FIXED, beta,
                         f"{qps:.0f}", f"{t_s/nq*1000:.3f}",
                         f"{r1:.4f}", f"{r10:.4f}"))
        print(f"  α={FIG8_ALPHA_FIXED:.3f}  β={beta:.3f}  "
              f"QPS={qps:6.0f}  R@1={r1:.4f}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True,
                   choices=["table2", "fig6", "fig7", "fig8"],
                   help="Which paper figure/table to generate data for.")
    p.add_argument("--dataset", required=True,
                   choices=["sift10m", "sift1m", "deep10m"],
                   help="Dataset identifier.")
    p.add_argument("--nb", type=int, default=10_000_000)
    p.add_argument("--index-path", default="",
                   help="Path to pre-built IndexSuCo (.idx). Load if (Ns,nc) match.")
    p.add_argument("--gt-path", default="",
                   help="Path to ground-truth .npy (shape [nq, k_gt]).")
    p.add_argument("--out", default="",
                   help="Output TSV path (default: <mode>_<dataset>.tsv).")
    # Dataset source paths
    p.add_argument("--data-dir", default="")
    p.add_argument("--mat-path", default="",
                   help="SIFT10M: path to .mat file.")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*70}")
    print(f"  Mode    : {args.mode}")
    print(f"  Dataset : {args.dataset}")
    print(f"{'='*70}\n")

    xb, xt, xq, gt = load_dataset(args)
    print(f"  d={xb.shape[1]}  nb={len(xb):,}  nq={len(xq):,}")
    if gt is None:
        print("  Warning: no ground truth loaded; recall metrics (R@k) will be NaN.")
        print("           Pass --gt-path or place dataset GT where auto-discovery can find it.")

    out = args.out or os.path.join(
        os.path.dirname(__file__),
        f"{args.mode}_{args.dataset}.tsv")

    with open(out, "w", newline="") as fout:
        writer = csv.writer(fout, delimiter="\t")
        if args.mode == "table2":
            run_table2(args, xb, xt, xq, gt, writer)
        elif args.mode == "fig6":
            run_fig6(args, xb, xt, xq, gt, writer)
        elif args.mode == "fig7":
            run_fig7(args, xb, xt, xq, gt, writer)
        elif args.mode == "fig8":
            run_fig8(args, xb, xt, xq, gt, writer)

    print(f"  Results written to {out}")


if __name__ == "__main__":
    main()
