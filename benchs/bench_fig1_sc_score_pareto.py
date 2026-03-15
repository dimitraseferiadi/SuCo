#!/usr/bin/env python3
"""
bench_fig1_sc_score_pareto.py
─────────────────────────────
Compute the SC-score Pareto property (paper Figure 1).

For each of `--nq` queries the script:
  1. Runs `index.get_sc_scores()` to obtain the full per-point SC-score vector.
  2. Sorts all database points by exact L2 distance (using IndexFlatL2) to get
     the true-rank ordering.
  3. Accumulates: avg_sc_at_rank[rank_i] += sc_score[rank_i]

After all queries the accumulated sums are divided by nq and written to a TSV:

    rank  avg_sc_score

which can be plotted as a scatter of (rank, avg_sc_score).

The paper plots this for Deep10M, SIFT10M and SpaceV10M; this script handles
any dataset that the caller loads via one of these backends:

  --dataset sift10m   → loads the .mat file used by bench_suco_sift10m.py
  --dataset deep10m   → loads Deep1B sub-set used by bench_suco_deep1b.py
  --dataset spacev10m → loads SpaceV10M used by bench_suco_spacev10m.py
  --dataset sift1m    → SIFT1M (for quick validation)
  --dataset deep1m    → Deep1M

Typical run time (Mac, Apple Silicon, 10 threads, nq=100):
  SIFT10M  ≈ 10 min  (argsort 10M floats × 100 queries)
  Deep10M  ≈ 10 min
  SIFT1M   ≈  3 min  (for validation)

Output
------
TSV file written to --out (default: pareto_sc_scores_<dataset>.tsv).
Columns: rank  avg_sc_score
The TSV has a header row and exactly ntotal data rows.

Usage example
-------------
python benchs/bench_fig1_sc_score_pareto.py \\
    --dataset sift10m \\
    --mat-path /path/to/SIFT10M/SIFT10Mfeatures.mat \\
    --index-path /path/to/indices/sift10m.idx \\
    --gt-path /path/to/indices/sift10m_gt.npy \\
    --nq 100 \\
    --cr 0.05 \\
    --out benchs/pareto_sc_scores_sift10m.tsv
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import faiss
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers (reuse patterns from existing bench scripts)
# ──────────────────────────────────────────────────────────────────────────────

def _load_sift10m(mat_path: str, nb: int, nq: int):
    """Load SIFT10M from .mat file."""
    from scipy.io import loadmat
    print(f"  Loading SIFT10M from {mat_path} …")
    try:
        data = loadmat(mat_path)
        key = [k for k in data if not k.startswith("_")][0]
        Xraw = data[key]
    except NotImplementedError:
        # MATLAB v7.3 files are HDF5-based and require h5py instead of scipy.io.loadmat.
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

            # Prefer common names used in MATLAB exports, otherwise pick
            # the first 2D dataset.
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
            Xraw = f[key][()]

    Xraw = np.asarray(Xraw)
    # Stored layout is usually (d, N); normalize to (N, d).
    if Xraw.ndim != 2:
        raise RuntimeError(f"Expected 2D matrix in {mat_path}, got shape {Xraw.shape}")
    if Xraw.shape[1] == 128:
        X = np.ascontiguousarray(Xraw.astype("float32"))
    elif Xraw.shape[0] == 128:
        X = np.ascontiguousarray(Xraw.T.astype("float32"))
    else:
        raise RuntimeError(
            f"Could not infer feature orientation from shape {Xraw.shape}; expected one axis to be 128"
        )
    assert X.ndim == 2 and X.shape[1] == 128, f"Expected (N,128), got {X.shape}"
    xb = X[:nb]
    xq = X[nb: nb + nq]
    return xb, xq


def _load_deep(data_dir: str, nb: int, nq: int):
    """Load Deep1B subset."""
    base_path = os.path.join(data_dir, "deep1b", "base.fvecs")
    qpath     = os.path.join(data_dir, "deep1b", "deep1B_queries.fvecs")
    def _fvecs(path, n):
        from faiss.contrib.vecs_io import fvecs_mmap

        x = fvecs_mmap(path)
        if n > x.shape[0]:
            raise ValueError(
                f"Requested {n} vectors from {path}, but file contains only {x.shape[0]}"
            )
        # Keep a contiguous float32 array for downstream FAISS operations.
        return np.ascontiguousarray(x[:n].astype("float32", copy=False))
    print(f"  Loading Deep vectors (nb={nb:,}) …")
    xb = _fvecs(base_path, nb)
    xq = _fvecs(qpath, nq)
    return xb, xq


def _load_spacev10m(data_dir: str, nb: int, nq: int):
    """Load SpaceV10M."""
    base_path = os.path.join(data_dir, "spacev10m", "base.i8bin")
    qpath     = os.path.join(data_dir, "spacev10m", "query.i8bin")
    def _i8bin(path, n):
        with open(path, "rb") as f:
            total, d = np.frombuffer(f.read(8), dtype=np.int32)
            buf = np.frombuffer(f.read(n * d), dtype=np.int8).reshape(n, d)
        return np.ascontiguousarray(buf.astype("float32"))
    print(f"  Loading SpaceV10M (nb={nb:,}) …")
    xb = _i8bin(base_path, nb)
    xq = _i8bin(qpath, nq)
    return xb, xq


def _load_sift1m(data_dir: str):
    """Load SIFT1M from standard fvecs layout."""
    def _fvecs(path):
        with open(path, "rb") as f:
            d = np.frombuffer(f.read(4), dtype=np.int32)[0]
            f.seek(0)
            data = np.fromfile(f, dtype=np.float32)
        return data.reshape(-1, d + 1)[:, 1:]
    base_dir = None
    for cand in ("sift1M", "sift1m", "sift"):
        p = os.path.join(data_dir, cand)
        if os.path.exists(os.path.join(p, "sift_base.fvecs")):
            base_dir = p
            break
    if base_dir is None:
        raise FileNotFoundError(
            "Could not find SIFT1M files under data dir. "
            "Expected one of: sift1M/, sift1m/, sift/ containing sift_base.fvecs"
        )

    xb = _fvecs(os.path.join(base_dir, "sift_base.fvecs"))
    xq = _fvecs(os.path.join(base_dir, "sift_query.fvecs"))
    return np.ascontiguousarray(xb.astype("float32")), np.ascontiguousarray(xq.astype("float32"))


# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_sc_pareto(
    index: "faiss.IndexSuCo",
    xb: np.ndarray,
    xq: np.ndarray,
    cr: float,
    verbose: bool = True,
) -> np.ndarray:
    """
    Return avg_sc_at_rank[i] = average SC-score of the i-th closest database
    point  (i=0 is the nearest neighbour) across all nq queries.

    Shape: (ntotal,)   dtype: float32
    """
    ntotal, d = xb.shape
    nq        = xq.shape[0]

    assert index.ntotal == ntotal, "index.ntotal mismatch"

    # Build a FlatL2 index for exact rank computation.
    flat = faiss.IndexFlatL2(d)
    flat.add(xb)

    acc = np.zeros(ntotal, dtype=np.float64)

    # get_sc_scores is not always exposed in Python builds; keep a fast
    # NumPy fallback that implements the same collision-count definition.
    has_get_sc_scores = hasattr(index, "get_sc_scores")

    effective_cr = float(cr)
    if not (0.0 < effective_cr < 1.0):
        effective_cr = float(index.collision_ratio)
    if not (0.0 < effective_cr < 1.0):
        raise ValueError(
            f"Invalid collision ratio: {effective_cr}. Expected value in (0,1)."
        )

    nsubspaces = int(index.nsubspaces)
    subspace_dim = d // nsubspaces
    collision_num = max(1, int(effective_cr * ntotal))

    if not has_get_sc_scores:
        print("  Note: IndexSuCo.get_sc_scores() not exposed in this Python build; "
              "using NumPy fallback.")

    def _sc_scores_numpy(qrow: np.ndarray) -> np.ndarray:
        sc_local = np.zeros(ntotal, dtype=np.uint16)
        for s in range(nsubspaces):
            st = s * subspace_dim
            ed = st + subspace_dim
            diff = xb[:, st:ed] - qrow[st:ed]
            dsub = np.einsum("ij,ij->i", diff, diff, optimize=True)
            top = np.argpartition(dsub, collision_num - 1)[:collision_num]
            sc_local[top] += 1
        return sc_local.astype(np.int32)

    for qi in range(nq):
        if verbose and (qi % max(1, nq // 10) == 0):
            pct = 100 * qi / nq
            print(f"    query {qi:5d}/{nq}  ({pct:.0f}%) …", flush=True)

        q = np.ascontiguousarray(xq[qi : qi + 1], dtype=np.float32)

        # 1. Exact L2 distances → rank ordering via argsort
        t0 = time.perf_counter()
        D_full, I_full = flat.search(q, ntotal)   # shape (1, ntotal)
        # I_full[0, i] = index of the i-th closest point
        rank_order = I_full[0]                    # rank_order[i] = label of rank-i point
        t_flat = time.perf_counter() - t0

        # 2. SC-scores for this query
        t0 = time.perf_counter()
        if has_get_sc_scores:
            sc = np.zeros(ntotal, dtype=np.int32)
            index.get_sc_scores(faiss.swig_ptr(q.ravel()), effective_cr,
                                faiss.swig_ptr(sc))
        else:
            sc = _sc_scores_numpy(q[0])
        t_sc = time.perf_counter() - t0

        # 3. Accumulate: acc[rank_i] += sc_score of the point at rank i
        acc += sc[rank_order].astype(np.float64)

        if verbose and qi == 0:
            print(f"      [first query] flat={t_flat*1000:.1f}ms  "
                  f"sc_scores={t_sc*1000:.1f}ms  "
                  f"est. total={(t_flat+t_sc)*nq/60:.1f} min")

    return (acc / nq).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_tsv(avg_sc: np.ndarray, path: str) -> None:
    ntotal = len(avg_sc)
    print(f"  Writing {ntotal:,} rows → {path} …")
    with open(path, "w") as f:
        f.write("rank\tavg_sc_score\n")
        for i, v in enumerate(avg_sc):
            f.write(f"{i+1}\t{v:.6f}\n")
    print(f"  Saved.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   choices=["sift10m", "deep10m", "spacev10m", "sift1m", "deep1m"],
                   help="Which dataset to use.")
    p.add_argument("--index-path", required=True,
                   help="Path to a pre-built IndexSuCo binary (.idx).")
    p.add_argument("--nb", type=int, default=10_000_000,
                   help="Number of database vectors (default 10M).")
    p.add_argument("--nq", type=int, default=100,
                   help="Number of queries to average over (default 100).")
    p.add_argument("--cr", type=float, default=-1.0,
                   help="collision_ratio override for get_sc_scores "
                        "(default -1 = use index default).")
    p.add_argument("--out", default="",
                   help="Output TSV path. Default: pareto_sc_scores_<dataset>.tsv "
                        "next to this script.")
    # Dataset-specific paths
    p.add_argument("--data-dir",  default="",
                   help="Root data directory (for sift1m / deep / spacev10m).")
    p.add_argument("--mat-path",  default="",
                   help="Path to SIFT10M .mat file.")
    p.add_argument("--gt-path",   default="",
                   help="(Unused) ground-truth path; reserved for future use.")
    p.add_argument("--nq-max",    type=int, default=10_000,
                   help="Maximum number of query vectors available in the file "
                        "(clips to --nq). Default 10 000.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    nq_load = min(args.nq, args.nq_max)
    if args.dataset == "sift10m":
        if not args.mat_path:
            sys.exit("--mat-path is required for --dataset sift10m")
        xb, xq_all = _load_sift10m(args.mat_path, args.nb, nq_load)
    elif args.dataset == "deep10m":
        if not args.data_dir:
            sys.exit("--data-dir is required for --dataset deep10m")
        xb, xq_all = _load_deep(args.data_dir, args.nb, nq_load)
    elif args.dataset == "spacev10m":
        if not args.data_dir:
            sys.exit("--data-dir is required for --dataset spacev10m")
        xb, xq_all = _load_spacev10m(args.data_dir, args.nb, nq_load)
    elif args.dataset == "sift1m":
        if not args.data_dir:
            sys.exit("--data-dir is required for --dataset sift1m")
        xb_full, xq_all = _load_sift1m(args.data_dir)
        xb = xb_full[:args.nb]
    elif args.dataset == "deep1m":
        if not args.data_dir:
            sys.exit("--data-dir is required for --dataset deep1m")
        xb_full, xq_all = _load_deep(args.data_dir, args.nb, nq_load)
        xb = xb_full[:args.nb]
    else:
        sys.exit(f"Unknown dataset: {args.dataset}")

    xq = xq_all[: args.nq]
    d  = xb.shape[1]
    ntotal = xb.shape[0]
    print(f"\n  Dataset : {args.dataset}")
    print(f"  d       : {d}")
    print(f"  ntotal  : {ntotal:,}")
    print(f"  nq      : {len(xq):,}")

    # ── Load index ────────────────────────────────────────────────────────────
    print(f"\n  Loading index from {args.index_path} …")
    t0 = time.perf_counter()
    try:
        index = faiss.read_index(args.index_path)
    except RuntimeError as e:
        # Backward compatibility: some SuCo indices were serialized with the
        # custom IndexSuCo header (magic 'SuCo') via index.write_index(path).
        # faiss.read_index() expects FAISS global dispatch headers (e.g. IxSC).
        msg = str(e)
        if "oCuS" in msg or "SuCo" in msg:
            index = faiss.IndexSuCo(d)
            index.read_index(args.index_path)
        else:
            raise
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s  "
          f"(ntotal={index.ntotal:,}  Ns={index.nsubspaces}  "
          f"nc={index.ncentroids_half}  α={index.collision_ratio})")
    assert index.ntotal == ntotal, \
        f"Index ntotal ({index.ntotal}) != data ntotal ({ntotal})"

    # ── Compute Pareto SC-scores ───────────────────────────────────────────────
    print(f"\n  Computing SC-score Pareto  (cr={args.cr}) …")
    t_start = time.perf_counter()
    avg_sc = compute_sc_pareto(index, xb, xq, cr=args.cr, verbose=True)
    t_total = time.perf_counter() - t_start
    print(f"\n  Done in {t_total/60:.1f} min")
    print(f"  avg_sc[0]  = {avg_sc[0]:.4f}  (nearest neighbour)")
    print(f"  avg_sc[-1] = {avg_sc[-1]:.4f}  (farthest point)")
    print(f"  avg_sc[99] = {avg_sc[99]:.4f}  (rank 100)")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = args.out or os.path.join(
        os.path.dirname(__file__),
        f"pareto_sc_scores_{args.dataset}.tsv")
    save_tsv(avg_sc, out)


if __name__ == "__main__":
    main()
