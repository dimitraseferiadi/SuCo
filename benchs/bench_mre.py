#!/usr/bin/env python3
"""
benchs/bench_mre.py

Compute Mean Relative Error (MRE) for IndexLSH on Deep1M and SIFT10M,
matching the LSH baselines used in the SuCo paper.

  MRE = mean_{q, j in [k]}[ L2(q, ann_j) / L2(q, true_j) ] - 1

Distances are always recomputed from xb using the returned index matrix I
(not from D, which contains Hamming distances for LSH).

Datasets
--------
  Deep1M   d=96,  nb=1M,  nq=10K  — loaded via faiss.contrib.datasets.DatasetDeep1B
  SIFT10M  d=128, nb=10M, nq=10K  — loaded from SIFT10Mfeatures.mat (HDF5/mat v7.3)
                                     GT loaded from /Users/dhm/Documents/indices/sift10m_gt.npy

LSH index files are reused from /Users/dhm/Documents/indices/ (same naming as
the other bench scripts):
    /Users/dhm/Documents/indices/sift10m_lsh_nbits{N}.idx
    /Users/dhm/Documents/indices/deep_nb1000000_lsh_nbits{N}.idx

Usage
-----
    # Both datasets (all paths are defaults — no flags needed):
    python benchs/bench_mre.py

    # Deep1M only:
    python benchs/bench_mre.py --skip-sift

    # Custom nbits sweeps:
    python benchs/bench_mre.py \\
        --deep-nbits 96 192 384 768 1536 \\
        --sift-nbits 128 256 512 1024
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    import faiss
    from faiss.contrib.datasets import DatasetDeep1B, set_dataset_basedir
except ImportError as e:
    sys.exit(f"Cannot import faiss: {e}\n"
             "Build FAISS with IndexSuCo and install the Python bindings.")


# ─────────────────────────────────────────────────────────────────────────────
# MRE  (works for LSH because we use I, not D)
# ─────────────────────────────────────────────────────────────────────────────

def mre(xb: np.ndarray, xq: np.ndarray, I: np.ndarray,
        gt: np.ndarray, k: int = 10) -> float:
    """
    Mean Relative Error over the k-NN result set.

    For every (query q, rank j in [k]):
        ratio_{q,j} = L2(xq[q], xb[I[q,j]]) / L2(xq[q], xb[gt[q,j]])

    MRE = mean(ratio) - 1.

    Distances are recomputed from the raw vectors (not from D), so this is
    valid for IndexLSH whose search() returns Hamming distances.
    Pairs where the true distance is 0 are treated as ratio = 1.
    """
    k_use = min(k, I.shape[1], gt.shape[1])
    nq    = len(xq)
    xq64  = xq.astype(np.float64)

    approx_l2 = np.empty((nq, k_use), np.float64)
    true_l2   = np.empty((nq, k_use), np.float64)

    for j in range(k_use):
        a = xq64 - xb[I[:, j]].astype(np.float64)
        approx_l2[:, j] = np.sqrt((a * a).sum(axis=1))

        t = xq64 - xb[gt[:, j]].astype(np.float64)
        true_l2[:, j]   = np.sqrt((t * t).sum(axis=1))

    mask   = true_l2 > 0
    ratios = np.where(mask, approx_l2 / np.where(mask, true_l2, 1.0), 1.0)
    return float(ratios.mean()) - 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(I: np.ndarray, gt: np.ndarray, k: int) -> float:
    hits = (I[:, :k] == gt[:, :1]).any(axis=1).sum()
    return float(hits) / I.shape[0]


def fmt_time(s: float) -> str:
    return f"{s/60:.1f}min" if s >= 60 else f"{s:.2f}s"


ADD_BATCH_SIZE = 100_000

def add_batched(index, xb: np.ndarray, verbose: bool = False) -> float:
    """Add xb to index in batches. Returns wall-clock seconds."""
    nb = len(xb)
    t0 = time.perf_counter()
    for i0 in range(0, nb, ADD_BATCH_SIZE):
        batch = np.ascontiguousarray(xb[i0:i0 + ADD_BATCH_SIZE], dtype="float32")
        index.add(batch)
        if verbose:
            done = min(i0 + ADD_BATCH_SIZE, nb)
            print(f"\r    {done:>10,} / {nb:,}  ({done/nb*100:.1f}%)",
                  end="", flush=True)
    if verbose:
        print()
    return time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
# SIFT10M loader  (HDF5 / mat v7.3)
# ─────────────────────────────────────────────────────────────────────────────

def load_sift10m(mat_path: str, nb: int, nq: int):
    """Return (xb, xq) as float32 arrays from SIFT10Mfeatures.mat."""
    try:
        import h5py
    except ImportError:
        sys.exit("h5py is required for SIFT10M.  Install: pip install h5py")

    if not os.path.exists(mat_path):
        sys.exit(f"Missing file: {mat_path}")

    with h5py.File(mat_path, "r") as f:
        fea      = f["fea"]
        total_n  = fea.shape[0]
        d        = fea.shape[1]
        if nb + nq > total_n:
            sys.exit(f"Requested nb={nb:,}+nq={nq:,}={nb+nq:,} but file has {total_n:,}.")
        print(f"  Loading base  [{0:,} .. {nb:,}) …", end=" ", flush=True)
        xb = fea[:nb, :].astype(np.float32)
        print(f"shape={xb.shape}")
        print(f"  Loading query [{nb:,} .. {nb+nq:,}) …", end=" ", flush=True)
        xq = fea[nb:nb + nq, :].astype(np.float32)
        print(f"shape={xq.shape}")

    return xb, xq


def compute_gt(xb: np.ndarray, xq: np.ndarray, k: int = 100,
               batch: int = 1000) -> np.ndarray:
    """Exact k-NN ground truth via IndexFlatL2 (batched over queries)."""
    d  = xb.shape[1]
    nq = xq.shape[0]
    gt = np.empty((nq, k), dtype=np.int64)
    flat = faiss.IndexFlatL2(d)
    add_batched(flat, xb)
    for s in range(0, nq, batch):
        e = min(s + batch, nq)
        _, gt[s:e] = flat.search(xq[s:e], k)
    return gt


# ─────────────────────────────────────────────────────────────────────────────
# LSH nbits sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep_lsh(
    xb: np.ndarray,
    xq: np.ndarray,
    gt: np.ndarray,
    *,
    k: int,
    nbits_list: list[int],
    index_prefix: str = "",
) -> list[dict]:
    """
    Build one IndexLSH per nbits value, search, and compute MRE.
    Returns a list of result dicts (one per nbits).
    """
    d  = xb.shape[1]
    nq = xq.shape[0]

    print(f"\n  {'nbits':>6}  {'MRE':>8}  {'Recall@1':>9}  {'R@10':>7}  "
          f"{'QPS':>8}  {'build':>7}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*8}  {'-'*7}")

    results = []
    for nbits in nbits_list:
        idx_path = (f"{index_prefix}_lsh_nbits{nbits}.idx"
                    if index_prefix else "")

        if idx_path and os.path.exists(idx_path):
            index   = faiss.read_index(idx_path)
            t_build = 0.0
        else:
            index = faiss.IndexLSH(d, nbits)
            index.train(xb[:min(len(xb), 100_000)])   # no-op for default rotate_data=False
            t0      = time.perf_counter()
            add_batched(index, xb)
            t_build = time.perf_counter() - t0
            if idx_path:
                faiss.write_index(index, idx_path)

        # warm-up + search
        index.search(xq[:1], k)
        t0 = time.perf_counter()
        _, I = index.search(xq, k)
        elapsed = time.perf_counter() - t0
        qps = nq / elapsed

        m    = mre(xb, xq, I, gt, k)
        r1   = recall_at_k(I, gt, 1)
        r10  = recall_at_k(I, gt, min(10, k, gt.shape[1]))

        print(f"  {nbits:>6}  {m:8.4f}  {r1:9.4f}  {r10:7.4f}  "
              f"{qps:8.0f}  {fmt_time(t_build):>7}", flush=True)

        results.append(dict(nbits=nbits, mre=m,
                            recall1=r1, recall10=r10, qps=qps))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

# Paper-default nbits ranges per dataset
_DEEP_NBITS = [96, 192, 384, 768, 1536]
_SIFT_NBITS = [128, 256, 512, 1024]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MRE benchmark for IndexLSH on Deep1M and SIFT10M.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data-dir", default="/Users/dhm/Documents/data/",
                   help="Root dir with Deep1B data (for Deep1M).")
    p.add_argument("--index-dir", default="/Users/dhm/Documents/indices/",
                   help="Directory with pre-built LSH index files and GT.")
    p.add_argument("--mat-path", default="/Users/dhm/Documents/data/SIFT10M/SIFT10Mfeatures.mat",
                   help="Path to SIFT10Mfeatures.mat.")

    # Dataset skips
    p.add_argument("--skip-deep", action="store_true", help="Skip Deep1M.")
    p.add_argument("--skip-sift", action="store_true", help="Skip SIFT10M.")

    # SIFT10M sizes
    p.add_argument("--nb", type=int, default=10_000_000,
                   help="Number of base vectors for SIFT10M.")
    p.add_argument("--nq", type=int, default=10_000,
                   help="Number of query vectors for SIFT10M.")

    # Sweep config
    p.add_argument("--deep-nbits", type=int, nargs="+", default=_DEEP_NBITS,
                   help="nbits values to sweep on Deep1M.")
    p.add_argument("--sift-nbits", type=int, nargs="+", default=_SIFT_NBITS,
                   help="nbits values to sweep on SIFT10M.")

    p.add_argument("--k", type=int, default=10, help="Top-k neighbours.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_dataset_basedir(args.data_dir)

    # Derive paths from --index-dir (matches the naming used by the other bench scripts):
    #   {index-dir}/deep_nb1000000_lsh_nbits{N}.idx
    #   {index-dir}/sift10m_lsh_nbits{N}.idx
    #   {index-dir}/sift10m_gt.npy
    idir            = args.index_dir.rstrip("/")
    deep_nb         = 10 ** 6        # bench_mre always uses Deep1M (nb=1M)
    deep_idx_prefix = f"{idir}/deep_nb{deep_nb}"
    sift_idx_prefix = f"{idir}/sift10m"
    sift_gt_path    = f"{idir}/sift10m_gt.npy"

    all_results: list[dict] = []
    bar = "=" * 65

    # ── Deep1M ────────────────────────────────────────────────────────────
    if not args.skip_deep:
        print(f"\n{bar}\n  Deep1M  (d=96, nb=1M, nq=10K)\n{bar}")
        ds = DatasetDeep1B(nb=deep_nb)
        xb = ds.get_database()
        xq = ds.get_queries()
        gt = ds.get_groundtruth(k=100)
        print(f"  base={xb.shape}  queries={xq.shape}  gt={gt.shape}")

        rows = sweep_lsh(
            xb, xq, gt,
            k=args.k,
            nbits_list=args.deep_nbits,
            index_prefix=deep_idx_prefix,
        )
        for r in rows:
            r["dataset"] = "Deep1M"
        all_results.extend(rows)
        del xb

    # ── SIFT10M ───────────────────────────────────────────────────────────
    if not args.skip_sift:
        if not args.mat_path:
            print("\n  SIFT10M skipped — pass --mat-path /path/to/SIFT10Mfeatures.mat")
        else:
            print(f"\n{bar}\n  SIFT10M  (d=128, nb={args.nb:,}, nq={args.nq:,})\n{bar}")
            xb, xq = load_sift10m(args.mat_path, args.nb, args.nq)

            if os.path.exists(sift_gt_path):
                print(f"  Loading GT from {sift_gt_path} …", flush=True)
                gt = np.load(sift_gt_path)
            else:
                print(f"  GT not found at {sift_gt_path} — recomputing …", flush=True)
                t0 = time.perf_counter()
                gt = compute_gt(xb, xq, k=100)
                print(f"  GT done in {fmt_time(time.perf_counter() - t0)}")
                np.save(sift_gt_path, gt)
                print(f"  GT saved → {sift_gt_path}")

            rows = sweep_lsh(
                xb, xq, gt,
                k=args.k,
                nbits_list=args.sift_nbits,
                index_prefix=sift_idx_prefix,
            )
            for r in rows:
                r["dataset"] = "SIFT10M"
            all_results.extend(rows)
            del xb

    if not all_results:
        print("\nNo results — all datasets skipped.")
        return

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{bar}\n  SUMMARY  (LSH MRE)\n{bar}")
    print(f"\n  {'Dataset':<8}  {'nbits':>6}  {'MRE':>8}  {'Recall@1':>9}  "
          f"{'R@10':>7}  {'QPS':>8}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*8}")
    for r in all_results:
        print(f"  {r['dataset']:<8}  {r['nbits']:>6}  {r['mre']:8.4f}  "
              f"{r['recall1']:9.4f}  {r['recall10']:7.4f}  {r['qps']:8.0f}")


if __name__ == "__main__":
    main()
