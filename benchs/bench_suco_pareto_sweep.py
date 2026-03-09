#!/usr/bin/env python3
"""
benchs/bench_suco_pareto_sweep.py

Draws full Recall–QPS Pareto frontier curves for SuCo alongside HNSW and
IVFFlat on SIFT1M, GIST1M, and Deep1M.

SuCo's index structure (centroids + IMI) is built (or loaded) once.
Only `collision_ratio` (α) and `candidate_ratio` (β) are mutated between
search calls — no rebuilding required.

Usage
-----
# Run all three datasets (requires pre-built index files):
    python benchs/bench_suco_pareto_sweep.py \\
        --data-dir /path/to/data/ \\
        --sift-index  /path/to/sift1m.idx \\
        --gist-index  /path/to/gist1m.idx \\
        --deep-index  /path/to/deep1m.idx

# Run a single dataset and skip the others:
    python benchs/bench_suco_pareto_sweep.py --data-dir /path/to/data/ \\
        --sift-index /path/to/sift1m.idx --skip-gist --skip-deep

# Let the script build the SuCo indices from scratch (omit --*-index flags):
    python benchs/bench_suco_pareto_sweep.py --data-dir /path/to/data/

Output
------
  benchs/pareto_sweep.png   — one 3-panel figure saved to disk
  benchs/pareto_sweep.tsv   — raw numbers for every (dataset, method, param) point
"""

import argparse
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Matplotlib (non-interactive backend — safe on headless machines)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# FAISS
# ---------------------------------------------------------------------------
try:
    import faiss
    from faiss.contrib.datasets import (
        DatasetSIFT1M,
        DatasetGIST1M,
        DatasetDeep1B,
        set_dataset_basedir,
    )
except ImportError as e:
    sys.exit(f"Cannot import faiss: {e}\n"
             "Build FAISS with IndexSuCo and install the Python bindings.")


# ============================================================================
# Shared helpers
# ============================================================================

def recall_at_k(I: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Fraction of queries whose true NN is in the top-k results."""
    hits = (I[:, :k] == gt[:, :1]).any(axis=1).sum()
    return float(hits) / I.shape[0]


def fmt_time(s: float) -> str:
    return f"{s/60:.1f}min" if s >= 60 else f"{s:.2f}s"


def print_section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def warmup_and_time(index, xq: np.ndarray, k: int):
    """One warm-up search, then a timed full search. Returns (I, qps)."""
    index.search(xq[:1], k)
    t0 = time.perf_counter()
    _, I = index.search(xq, k)
    elapsed = time.perf_counter() - t0
    return I, xq.shape[0] / elapsed


# ============================================================================
# α / β sweep grid
# ============================================================================

# We sweep α (collision_ratio) finely.  β = α / 10 follows the paper's
# convention (β ≈ α / 10).  You can override BETA_FACTORS below to explore
# the β dimension independently.
ALPHA_VALUES = [0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05,
                0.07, 0.10, 0.15, 0.20, 0.30]
BETA_FACTOR  = 0.10   # β = α * BETA_FACTOR  (matches paper default at α=0.05)


# ============================================================================
# SuCo Pareto sweep  (search-time only — index built once)
# ============================================================================

def load_or_build_suco(
    xb: np.ndarray,
    xt: np.ndarray,
    *,
    d: int,
    nsubspaces: int,
    ncentroids_half: int = 50,
    niter: int = 10,
    index_path: str = "",
) -> "faiss.IndexSuCo":
    """
    Return a trained + populated IndexSuCo.
    If index_path exists it is loaded (fast); otherwise train+add+optionally save.
    collision_ratio / candidate_ratio are set to their defaults here but will
    be overwritten during the sweep.
    """
    index = faiss.IndexSuCo(d, nsubspaces, ncentroids_half, 0.05, 0.005, niter)
    index.verbose = False

    if index_path and os.path.exists(index_path):
        print(f"    Loading SuCo index from {index_path} …", flush=True)
        t0 = time.perf_counter()
        index.read_index(index_path)
        print(f"    Loaded in {fmt_time(time.perf_counter() - t0)}")
    else:
        print(f"    Training SuCo on {len(xt):,} vectors …", flush=True)
        t0 = time.perf_counter()
        index.train(xt)
        print(f"    Trained  in {fmt_time(time.perf_counter() - t0)}")

        print(f"    Adding   {len(xb):,} vectors …", flush=True)
        t0 = time.perf_counter()
        index.add(xb)
        print(f"    Added    in {fmt_time(time.perf_counter() - t0)}")

        if index_path:
            index.write_index(index_path)
            print(f"    Saved  → {index_path}")

    return index


def sweep_suco(
    index: "faiss.IndexSuCo",
    xq: np.ndarray,
    gt: np.ndarray,
    *,
    k: int = 10,
    alpha_values=ALPHA_VALUES,
    beta_factor: float = BETA_FACTOR,
) -> list[dict]:
    """
    Mutate index.collision_ratio / index.candidate_ratio across alpha_values,
    measure QPS and Recall@1 for each, and return a list of result dicts.
    No index rebuild — only search parameters change.
    """
    results = []
    print(f"\n    {'alpha':>7}  {'beta':>7}  {'QPS':>8}  {'R@1':>7}  {'R@10':>7}")
    print(f"    {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*7}")

    for alpha in alpha_values:
        beta = alpha * beta_factor
        index.collision_ratio = alpha
        index.candidate_ratio = beta

        I, qps = warmup_and_time(index, xq, k)
        r1  = recall_at_k(I, gt, 1)
        r10 = recall_at_k(I, gt, min(10, k))

        print(f"    {alpha:7.4f}  {beta:7.4f}  {qps:8.0f}  {r1:7.4f}  {r10:7.4f}",
              flush=True)
        results.append(dict(
            method="SuCo",
            alpha=alpha, beta=beta,
            qps=qps, recall1=r1, recall10=r10,
        ))

    return results


# ============================================================================
# HNSW sweep
# ============================================================================

EF_SEARCH_VALUES = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def build_hnsw(xb: np.ndarray, *, M: int = 32, ef_construction: int = 200) -> "faiss.IndexHNSWFlat":
    d = xb.shape[1]
    print(f"    Building HNSW (M={M}, efConstruction={ef_construction}) "
          f"on {len(xb):,} vectors …", flush=True)
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    t0 = time.perf_counter()
    index.add(xb)
    print(f"    Built   in {fmt_time(time.perf_counter() - t0)}")
    return index


def sweep_hnsw(
    index: "faiss.IndexHNSWFlat",
    xq: np.ndarray,
    gt: np.ndarray,
    *,
    k: int = 10,
    ef_values=EF_SEARCH_VALUES,
) -> list[dict]:
    results = []
    print(f"\n    {'efSearch':>10}  {'QPS':>8}  {'R@1':>7}  {'R@10':>7}")
    print(f"    {'-'*10}  {'-'*8}  {'-'*7}  {'-'*7}")

    for ef in ef_values:
        index.hnsw.efSearch = ef
        I, qps = warmup_and_time(index, xq, k)
        r1  = recall_at_k(I, gt, 1)
        r10 = recall_at_k(I, gt, min(10, k))
        print(f"    {ef:>10}  {qps:8.0f}  {r1:7.4f}  {r10:7.4f}", flush=True)
        results.append(dict(
            method="HNSW", param=ef,
            qps=qps, recall1=r1, recall10=r10,
        ))

    return results


# ============================================================================
# IVFFlat sweep
# ============================================================================

NPROBE_VALUES = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def build_ivfflat(xb: np.ndarray, xt: np.ndarray, *, nlist: int = 1024) -> "faiss.IndexIVFFlat":
    d = xb.shape[1]
    print(f"    Building IVFFlat (nlist={nlist}) on {len(xb):,} vectors …", flush=True)
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    t0 = time.perf_counter()
    index.train(xt)
    index.add(xb)
    print(f"    Built   in {fmt_time(time.perf_counter() - t0)}")
    return index


def sweep_ivfflat(
    index: "faiss.IndexIVFFlat",
    xq: np.ndarray,
    gt: np.ndarray,
    *,
    k: int = 10,
    nprobe_values=NPROBE_VALUES,
) -> list[dict]:
    results = []
    print(f"\n    {'nprobe':>8}  {'QPS':>8}  {'R@1':>7}  {'R@10':>7}")
    print(f"    {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}")

    for nprobe in nprobe_values:
        index.nprobe = nprobe
        I, qps = warmup_and_time(index, xq, k)
        r1  = recall_at_k(I, gt, 1)
        r10 = recall_at_k(I, gt, min(10, k))
        print(f"    {nprobe:>8}  {qps:8.0f}  {r1:7.4f}  {r10:7.4f}", flush=True)
        results.append(dict(
            method="IVFFlat", param=nprobe,
            qps=qps, recall1=r1, recall10=r10,
        ))

    return results


# ============================================================================
# IVFPQ sweep
# ============================================================================

def build_ivfpq(
    xb: np.ndarray,
    xt: np.ndarray,
    *,
    nlist: int = 1024,
    M: int = 8,
    nbits: int = 8,
) -> "faiss.IndexIVFPQ":
    """Build an IVFPQ index.  M sub-quantisers, nbits bits each."""
    d = xb.shape[1]
    assert d % M == 0, f"d ({d}) must be divisible by M ({M})"
    print(f"    Building IVFPQ (nlist={nlist}, M={M}, nbits={nbits}) "
          f"on {len(xb):,} vectors …", flush=True)
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
    t0 = time.perf_counter()
    index.train(xt)
    index.add(xb)
    print(f"    Built   in {fmt_time(time.perf_counter() - t0)}")
    return index


def sweep_ivfpq(
    index: "faiss.IndexIVFPQ",
    xq: np.ndarray,
    gt: np.ndarray,
    *,
    k: int = 10,
    nprobe_values=NPROBE_VALUES,
) -> list[dict]:
    results = []
    print(f"\n    {'nprobe':>8}  {'QPS':>8}  {'R@1':>7}  {'R@10':>7}")
    print(f"    {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}")

    for nprobe in nprobe_values:
        index.nprobe = nprobe
        I, qps = warmup_and_time(index, xq, k)
        r1  = recall_at_k(I, gt, 1)
        r10 = recall_at_k(I, gt, min(10, k))
        print(f"    {nprobe:>8}  {qps:8.0f}  {r1:7.4f}  {r10:7.4f}", flush=True)
        results.append(dict(
            method="IVFPQ", param=nprobe,
            qps=qps, recall1=r1, recall10=r10,
        ))

    return results


# ============================================================================
# Per-dataset runner
# ============================================================================

def run_dataset(
    name: str,
    xb: np.ndarray,
    xq: np.ndarray,
    xt: np.ndarray,
    gt: np.ndarray,
    *,
    nsubspaces: int,
    nlist: int = 1024,
    hnsw_M: int = 32,
    k: int = 10,
    suco_index_path: str = "",
    alpha_values=ALPHA_VALUES,
    skip_hnsw: bool = False,
    skip_ivfflat: bool = False,
) -> list[dict]:
    """
    Build / load all three indices for one dataset, run all sweeps, and return
    the combined results list (each entry tagged with `dataset` and `method`).
    """
    print_section(f"Dataset: {name}  (d={xb.shape[1]}, nb={xb.shape[0]:,}, nq={xq.shape[0]:,})")
    all_results: list[dict] = []

    # ── SuCo ──────────────────────────────────────────────────────────────
    print("\n  [SuCo] α-sweep (index loaded once, only search params change)")
    suco = load_or_build_suco(
        xb, xt,
        d=xb.shape[1],
        nsubspaces=nsubspaces,
        index_path=suco_index_path,
    )
    suco_pts = sweep_suco(suco, xq, gt, k=k, alpha_values=alpha_values)
    for pt in suco_pts:
        pt["dataset"] = name
    all_results.extend(suco_pts)
    del suco   # free RAM — raw vectors are stored inside IndexSuCo

    # ── HNSW ──────────────────────────────────────────────────────────────
    if not skip_hnsw:
        print("\n  [HNSW] efSearch sweep")
        hnsw = build_hnsw(xb, M=hnsw_M)
        hnsw_pts = sweep_hnsw(hnsw, xq, gt, k=k)
        for pt in hnsw_pts:
            pt["dataset"] = name
        all_results.extend(hnsw_pts)
        del hnsw

    # ── IVFFlat ──────────────────────────────────────────────────────────
    if not skip_ivfflat:
        print("\n  [IVFFlat] nprobe sweep")
        ivf = build_ivfflat(xb, xt, nlist=nlist)
        ivf_pts = sweep_ivfflat(ivf, xq, gt, k=k)
        for pt in ivf_pts:
            pt["dataset"] = name
        all_results.extend(ivf_pts)
        del ivf

    return all_results


# ============================================================================
# Pareto helpers
# ============================================================================

def pareto_frontier(pts: list[dict], recall_key="recall1") -> list[dict]:
    """
    Keep only Pareto-optimal points: for each unique recall level, keep the
    point with the highest QPS; then filter so that QPS is non-decreasing as
    recall decreases  (i.e. the lower-left envelope).
    """
    sorted_pts = sorted(pts, key=lambda p: p[recall_key])
    frontier = []
    best_qps = -1.0
    for pt in sorted_pts:
        if pt["qps"] > best_qps:
            frontier.append(pt)
            best_qps = pt["qps"]
    return frontier


# ============================================================================
# Plotting
# ============================================================================

STYLE = {
    "SuCo":    dict(color="#f4a261", marker="o", lw=2.5, ms=7,  zorder=6,
                    label="SuCo  (α sweep)"),
    "HNSW":    dict(color="#e63946", marker="^", lw=2.0, ms=6,  zorder=5,
                    label="HNSW  (efSearch sweep)"),
    "IVFFlat": dict(color="#457b9d", marker="s", lw=2.0, ms=6,  zorder=4,
                    label="IVFFlat  (nprobe sweep)"),
    "IVFPQ":   dict(color="#2a9d8f", marker="D", lw=2.0, ms=6,  zorder=3,
                    label="IVFPQ  (nprobe sweep)"),
}


def plot_pareto(
    ax,
    results: list[dict],
    dataset_name: str,
    recall_key: str = "recall1",
    x_label: str = "Recall@1",
    x_lo: float = 0.3,
) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlim(x_lo, 1.0)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("QPS  (log scale)", fontsize=12)
    ax.set_title(dataset_name, fontsize=13, fontweight="bold")

    for method, style in STYLE.items():
        pts = [p for p in results if p["method"] == method]
        if not pts:
            continue
        frontier = pareto_frontier(pts, recall_key=recall_key)
        r = [p[recall_key] for p in frontier]
        q = [p["qps"]      for p in frontier]
        ax.plot(r, q,
                color=style["color"], marker=style["marker"],
                linewidth=style["lw"], markersize=style["ms"],
                zorder=style["zorder"], label=style["label"],
                alpha=0.92)

    ax.legend(fontsize=9, loc="lower right")


# ============================================================================
# TSV export
# ============================================================================

def write_tsv(results: list[dict], path: str) -> None:
    import csv
    fieldnames = ["dataset", "method", "alpha", "beta", "param",
                  "qps", "recall1", "recall10"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"  TSV  → {path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SuCo vs HNSW vs IVFFlat: full Recall–QPS Pareto sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="data/",
                   help="Root directory with SIFT1M / GIST1M / Deep1B data.")

    # Per-dataset index paths (optional — build from scratch if omitted)
    p.add_argument("--sift-index", default="",
                   help="Path to a pre-built SIFT1M SuCo index file.")
    p.add_argument("--gist-index", default="",
                   help="Path to a pre-built GIST1M SuCo index file.")
    p.add_argument("--deep-index", default="",
                   help="Path to a pre-built Deep1M  SuCo index file.")

    # Dataset skipping
    p.add_argument("--skip-sift", action="store_true")
    p.add_argument("--skip-gist", action="store_true")
    p.add_argument("--skip-deep", action="store_true")

    # Baseline skipping
    p.add_argument("--skip-hnsw",    action="store_true",
                   help="Skip HNSW sweep (saves build time).")
    p.add_argument("--skip-ivfflat", action="store_true",
                   help="Skip IVFFlat sweep.")
    p.add_argument("--skip-ivfpq",   action="store_true",
                   help="Skip IVFPQ sweep.")

    # Sweep configuration
    p.add_argument("--alpha-values", type=float, nargs="+",
                   default=ALPHA_VALUES,
                   help="α values to sweep.  β = α × --beta-factor.")
    p.add_argument("--beta-factor", type=float, default=BETA_FACTOR,
                   help="β = α × beta_factor.")
    p.add_argument("--k", type=int, default=10,
                   help="Number of nearest neighbours to retrieve.")

    # Output
    p.add_argument("--out-plot", default="benchs/pareto_sweep.png")
    p.add_argument("--out-tsv",  default="benchs/pareto_sweep.tsv")

    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()
    set_dataset_basedir(args.data_dir)

    all_results: list[dict] = []

    # ── SIFT1M (d=128) ──────────────────────────────────────────────────────
    if not args.skip_sift:
        print_section("Loading SIFT1M")
        ds = DatasetSIFT1M()
        xb = ds.get_database()
        xq = ds.get_queries()
        xt = ds.get_train(maxtrain=100_000)
        gt = ds.get_groundtruth(k=100)
        print(f"  base={xb.shape}  queries={xq.shape}  train={xt.shape}  gt={gt.shape}")

        all_results += run_dataset(
            "SIFT1M", xb, xq, xt, gt,
            nsubspaces=8,
            suco_index_path=args.sift_index,
            alpha_values=args.alpha_values,
            skip_hnsw=args.skip_hnsw,
            skip_ivfflat=args.skip_ivfflat,
            skip_ivfpq=args.skip_ivfpq,
            k=args.k,
        )
        del xb, xt  # free RAM (xq and gt still needed for plot labels, but we're done)

    # ── GIST1M (d=960) ──────────────────────────────────────────────────────
    if not args.skip_gist:
        print_section("Loading GIST1M")
        ds = DatasetGIST1M()
        xb = ds.get_database()
        xq = ds.get_queries()
        xt = ds.get_train(maxtrain=100_000)
        gt = ds.get_groundtruth(k=100)
        print(f"  base={xb.shape}  queries={xq.shape}  train={xt.shape}  gt={gt.shape}")

        all_results += run_dataset(
            "GIST1M", xb, xq, xt, gt,
            nsubspaces=40,
            suco_index_path=args.gist_index,
            alpha_values=args.alpha_values,
            skip_hnsw=args.skip_hnsw,
            skip_ivfflat=args.skip_ivfflat,
            skip_ivfpq=args.skip_ivfpq,
            k=args.k,
        )
        del xb, xt

    # ── Deep1M (d=96) ────────────────────────────────────────────────────────
    if not args.skip_deep:
        print_section("Loading Deep1M")
        ds = DatasetDeep1B(nb=10**6)
        xb = ds.get_database()
        xq = ds.get_queries()
        xt = ds.get_train(maxtrain=500_000)
        gt = ds.get_groundtruth(k=100)
        print(f"  base={xb.shape}  queries={xq.shape}  train={xt.shape}  gt={gt.shape}")

        all_results += run_dataset(
            "Deep1M", xb, xq, xt, gt,
            nsubspaces=8,
            suco_index_path=args.deep_index,
            alpha_values=args.alpha_values,
            skip_hnsw=args.skip_hnsw,
            skip_ivfflat=args.skip_ivfflat,
            skip_ivfpq=args.skip_ivfpq,
            k=args.k,
        )
        del xb, xt

    if not all_results:
        print("No results collected — all datasets were skipped.")
        return

    # ── TSV export ────────────────────────────────────────────────────────
    write_tsv(all_results, args.out_tsv)

    # ── Plot ─────────────────────────────────────────────────────────────
    datasets_present = list(dict.fromkeys(r["dataset"] for r in all_results))
    ncols = len(datasets_present)

    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6),
                             facecolor="#f8f9fa")
    if ncols == 1:
        axes = [axes]

    x_lo_map = {"SIFT1M": 0.35, "GIST1M": 0.15, "Deep1M": 0.40}
    d_map    = {"SIFT1M": "d=128, nb=1M, nq=10K",
                "GIST1M": "d=960, nb=1M, nq=1K",
                "Deep1M": "d=96, nb=1M, nq=10K"}

    for ax, ds_name in zip(axes, datasets_present):
        pts = [r for r in all_results if r["dataset"] == ds_name]
        title = f"{ds_name}\n({d_map.get(ds_name, '')})"
        plot_pareto(ax, pts, title,
                    x_lo=x_lo_map.get(ds_name, 0.3))

    fig.suptitle(
        "SuCo vs HNSW vs IVFFlat — Recall@1 / QPS Pareto Frontier\n"
        "SuCo curve: α sweep with β = α × 0.1  |  index built once",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=140, bbox_inches="tight")
    print(f"\n  Plot → {args.out_plot}")


if __name__ == "__main__":
    main()
