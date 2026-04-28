#!/usr/bin/env python3
"""
benchs/plot_cspg_paper.py

Generate paper-style plots from results produced by bench_cspg_paper.py.

Figures produced:
  fig_construction_time.{pdf,png}   — Bar chart: build time per index (Table 1 style)
  fig_construction_mem.{pdf,png}    — Bar chart: index memory per index
  fig_recall_k{K}.{pdf,png}         — Recall@K vs ms/query, one panel per dataset
  fig_ablation_m.{pdf,png}          — Recall@10 vs time, varying m
  fig_ablation_lam.{pdf,png}        — Recall@10 vs time, varying λ
  fig_ablation_ef1.{pdf,png}        — Recall@10 vs time, varying ef1
  fig_robustness.{pdf,png}          — Boxplots of per-query recall

Usage:
  python benchs/plot_cspg_paper.py --results-dir benchs/results_cspg/
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style config
# ---------------------------------------------------------------------------

DATASETS_ORDER = ["sift1m", "deep1m", "gist1m", "uqv1m", "openai1m", "sift10m"]
DATASET_LABELS = {
    "sift1m":     "SIFT1M",
    "deep1m":     "Deep1M",
    "gist1m":     "GIST1M",
    "uqv1m":      "UQV1M",
    "openai1m":   "OpenAI1M",
    "sift10m":    "SIFT10M",
}

# Colour / marker registry — keys match label strings produced by bench script
COLORS = {
    "CSPG":   "#e74c3c",
    "HNSW":   "#3498db",
    "IVFFlat": "#2ecc71",
    "IVFPQ":  "#f39c12",
    "Flat":   "#9b59b6",
}
MARKERS = {
    "CSPG":   "o",
    "HNSW":   "s",
    "IVFFlat": "D",
    "IVFPQ":  "^",
    "Flat":   "*",
}

# Ablation colour gradients
ABLATION_CMAP = plt.cm.tab10


def _color(label):
    for k, c in COLORS.items():
        if label.startswith(k):
            return c
    return "#555555"


def _marker(label):
    for k, m in MARKERS.items():
        if label.startswith(k):
            return m
    return "o"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(results_dir):
    results = {}
    for fname in os.listdir(results_dir):
        if fname.startswith("results_cspg_") and fname.endswith(".json"):
            ds = fname[len("results_cspg_"):-len(".json")]
            with open(os.path.join(results_dir, fname)) as f:
                results[ds] = json.load(f)
    return results


def _save(fig, output_dir, stem):
    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {stem}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Construction time & memory (bar charts)
# ---------------------------------------------------------------------------

def plot_construction(results, output_dir):
    datasets = [ds for ds in DATASETS_ORDER if ds in results
                and "construction" in results[ds]]
    if not datasets:
        print("  No construction data found — skipping.")
        return

    # Collect all index labels across all datasets
    all_labels = []
    for ds in datasets:
        for lbl in results[ds]["construction"]:
            if lbl not in all_labels:
                all_labels.append(lbl)

    x     = np.arange(len(datasets))
    width = 0.8 / max(len(all_labels), 1)

    for metric, ylabel, stem in [
        ("build_time_s", "Construction Time (s)", "fig_construction_time"),
        ("memory_mb",    "Memory (MB)",            "fig_construction_mem"),
    ]:
        fig, ax = plt.subplots(figsize=(max(10, 2 * len(datasets)), 5))
        for i, lbl in enumerate(all_labels):
            vals = []
            for ds in datasets:
                v = results[ds]["construction"].get(lbl, {}).get(metric, -1)
                vals.append(max(v, 0.01))
            color = _color(lbl)
            ax.bar(x + i * width, vals, width, label=lbl, color=color, alpha=0.85)

        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x + width * (len(all_labels) - 1) / 2)
        ax.set_xticklabels([DATASET_LABELS.get(ds, ds) for ds in datasets], rotation=15)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(ylabel)
        fig.tight_layout()
        _save(fig, output_dir, stem)


# ---------------------------------------------------------------------------
# Figure 2-4: Recall vs Time  (one grid per k value)
# ---------------------------------------------------------------------------

def plot_recall_vs_time(results, output_dir, k):
    bm_key   = f"recall_k{k}"
    datasets = [ds for ds in DATASETS_ORDER if ds in results
                and bm_key in results[ds]]
    if not datasets:
        print(f"  No {bm_key} data — skipping.")
        return

    n_ds = len(datasets)
    cols = min(3, n_ds)
    rows = (n_ds + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, ds in enumerate(datasets):
        ax   = axes_flat[idx]
        data = results[ds][bm_key]
        for label, pts in data.items():
            if not pts:
                continue
            recalls = [p["recall"] for p in pts]
            times   = [p["ms_per_query"] for p in pts]
            ax.plot(times, recalls,
                    marker=_marker(label), color=_color(label),
                    label=label, markersize=5, linewidth=1.5)

        ax.set_xlabel("Time (ms/query)")
        ax.set_ylabel(f"Recall@{k}")
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

    for idx in range(n_ds, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Recall@{k} vs Query Time — CSPG vs Baselines", fontsize=13)
    fig.tight_layout()
    _save(fig, output_dir, f"fig_recall_k{k}")


# ---------------------------------------------------------------------------
# Figure 5: Ablation — varying m
# ---------------------------------------------------------------------------

def plot_ablation(results, output_dir, bm_key, title_suffix, stem):
    datasets = [ds for ds in DATASETS_ORDER if ds in results
                and bm_key in results[ds]]
    if not datasets:
        print(f"  No {bm_key} data — skipping.")
        return

    n_ds = len(datasets)
    cols = min(3, n_ds)
    rows = (n_ds + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, ds in enumerate(datasets):
        ax   = axes_flat[idx]
        data = results[ds][bm_key]
        labels = list(data.keys())
        cmap   = ABLATION_CMAP(np.linspace(0, 0.8, max(len(labels), 1)))

        for i, (label, pts) in enumerate(data.items()):
            if not pts:
                continue
            recalls = [p["recall"] for p in pts]
            times   = [p["ms_per_query"] for p in pts]
            ax.plot(times, recalls, marker="o", color=cmap[i],
                    label=label, markersize=5, linewidth=1.5)

        ax.set_xlabel("Time (ms/query)")
        ax.set_ylabel("Recall@10")
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.legend(fontsize=6, loc="lower right")
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

    for idx in range(n_ds, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Ablation: {title_suffix}", fontsize=13)
    fig.tight_layout()
    _save(fig, output_dir, stem)


# ---------------------------------------------------------------------------
# Figure 6: Robustness boxplots
# ---------------------------------------------------------------------------

def plot_robustness(results, output_dir):
    datasets = [ds for ds in DATASETS_ORDER if ds in results
                and "robustness" in results[ds]]
    if not datasets:
        print("  No robustness data — skipping.")
        return

    fig, ax = plt.subplots(figsize=(max(14, 3 * len(datasets)), 6))
    pos, labels, bxdata, colors = [], [], [], []
    p = 0

    for ds in datasets:
        rob = results[ds]["robustness"]
        ds_label = DATASET_LABELS.get(ds, ds)
        for lbl, stats in rob.items():
            bxdata.append({
                "med":    stats.get("median_recall", 0),
                "q1":     stats.get("q25_recall",    0),
                "q3":     stats.get("q75_recall",    0),
                "whislo": stats.get("min_recall",    0),
                "whishi": stats.get("max_recall",    0),
                "label":  lbl,
            })
            pos.append(p)
            labels.append(f"{ds_label}\n{lbl}")
            colors.append(_color(lbl))
            p += 1
        p += 1  # gap

    if bxdata:
        bxp = ax.bxp(bxdata, positions=pos, widths=0.6,
                     patch_artist=True, showfliers=False)
        for patch, color in zip(bxp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Recall@20")
    ax.set_ylim([0, 1.05])
    ax.set_title("Robustness: Per-Query Recall on Unseen Queries (k=20)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "fig_robustness")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot CSPG benchmark results")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir",  default=None)
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results_cspg")
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "plots")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from {args.results_dir}")
    results = load_results(args.results_dir)
    if not results:
        print("No results_cspg_*.json files found.")
        sys.exit(1)

    print(f"Found: {sorted(results.keys())}")
    print(f"Output: {args.output_dir}\n")

    print("Construction charts…")
    plot_construction(results, args.output_dir)

    for k in [10, 20, 50]:
        print(f"Recall@{k} curves…")
        plot_recall_vs_time(results, args.output_dir, k)

    print("Ablation: m…")
    plot_ablation(results, args.output_dir, "ablation_m",
                  "Varying num_partitions m", "fig_ablation_m")

    print("Ablation: λ…")
    plot_ablation(results, args.output_dir, "ablation_lam",
                  "Varying routing ratio λ", "fig_ablation_lam")

    print("Ablation: ef1…")
    plot_ablation(results, args.output_dir, "ablation_ef1",
                  "Varying first-stage ef1", "fig_ablation_ef1")

    print("Robustness boxplots…")
    plot_robustness(results, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()