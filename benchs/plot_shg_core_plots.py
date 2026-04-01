#!/usr/bin/env python3
"""
Generate SHG-focused paper plots from bench_shg_paper.py results.

This script produces:
    - fig4a_construction_time_shg_hnsw_panorama
    - fig4b_memory_cost_shg_hnsw_panorama
  - fig5_recall_vs_time_k20_shg_hnsw_panorama
  - fig6_recall_vs_time_k50_shg_hnsw_panorama
  - fig9_ablation_shg_hnsw_no_shortcut_no_lb

The recall-vs-time figures include only SHG, HNSW, and Panorama.
The ablation figure includes SHG, HNSW, SHG-no-shortcut, and SHG-no-lb.

Color palette is aligned with benchs/plot_benchmarks_from_logs.py:
  - HNSW: #e63946
  - SHG: #f4a261
  - Panorama: #6ab187
  - SHG-no-shortcut: #457b9d
  - SHG-no-lb: #a8dadc

Usage:
  python benchs/plot_shg_core_plots.py --results-dir benchs/results/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS_ORDER = ["openai", "enron", "gist1m", "msong", "uqv", "msturing10m"]
DATASET_LABELS = {
    "openai": "OpenAI",
    "enron": "Enron",
    "gist1m": "GIST1M",
    "msong": "Msong",
    "uqv": "UQ-V",
    "msturing10m": "MsTuring10M",
}

# Colors borrowed from benchs/plot_benchmarks_from_logs.py
COLORS = {
    "HNSW": "#a8dadc" ,
    "SHG": "#457b9d",
    "PANORAMA": "#6ab187",
    "SHG_NO_SHORTCUT": "#e63946",
    "SHG_NO_LB": "#f4a261",
}

MARKERS = {
    "SHG": "o",
    "HNSW": "^",
    "PANORAMA": "P",
    "SHG_NO_SHORTCUT": "s",
    "SHG_NO_LB": "D",
}


def load_results(results_dir: Path) -> dict[str, dict]:
    """Load all results_*.json files from a directory."""
    results: dict[str, dict] = {}
    for fname in os.listdir(results_dir):
        if fname.startswith("results_") and fname.endswith(".json"):
            ds_name = fname[len("results_") : -len(".json")]
            with open(results_dir / fname, "r", encoding="utf-8") as f:
                results[ds_name] = json.load(f)
    return results


def normalize_key(s: str) -> str:
    # Normalize case and separators to tolerate mixed key styles.
    return re.sub(r"[^a-z0-9]+", "-", s.strip().lower()).strip("-")


def find_series(data: dict, candidates: list[str]) -> list[dict]:
    """Resolve a series from potentially inconsistent key naming."""
    if not isinstance(data, dict):
        return []

    candidate_norm = {normalize_key(c) for c in candidates}
    key_map = {normalize_key(k): v for k, v in data.items()}

    for c in candidate_norm:
        v = key_map.get(c)
        if isinstance(v, list):
            return v

    # Fallback: match by token containment to handle minor naming drift.
    candidate_tokens = [set(c.split("-")) for c in candidate_norm]
    for key_norm, value in key_map.items():
        key_tokens = set(key_norm.split("-"))
        if any(tok.issubset(key_tokens) for tok in candidate_tokens):
            return value if isinstance(value, list) else []
    return []


def set_recall_ylim(ax: plt.Axes, recall_values: list[float]) -> None:
    """Set y-axis to start near the minimum recall instead of zero."""
    if not recall_values:
        ax.set_ylim([0.0, 1.05])
        return

    r_min = min(recall_values)
    r_max = max(recall_values)
    pad = 0.01
    y_low = max(0.0, r_min - pad)
    y_high = min(1.05, r_max + pad)
    if y_high <= y_low:
        y_high = min(1.05, y_low + 0.05)
    ax.set_ylim([y_low, y_high])


def plot_construction_core(results: dict[str, dict], output_dir: Path) -> None:
    """Plot Fig 4a/4b bars for SHG, HNSW, Panorama only."""
    datasets = [ds for ds in DATASETS_ORDER if ds in results]
    if not datasets:
        return

    indices = ["SHG", "HNSW", "Panorama"]
    x = np.arange(len(datasets))
    width = 0.24

    # Fig 4a: construction time
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, idx_name in enumerate(indices):
        vals = []
        for ds in datasets:
            r = results[ds].get("construction", {}).get(idx_name, {})
            v = r.get("build_time_s", 0)
            vals.append(v if v > 0 else 0.01)
        ax.bar(
            x + i * width,
            vals,
            width,
            label=idx_name,
            color=COLORS.get(idx_name.upper(), "#333333"),
            edgecolor="black",
            linewidth=0.6,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Construction Time (s)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([DATASET_LABELS.get(ds, ds) for ds in datasets], rotation=15)
    ax.set_title("Index Construction Time (SHG, HNSW, Panorama)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig4a_construction_time_shg_hnsw_panorama.pdf", dpi=150)
    fig.savefig(output_dir / "fig4a_construction_time_shg_hnsw_panorama.png", dpi=150)
    plt.close(fig)
    print("  Saved fig4a_construction_time_shg_hnsw_panorama")

    # Fig 4b: memory cost
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, idx_name in enumerate(indices):
        vals = []
        for ds in datasets:
            r = results[ds].get("construction", {}).get(idx_name, {})
            v = r.get("memory_mb", 0)
            vals.append(v if v > 0 else 0.01)
        ax.bar(
            x + i * width,
            vals,
            width,
            label=idx_name,
            color=COLORS.get(idx_name.upper(), "#333333"),
            edgecolor="black",
            linewidth=0.6,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Memory Cost (MB)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([DATASET_LABELS.get(ds, ds) for ds in datasets], rotation=15)
    ax.set_title("Index Memory Cost (SHG, HNSW, Panorama)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig4b_memory_cost_shg_hnsw_panorama.pdf", dpi=150)
    fig.savefig(output_dir / "fig4b_memory_cost_shg_hnsw_panorama.png", dpi=150)
    plt.close(fig)
    print("  Saved fig4b_memory_cost_shg_hnsw_panorama")


def plot_recall_vs_time_core(results: dict[str, dict], output_dir: Path, k_key: str, k_val: int) -> None:
    """Plot fig5/fig6 style curves with SHG, HNSW, and Panorama only."""
    datasets = [ds for ds in DATASETS_ORDER if ds in results and k_key in results[ds]]
    if not datasets:
        return

    n_ds = len(datasets)
    cols = min(4, n_ds)
    rows = (n_ds + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_ds == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    curve_spec = [
        ("SHG", ["SHG"]),
        ("HNSW", ["HNSW"]),
        ("Panorama", ["Panorama", "PANORAMA"]),
    ]

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        data = results[ds][k_key]
        plotted_recalls: list[float] = []

        for display_name, key_candidates in curve_spec:
            points = find_series(data, key_candidates)
            if not points:
                continue

            recalls = [p.get("recall", 0.0) for p in points]
            times = [p.get("ms_per_query", 0.0) for p in points]
            plotted_recalls.extend(recalls)
            palette_key = display_name.upper()
            ax.plot(
                times,
                recalls,
                marker=MARKERS.get(palette_key, "o"),
                color=COLORS.get(palette_key, "#333333"),
                label=display_name,
                markersize=5,
                linewidth=1.8,
            )

        ax.set_xlabel("Time (ms/query)")
        ax.set_ylabel(f"Recall@{k_val}")
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.legend(fontsize=8, loc="lower right")
        set_recall_ylim(ax, plotted_recalls)
        ax.grid(True, alpha=0.3)

    for idx in range(n_ds, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Recall vs Query Time (k={k_val})", fontsize=14)
    fig.tight_layout()

    fig_name = (
        "fig5_recall_vs_time_k20_shg_hnsw_panorama"
        if k_val == 20
        else "fig6_recall_vs_time_k50_shg_hnsw_panorama"
    )
    fig.savefig(output_dir / f"{fig_name}.pdf", dpi=150)
    fig.savefig(output_dir / f"{fig_name}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_name}")


def plot_ablation_core(results: dict[str, dict], output_dir: Path) -> None:
    """Plot ablation with SHG, HNSW, SHG-no-shortcut, and SHG-no-lb."""
    datasets = [
        ds
        for ds in DATASETS_ORDER
        if ds in results and ("ablation" in results[ds] or "recall_k20" in results[ds])
    ]
    if not datasets:
        return

    n_ds = len(datasets)
    cols = min(4, n_ds)
    rows = (n_ds + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_ds == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    ablation_spec = [
        ("SHG", ["SHG"], "SHG", "-"),
        ("HNSW", ["HNSW"], "HNSW", "-"),
        ("SHG (no shortcut)", ["SHG-no-shortcut", "SHG-NO-SHORTCUT"], "SHG_NO_SHORTCUT", "--"),
        ("SHG (no LB pruning)", ["SHG-no-lb", "SHG-NO-LB"], "SHG_NO_LB", "-.")
    ]

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        # Most benchmark outputs store variant runs under top-level "ablation".
        # Keep recall_k20 as a backward-compatible fallback.
        data = results[ds].get("ablation") or results[ds].get("recall_k20", {})
        plotted_recalls: list[float] = []

        for display_name, key_candidates, style_key, line_style in ablation_spec:
            points = find_series(data, key_candidates)
            if not points:
                continue

            recalls = [p.get("recall", 0.0) for p in points]
            times = [p.get("ms_per_query", 0.0) for p in points]
            plotted_recalls.extend(recalls)
            ax.plot(
                times,
                recalls,
                marker=MARKERS.get(style_key, "o"),
                color=COLORS.get(style_key, "#333333"),
                label=display_name,
                markersize=5.5,
                linewidth=2.0,
                linestyle=line_style,
                alpha=0.95,
            )

        ax.set_xlabel("Time (ms/query)")
        ax.set_ylabel("Recall@20")
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.legend(fontsize=7, loc="lower right")
        set_recall_ylim(ax, plotted_recalls)
        ax.grid(True, alpha=0.3)

    for idx in range(n_ds, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Ablation: SHG/HNSW and SHG Variants (k=20)", fontsize=14)
    fig.tight_layout()

    fig_name = "fig9_ablation_shg_hnsw_no_shortcut_no_lb"
    fig.savefig(output_dir / f"{fig_name}.pdf", dpi=150)
    fig.savefig(output_dir / f"{fig_name}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SHG core paper figures")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing results_*.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for generated plots (default: <results-dir>/plots)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.results_dir is None:
        results_dir = Path(__file__).resolve().parent / "results"
    else:
        results_dir = Path(args.results_dir)

    if args.output_dir is None:
        output_dir = results_dir / "plots"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir}")
    results = load_results(results_dir)
    if not results:
        print("No results found!")
        sys.exit(1)

    print(f"Found results for: {list(results.keys())}")
    print(f"Plots will be saved to {output_dir}\n")

    print("Plotting construction/memory bars (Fig 4a/4b): SHG, HNSW, Panorama...")
    plot_construction_core(results, output_dir)

    print("Plotting recall vs time (k=20): SHG, HNSW, Panorama...")
    plot_recall_vs_time_core(results, output_dir, "recall_k20", 20)

    print("Plotting recall vs time (k=50): SHG, HNSW, Panorama...")
    plot_recall_vs_time_core(results, output_dir, "recall_k50", 50)

    print("Plotting ablation: SHG, HNSW, SHG-no-shortcut, SHG-no-lb...")
    plot_ablation_core(results, output_dir)

    print(f"\nAll requested plots saved to {output_dir}")


if __name__ == "__main__":
    main()
