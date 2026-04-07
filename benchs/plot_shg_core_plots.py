#!/usr/bin/env python3
"""
Generate SHG-focused paper plots from bench_shg_paper.py results.

This script produces:
    - fig4a_construction_time_shg_hnsw_panorama
    - fig4b_memory_cost_shg_hnsw_panorama
  - fig5_recall_vs_time_k20_shg_hnsw_panorama
  - fig6_recall_vs_time_k50_shg_hnsw_panorama
  - fig9_ablation_shg_hnsw_no_shortcut_no_lb

The recall-vs-time figures include SHG, HNSW, Panorama, and IVFFlat.
The ablation figure includes SHG, HNSW, IVFFlat, SHG-no-shortcut, and SHG-no-lb.

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
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paper-quality global style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        # Paper-like typography and compact layout (style only).
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Linux Libertine O", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.fancybox": False,
        "legend.edgecolor": "0.65",
        "legend.borderpad": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.8,
        "patch.linewidth": 0.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "grid.linewidth": 0.4,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def _clean_ax(ax: plt.Axes) -> None:
    """Remove top/right spines, set subtle y-grid, put grid behind data."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.2")
    ax.spines["bottom"].set_color("0.2")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


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
    "HNSW": "#a8dadc",
    "SHG": "#457b9d",
    "PANORAMA": "#6ab187",
    "IVFFLAT": "#2f2f2f",
    "SHG_NO_SHORTCUT": "#e63946",
    "SHG_NO_LB": "#f4a261",
}

MARKERS = {
    "SHG": "o",
    "HNSW": "^",
    "PANORAMA": "P",
    "IVFFLAT": "X",
    "SHG_NO_SHORTCUT": "s",
    "SHG_NO_LB": "D",
}

CURVE_ZORDER = {
    "SHG": 10,
    "HNSW": 6,
    "PANORAMA": 5,
    "IVFFLAT": 4,
    "SHG_NO_SHORTCUT": 3,
    "SHG_NO_LB": 2,
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
    """Plot Fig 4a/4b bars for SHG, HNSW, Panorama, and IVFFlat."""
    datasets = [ds for ds in DATASETS_ORDER if ds in results]
    if not datasets:
        return

    indices = ["SHG", "HNSW", "Panorama", "IVFFlat"]
    x = np.arange(len(datasets))
    width = 0.18

    # Fig 4a: construction time
    fig, ax = plt.subplots(figsize=(7, 3))
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
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Construction time (s)")
    ax.set_xticks(x + width * (len(indices) - 1) / 2)
    ax.set_xticklabels([DATASET_LABELS.get(ds, ds) for ds in datasets], rotation=20, ha="right")
    ax.legend(loc="upper left", ncol=2)
    _clean_ax(ax)
    fig.savefig(output_dir / "fig4a_construction_time_shg_hnsw_panorama.pdf")
    fig.savefig(output_dir / "fig4a_construction_time_shg_hnsw_panorama.png")
    plt.close(fig)
    print("  Saved fig4a_construction_time_shg_hnsw_panorama")

    # Fig 4b: memory cost
    fig, ax = plt.subplots(figsize=(7, 3))
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
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Memory (MB)")
    ax.set_xticks(x + width * (len(indices) - 1) / 2)
    ax.set_xticklabels([DATASET_LABELS.get(ds, ds) for ds in datasets], rotation=20, ha="right")
    ax.legend(loc="upper left", ncol=2)
    _clean_ax(ax)
    fig.savefig(output_dir / "fig4b_memory_cost_shg_hnsw_panorama.pdf")
    fig.savefig(output_dir / "fig4b_memory_cost_shg_hnsw_panorama.png")
    plt.close(fig)
    print("  Saved fig4b_memory_cost_shg_hnsw_panorama")


def plot_recall_vs_time_core(results: dict[str, dict], output_dir: Path, k_key: str, k_val: int) -> None:
    """Plot fig5/fig6 style curves with SHG, HNSW, Panorama, and IVFFlat."""
    datasets = [ds for ds in DATASETS_ORDER if ds in results and k_key in results[ds]]
    if not datasets:
        return

    n_ds = len(datasets)
    cols = min(3, n_ds)
    rows = (n_ds + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.0 * rows), constrained_layout=True)
    if n_ds == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    curve_spec = [
        ("SHG", ["SHG"]),
        ("HNSW", ["HNSW"]),
        ("Panorama", ["Panorama", "PANORAMA"]),
        ("IVFFlat", ["IVFFlat", "IVF-Flat", "IVF_FLAT"]),
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
                zorder=CURVE_ZORDER.get(palette_key, 1),
            )

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("Time (ms/query)")
        ax.set_ylabel(f"Recall@{k_val}")
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.legend(loc="lower right")
        set_recall_ylim(ax, plotted_recalls)
        _clean_ax(ax)

    for idx in range(n_ds, len(axes)):
        axes[idx].set_visible(False)

    fig_name = (
        "fig5_recall_vs_time_k20_shg_hnsw_panorama"
        if k_val == 20
        else "fig6_recall_vs_time_k50_shg_hnsw_panorama"
    )
    fig.savefig(output_dir / f"{fig_name}.pdf")
    fig.savefig(output_dir / f"{fig_name}.png")
    plt.close(fig)
    print(f"  Saved {fig_name}")


def plot_ablation_core(results: dict[str, dict], output_dir: Path) -> None:
    """Plot ablation with SHG, HNSW, IVFFlat, SHG-no-shortcut, and SHG-no-lb."""
    datasets = [
        ds
        for ds in DATASETS_ORDER
        if ds in results and ("ablation" in results[ds] or "recall_k20" in results[ds])
    ]
    if not datasets:
        return

    n_ds = len(datasets)
    cols = min(3, n_ds)
    rows = (n_ds + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.0 * rows), constrained_layout=True)
    if n_ds == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    ablation_spec = [
        ("SHG", ["SHG"], "SHG", "-"),
        ("HNSW", ["HNSW"], "HNSW", "-"),
        ("IVFFlat", ["IVFFlat", "IVF-Flat", "IVF_FLAT"], "IVFFLAT", ":"),
        ("SHG (no shortcut)", ["SHG-no-shortcut", "SHG-NO-SHORTCUT"], "SHG_NO_SHORTCUT", "--"),
        ("SHG (no LB pruning)", ["SHG-no-lb", "SHG-NO-LB"], "SHG_NO_LB", "-."),
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
                linestyle=line_style,
                alpha=0.95,
                zorder=CURVE_ZORDER.get(style_key, 1),
            )

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("Time (ms/query)")
        ax.set_ylabel("Recall@20")
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.legend(loc="lower right")
        set_recall_ylim(ax, plotted_recalls)
        _clean_ax(ax)

    for idx in range(n_ds, len(axes)):
        axes[idx].set_visible(False)

    fig_name = "fig9_ablation_shg_hnsw_no_shortcut_no_lb"
    fig.savefig(output_dir / f"{fig_name}.pdf")
    fig.savefig(output_dir / f"{fig_name}.png")
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

    print("Plotting construction/memory bars (Fig 4a/4b): SHG, HNSW, Panorama, IVFFlat...")
    plot_construction_core(results, output_dir)

    print("Plotting recall vs time (k=20): SHG, HNSW, Panorama, IVFFlat...")
    plot_recall_vs_time_core(results, output_dir, "recall_k20", 20)

    print("Plotting recall vs time (k=50): SHG, HNSW, Panorama, IVFFlat...")
    plot_recall_vs_time_core(results, output_dir, "recall_k50", 50)

    print("Plotting ablation: SHG, HNSW, IVFFlat, SHG-no-shortcut, SHG-no-lb...")
    plot_ablation_core(results, output_dir)

    print(f"\nAll requested plots saved to {output_dir}")


if __name__ == "__main__":
    main()
