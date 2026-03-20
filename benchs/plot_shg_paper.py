#!/usr/bin/env python3
"""
benchs/plot_shg_paper.py

Generate paper-style plots from benchmark results produced by bench_shg_paper.py.

Generates:
  - Figure 4a: Construction time (bar chart, all datasets)
  - Figure 4b: Memory cost (bar chart, all datasets)
  - Figure 5/6: Recall vs time curves (k=20, k=50) per dataset
  - Figure 8: Robustness boxplots
  - Figure 9: Ablation study (SHG vs SHG-no-shortcut vs HNSW)

Usage:
  python benchs/plot_shg_paper.py --results-dir benchs/results/
"""

import argparse
import json
import os
import sys

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

INDEX_COLORS = {
    "SHG": "#e74c3c",
    "HNSW": "#3498db",
    "PANORAMA": "#2ecc71",
    "IVFFLAT": "#9b59b6",
    "IVFPQ": "#f39c12",
    "SHG-NO-SHORTCUT": "#c0392b",
    "SHG-NO-LB": "#e67e22",
    "SHG-NO-BOTH": "#95a5a6",
}

INDEX_MARKERS = {
    "SHG": "o",
    "HNSW": "s",
    "PANORAMA": "D",
    "IVFFLAT": "^",
    "IVFPQ": "v",
    "SHG-NO-SHORTCUT": "x",
    "SHG-NO-LB": "+",
    "SHG-NO-BOTH": "*",
}


def load_results(results_dir):
    """Load all result JSON files."""
    results = {}
    for fname in os.listdir(results_dir):
        if fname.startswith("results_") and fname.endswith(".json"):
            ds_name = fname[len("results_"):-len(".json")]
            with open(os.path.join(results_dir, fname)) as f:
                results[ds_name] = json.load(f)
    return results


def plot_construction(results, output_dir):
    """Plot construction time and memory cost bar charts."""
    datasets = [ds for ds in DATASETS_ORDER if ds in results]
    if not datasets:
        return

    indices = ["SHG", "HNSW", "Panorama", "IVFFlat", "IVFPQ"]

    # Construction time
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(datasets))
    width = 0.15
    for i, idx_name in enumerate(indices):
        times = []
        for ds in datasets:
            r = results[ds].get("construction", {}).get(idx_name, {})
            t = r.get("build_time_s", 0)
            times.append(t if t > 0 else 0.01)
        ax.bar(x + i * width, times, width, label=idx_name,
               color=INDEX_COLORS.get(idx_name.upper(), "#333"))
    ax.set_yscale("log")
    ax.set_ylabel("Construction Time (s)")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([DATASET_LABELS.get(ds, ds) for ds in datasets], rotation=15)
    ax.legend()
    ax.set_title("Index Construction Time")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4a_construction_time.pdf"), dpi=150)
    fig.savefig(os.path.join(output_dir, "fig4a_construction_time.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig4a_construction_time")

    # Memory cost
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, idx_name in enumerate(indices):
        mems = []
        for ds in datasets:
            r = results[ds].get("construction", {}).get(idx_name, {})
            m = r.get("memory_mb", 0)
            mems.append(m if m > 0 else 0.01)
        ax.bar(x + i * width, mems, width, label=idx_name,
               color=INDEX_COLORS.get(idx_name.upper(), "#333"))
    ax.set_yscale("log")
    ax.set_ylabel("Memory Cost (MB)")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([DATASET_LABELS.get(ds, ds) for ds in datasets], rotation=15)
    ax.legend()
    ax.set_title("Index Memory Cost")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4b_memory_cost.pdf"), dpi=150)
    fig.savefig(os.path.join(output_dir, "fig4b_memory_cost.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig4b_memory_cost")


def plot_recall_vs_time(results, output_dir, k_key, k_val):
    """Plot recall vs time curves for each dataset."""
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

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        data = results[ds][k_key]

        for label, points in data.items():
            if not points:
                continue
            recalls = [p["recall"] for p in points]
            times = [p["ms_per_query"] for p in points]
            color = INDEX_COLORS.get(label, "#333")
            marker = INDEX_MARKERS.get(label, "o")
            ax.plot(times, recalls, marker=marker, color=color,
                    label=label, markersize=5, linewidth=1.5)

        ax.set_xlabel("Time (ms/query)")
        ax.set_ylabel("Recall")
        ax.set_title(f"{DATASET_LABELS.get(ds, ds)} (k={k_val})")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_ds, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    fig_name = f"fig5_recall_vs_time_k{k_val}" if k_val == 20 else f"fig6_recall_vs_time_k{k_val}"
    fig.savefig(os.path.join(output_dir, f"{fig_name}.pdf"), dpi=150)
    fig.savefig(os.path.join(output_dir, f"{fig_name}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_name}")


def plot_robustness(results, output_dir):
    """Plot robustness boxplots (Figure 8 style)."""
    datasets = [ds for ds in DATASETS_ORDER if ds in results and "robustness" in results[ds]]
    if not datasets:
        return

    indices_to_plot = ["SHG", "HNSW", "PANORAMA", "IVFFLAT", "IVFPQ"]

    fig, ax = plt.subplots(figsize=(14, 6))

    positions = []
    labels = []
    box_data = []
    colors = []
    pos = 0

    for ds in datasets:
        rob = results[ds]["robustness"]
        ds_label = DATASET_LABELS.get(ds, ds)

        for idx_name in indices_to_plot:
            if idx_name not in rob:
                continue
            r = rob[idx_name]
            # Reconstruct approximate distribution from summary stats
            box_data.append({
                "med": r.get("median_recall", 0),
                "q1": r.get("q25_recall", 0),
                "q3": r.get("q75_recall", 0),
                "whislo": r.get("min_recall", 0),
                "whishi": r.get("max_recall", 0),
                "label": idx_name,
            })
            positions.append(pos)
            labels.append(f"{ds_label}\n{idx_name}")
            colors.append(INDEX_COLORS.get(idx_name, "#333"))
            pos += 1
        pos += 1  # gap between datasets

    if box_data:
        bxp = ax.bxp(box_data, positions=positions, widths=0.6,
                      patch_artist=True, showfliers=False)
        for patch, color in zip(bxp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel("Recall")
        ax.set_title("Robustness: Recall Distribution on Unseen Queries (k=20)")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig8_robustness.pdf"), dpi=150)
    fig.savefig(os.path.join(output_dir, "fig8_robustness.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig8_robustness")


def plot_ablation(results, output_dir):
    """Plot ablation study: SHG vs SHG-no-shortcut vs SHG-no-lb vs HNSW (Figure 9 style)."""
    datasets = [ds for ds in DATASETS_ORDER if ds in results and "recall_k20" in results[ds]]
    if not datasets:
        return

    ablation_keys = ["SHG", "HNSW", "SHG-NO-SHORTCUT", "SHG-NO-LB", "SHG-NO-BOTH"]
    ablation_labels = {
        "SHG": "SHG",
        "HNSW": "HNSW",
        "SHG-NO-SHORTCUT": "SHG (no shortcut)",
        "SHG-NO-LB": "SHG (no LB pruning)",
        "SHG-NO-BOTH": "SHG (no shortcut, no LB)",
    }

    n_ds = len(datasets)
    cols = min(4, n_ds)
    rows = (n_ds + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_ds == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        data = results[ds]["recall_k20"]

        for label in ablation_keys:
            key = label.lower() if label in ["SHG", "HNSW"] else label
            if key not in data and label not in data:
                continue
            points = data.get(key, data.get(label, []))
            if not points:
                continue
            recalls = [p["recall"] for p in points]
            times = [p["ms_per_query"] for p in points]
            color = INDEX_COLORS.get(label, "#333")
            marker = INDEX_MARKERS.get(label, "o")
            ax.plot(times, recalls, marker=marker, color=color,
                    label=ablation_labels.get(label, label),
                    markersize=5, linewidth=1.5)

        ax.set_xlabel("Time (ms/query)")
        ax.set_ylabel("Recall")
        ax.set_title(f"{DATASET_LABELS.get(ds, ds)}")
        ax.legend(fontsize=6, loc="lower right")
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

    for idx in range(n_ds, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Ablation Study: Effect of Shortcuts and LB Pruning (k=20)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig9_ablation.pdf"), dpi=150)
    fig.savefig(os.path.join(output_dir, "fig9_ablation.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig9_ablation")


def main():
    parser = argparse.ArgumentParser(description="Plot SHG paper benchmark results")
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory with results_*.json files",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output plots (default: same as results-dir)",
    )
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results"
        )
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "plots")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from {args.results_dir}")
    results = load_results(args.results_dir)
    if not results:
        print("No results found!")
        sys.exit(1)

    print(f"Found results for: {list(results.keys())}")
    print(f"Plots will be saved to {args.output_dir}\n")

    print("Plotting construction benchmarks...")
    plot_construction(results, args.output_dir)

    print("Plotting recall vs time (k=20)...")
    plot_recall_vs_time(results, args.output_dir, "recall_k20", 20)

    print("Plotting recall vs time (k=50)...")
    plot_recall_vs_time(results, args.output_dir, "recall_k50", 50)

    print("Plotting robustness...")
    plot_robustness(results, args.output_dir)

    print("Plotting ablation study...")
    plot_ablation(results, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
