#!/usr/bin/env python3
"""
benchs/plot_cspg_paper_figures.py

Reproduce paper-quality plots from the CSPG NeurIPS 2024 paper:
  Ming Yang, Yuzheng Cai, Weiguo Zheng.
  "CSPG: Crossing Sparse Proximity Graphs for Approximate Nearest Neighbor
   Search." NeurIPS 2024.

Comparison: HNSW vs CSPG-HNSW (built on FAISS IndexCSPG).

Figures produced (matching paper numbering):
  table1_construction.{pdf,png}      — Table 1: Construction cost comparison
  fig3_qps_recall_k10.{pdf,png}      — Figure 3: QPS vs Recall@10 per dataset
  fig4_varying_n.{pdf,png}           — Figure 4: QPS vs Recall@10 varying n
  fig5_ablation_m.{pdf,png}          — Figure 5: QPS vs Recall@10 varying m
  fig6_ablation_lambda.{pdf,png}     — Figure 6: QPS vs Recall@10 varying λ
  fig7_ablation_ef1.{pdf,png}        — Figure 7: QPS vs Recall@10 varying ef1
  fig8_detour_factor.{pdf,png}       — Figure 8: Detour factor w vs Recall@10
  fig9_robustness.{pdf,png}          — Figure 9: Robustness boxplots

Usage:
  python benchs/plot_cspg_paper_figures.py --results-dir benchs/results_cspg/
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Style configuration (matching paper aesthetics)
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

DATASETS_ORDER = ["sift1m", "deep1m", "gist1m", "uqv1m", "openai1m", "sift10m"]
PAPER_CORE_DATASETS = ["sift1m", "deep1m", "gist1m", "sift10m"]
DATASET_LABELS = {
    "sift1m":   "SIFT1M",
    "deep1m":   "DEEP1M",
    "gist1m":   "GIST1M",
    "uqv1m":    "UQV1M",
    "openai1m": "OpenAI1M",
    "sift10m":  "SIFT10M",
}

# Colors/markers for main comparison (Fig 3)
STYLE_MAIN = {
    "HNSW":      {"color": "#3498db", "marker": "s", "ls": "--", "label": "HNSW"},
    "CSPG-HNSW": {"color": "#e74c3c", "marker": "o", "ls": "-",  "label": "CSPG-HNSW"},
}

# Ablation color palette (Figs 5-7)
ABLATION_CMAP = plt.cm.tab10


# ---------------------------------------------------------------------------
# Label classification
# ---------------------------------------------------------------------------

def _classify(label):
    """Map a JSON label like 'CSPG(m=2,λ=0.5)' to a display class."""
    if label.startswith("CSPG"):
        return "CSPG-HNSW"
    return label  # "HNSW", "IVFFlat", etc.


def _style(label):
    cls = _classify(label)
    return STYLE_MAIN.get(cls, {"color": "#555", "marker": "o", "ls": "-", "label": cls})


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_results(results_dir):
    """Load all results_cspg_*.json files into {dataset_name: data}."""
    results = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("results_cspg_") and fname.endswith(".json"):
            ds = fname[len("results_cspg_"):-len(".json")]
            with open(os.path.join(results_dir, fname)) as f:
                results[ds] = json.load(f)
    return results


def _ms_to_qps(ms_per_query):
    """Convert ms/query to queries per second."""
    if ms_per_query <= 0:
        return 0.0
    return 1000.0 / ms_per_query


def _save(fig, output_dir, stem):
    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {stem}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Table 1: Construction cost comparison
# ---------------------------------------------------------------------------

def plot_table1(results, output_dir):
    """Render Table 1 as a figure: DS (data size), IS (index size), IT (time)."""
    datasets = [ds for ds in DATASETS_ORDER if ds in results
                and "construction" in results[ds]]
    if not datasets:
        print("  No construction data — skipping Table 1.")
        return

    # Build table data
    # Columns: Dataset | index | IS (MB) | IT (s) for each dataset
    fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(datasets)), 4))
    ax.axis("off")

    # Collect all index labels
    all_labels = []
    for ds in datasets:
        for lbl in results[ds]["construction"]:
            cls = _classify(lbl)
            if cls not in all_labels:
                all_labels.append(cls)

    # Column headers
    col_headers = []
    for ds in datasets:
        ds_label = DATASET_LABELS.get(ds, ds.upper())
        col_headers.extend([f"IS (MB)\n{ds_label}", f"IT (s)\n{ds_label}"])

    row_labels = all_labels
    cell_data = []
    for cls in all_labels:
        row = []
        for ds in datasets:
            con = results[ds]["construction"]
            # Find the matching label
            matched = None
            for lbl in con:
                if _classify(lbl) == cls:
                    matched = con[lbl]
                    break
            if matched:
                is_mb = matched.get("memory_mb", -1)
                it_s = matched.get("build_time_s", -1)
                row.append(f"{is_mb:.0f}" if is_mb >= 0 else "—")
                row.append(f"{it_s:.0f}" if it_s >= 0 else "—")
            else:
                row.extend(["—", "—"])
        cell_data.append(row)

    # Color cells: CSPG rows in light red, HNSW rows in light blue
    cell_colors = []
    for cls in all_labels:
        if cls == "CSPG-HNSW":
            cell_colors.append(["#fde0dc"] * len(col_headers))
        elif cls == "HNSW":
            cell_colors.append(["#dce8fd"] * len(col_headers))
        else:
            cell_colors.append(["#f0f0f0"] * len(col_headers))

    table = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_headers,
        cellColours=cell_colors,
        rowColours=["#fde0dc" if c == "CSPG-HNSW" else "#dce8fd" if c == "HNSW"
                    else "#f0f0f0" for c in all_labels],
        colColours=["#e8e8e8"] * len(col_headers),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    ax.set_title("Table 1: Index Construction Cost — IS = Index Size (MB), IT = Construction Time (s)",
                 fontsize=12, fontweight="bold", pad=20)
    fig.tight_layout()
    _save(fig, output_dir, "table1_construction")


# ---------------------------------------------------------------------------
# Figure 3: QPS vs Recall@10 (one panel per dataset)
# ---------------------------------------------------------------------------

def plot_fig3(results, output_dir):
    """QPS vs Recall@10 curves — HNSW vs CSPG-HNSW, one panel per dataset."""
    bm_key = "recall_k10"
    datasets = [ds for ds in DATASETS_ORDER if ds in results
                and bm_key in results[ds]]
    if not datasets:
        print("  No recall_k10 data — skipping Figure 3.")
        return

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(4.5 * n_ds, 3.8), squeeze=False)
    axes = axes.flatten()

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        data = results[ds][bm_key]

        for label, pts in data.items():
            if not pts:
                continue
            s = _style(label)
            recalls = [p["recall"] for p in pts]
            qps = [_ms_to_qps(p["ms_per_query"]) for p in pts]
            ax.plot(recalls, qps,
                    marker=s["marker"], color=s["color"], linestyle=s["ls"],
                    label=s["label"], markersize=5, linewidth=1.5)

        ax.set_xlabel("Recall@10")
        ax.set_ylabel("QPS")
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
        _auto_xlim(ax, data)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    # Shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=9, bbox_to_anchor=(0.5, 1.05))
    # Remove per-axis legends
    for ax in axes:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    fig.suptitle("Figure 3: QPS vs Recall@10 — HNSW vs CSPG-HNSW", fontsize=13, y=1.10)
    fig.tight_layout()
    _save(fig, output_dir, "fig3_qps_recall_k10")

    # Also produce k=20 and k=50 variants if data is available
    for k in [20, 50]:
        _plot_qps_recall(results, output_dir, k)


def _plot_qps_recall(results, output_dir, k):
    """Helper: QPS vs Recall@K for arbitrary k."""
    bm_key = f"recall_k{k}"
    datasets = [ds for ds in DATASETS_ORDER if ds in results
                and bm_key in results[ds]]
    if not datasets:
        return

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(4.5 * n_ds, 3.8), squeeze=False)
    axes = axes.flatten()

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        data = results[ds][bm_key]
        for label, pts in data.items():
            if not pts:
                continue
            s = _style(label)
            recalls = [p["recall"] for p in pts]
            qps = [_ms_to_qps(p["ms_per_query"]) for p in pts]
            ax.plot(recalls, qps,
                    marker=s["marker"], color=s["color"], linestyle=s["ls"],
                    label=s["label"], markersize=5, linewidth=1.5)

        ax.set_xlabel(f"Recall@{k}")
        ax.set_ylabel("QPS")
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
        _auto_xlim(ax, data)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=9, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(f"QPS vs Recall@{k} — HNSW vs CSPG-HNSW", fontsize=13, y=1.10)
    fig.tight_layout()
    _save(fig, output_dir, f"fig3_qps_recall_k{k}")


# ---------------------------------------------------------------------------
# Figure 4: QPS vs Recall@10 varying dataset size n
# ---------------------------------------------------------------------------

def plot_fig4(results, output_dir):
    """
    QPS vs Recall@10 when varying the dataset size n.
    Expects result files named results_cspg_sift{scale}.json
    where scale ∈ {0.1m, 0.2m, 0.5m, 2m, 5m}.
    """
    scale_keys = ["sift0.1m", "sift0.2m", "sift0.5m", "sift2m", "sift5m"]
    scale_labels = {
        "sift0.1m": "SIFT0.1M", "sift0.2m": "SIFT0.2M",
        "sift0.5m": "SIFT0.5M", "sift2m":   "SIFT2M",
        "sift5m":   "SIFT5M",
    }
    available = [sk for sk in scale_keys if sk in results and "recall_k10" in results[sk]]
    if not available:
        print("  No varying-n data (sift0.1m..sift5m) — skipping Figure 4.")
        return

    n_panels = len(available)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.8), squeeze=False)
    axes = axes.flatten()

    for idx, sk in enumerate(available):
        ax = axes[idx]
        data = results[sk]["recall_k10"]
        for label, pts in data.items():
            if not pts:
                continue
            s = _style(label)
            recalls = [p["recall"] for p in pts]
            qps = [_ms_to_qps(p["ms_per_query"]) for p in pts]
            ax.plot(recalls, qps,
                    marker=s["marker"], color=s["color"], linestyle=s["ls"],
                    label=s["label"], markersize=5, linewidth=1.5)

        ax.set_xlabel("Recall@10")
        ax.set_ylabel("QPS")
        ax.set_title(scale_labels.get(sk, sk.upper()))
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
        _auto_xlim(ax, data)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=9, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Figure 4: Query performance when varying the dataset size $n$",
                 fontsize=13, y=1.10)
    fig.tight_layout()
    _save(fig, output_dir, "fig4_varying_n")


# ---------------------------------------------------------------------------
# Figure 5: Ablation — varying number of partitions m
# ---------------------------------------------------------------------------

def plot_fig5(results, output_dir):
    """QPS vs Recall@10 varying m (num_partitions)."""
    bm_key = "ablation_m"
    datasets = [ds for ds in PAPER_CORE_DATASETS if ds in results
                and bm_key in results[ds] and results[ds][bm_key]]
    if not datasets:
        print("  No ablation_m data — skipping Figure 5.")
        return

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(4.5 * n_ds, 3.8), squeeze=False)
    axes = axes.flatten()

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        data = results[ds][bm_key]
        labels_sorted = sorted(data.keys(), key=_ablation_sort_key)
        cmap = ABLATION_CMAP(np.linspace(0, 0.8, max(len(labels_sorted), 1)))

        for i, (label, pts) in enumerate(
                [(l, data[l]) for l in labels_sorted]):
            if not pts:
                continue
            # Extract short display label (e.g., "m=1")
            disp = _extract_param_label(label, "m")
            recalls = [p["recall"] for p in pts]
            qps = [_ms_to_qps(p["ms_per_query"]) for p in pts]
            ax.plot(recalls, qps, marker="o", color=cmap[i],
                    label=disp, markersize=5, linewidth=1.5)

        ax.set_xlabel("Recall@10")
        ax.set_ylabel("QPS")
        ax.set_title(f"CSPG-HNSW\n{DATASET_LABELS.get(ds, ds)}")
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
        _auto_xlim(ax, data)
        ax.grid(True, alpha=0.3)

    # Shared legend
    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper center", ncol=len(labels_leg),
               fontsize=9, bbox_to_anchor=(0.5, 1.05))
    for a in axes:
        leg = a.get_legend()
        if leg:
            leg.remove()

    fig.suptitle("Figure 5: Query performance when varying the number of partitions $m$",
                 fontsize=13, y=1.10)
    fig.tight_layout()
    _save(fig, output_dir, "fig5_ablation_m")


# ---------------------------------------------------------------------------
# Figure 6: Ablation — varying sampling ratio λ
# ---------------------------------------------------------------------------

def plot_fig6(results, output_dir):
    """QPS vs Recall@10 varying λ (routing ratio)."""
    bm_key = "ablation_lam"
    datasets = [ds for ds in PAPER_CORE_DATASETS if ds in results
                and bm_key in results[ds] and results[ds][bm_key]]
    if not datasets:
        print("  No ablation_lam data — skipping Figure 6.")
        return

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(4.5 * n_ds, 3.8), squeeze=False)
    axes = axes.flatten()

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        data = results[ds][bm_key]
        labels_sorted = sorted(data.keys(), key=_ablation_sort_key)
        cmap = ABLATION_CMAP(np.linspace(0, 0.8, max(len(labels_sorted), 1)))

        for i, (label, pts) in enumerate(
                [(l, data[l]) for l in labels_sorted]):
            if not pts:
                continue
            disp = _extract_param_label(label, "λ")
            recalls = [p["recall"] for p in pts]
            qps = [_ms_to_qps(p["ms_per_query"]) for p in pts]
            ax.plot(recalls, qps, marker="o", color=cmap[i],
                    label=disp, markersize=5, linewidth=1.5)

        ax.set_xlabel("Recall@10")
        ax.set_ylabel("QPS")
        ax.set_title(f"CSPG-HNSW\n{DATASET_LABELS.get(ds, ds)}")
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
        _auto_xlim(ax, data)
        ax.grid(True, alpha=0.3)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper center", ncol=len(labels_leg),
               fontsize=9, bbox_to_anchor=(0.5, 1.05))

    fig.suptitle(r"Figure 6: Query performance when varying the sampling ratio $\lambda$",
                 fontsize=13, y=1.10)
    fig.tight_layout()
    _save(fig, output_dir, "fig6_ablation_lambda")


# ---------------------------------------------------------------------------
# Figure 7: Ablation — varying ef1
# ---------------------------------------------------------------------------

def plot_fig7(results, output_dir):
    """QPS vs Recall@10 varying ef1 (first-stage candidate set size)."""
    bm_key = "ablation_ef1"
    datasets = [ds for ds in PAPER_CORE_DATASETS if ds in results
                and bm_key in results[ds] and results[ds][bm_key]]
    if not datasets:
        print("  No ablation_ef1 data — skipping Figure 7.")
        return

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(4.5 * n_ds, 3.8), squeeze=False)
    axes = axes.flatten()

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        data = results[ds][bm_key]
        labels_sorted = sorted(data.keys(), key=_ablation_sort_key)
        cmap = ABLATION_CMAP(np.linspace(0, 0.8, max(len(labels_sorted), 1)))

        for i, (label, pts) in enumerate(
                [(l, data[l]) for l in labels_sorted]):
            if not pts:
                continue
            disp = _extract_param_label(label, "ef1")
            recalls = [p["recall"] for p in pts]
            qps = [_ms_to_qps(p["ms_per_query"]) for p in pts]
            ax.plot(recalls, qps, marker="o", color=cmap[i],
                    label=disp, markersize=5, linewidth=1.5)

        ax.set_xlabel("Recall@10")
        ax.set_ylabel("QPS")
        ax.set_title(f"CSPG-HNSW\n{DATASET_LABELS.get(ds, ds)}")
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
        _auto_xlim(ax, data)
        ax.grid(True, alpha=0.3)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper center", ncol=len(labels_leg),
               fontsize=9, bbox_to_anchor=(0.5, 1.05))

    fig.suptitle(r"Figure 7: Query performance when varying the candidate set size $ef_1$ in the first stage",
                 fontsize=13, y=1.10)
    fig.tight_layout()
    _save(fig, output_dir, "fig7_ablation_ef1")


# ---------------------------------------------------------------------------
# Figure 8: Detour factor w vs Recall@10 varying dataset size
# ---------------------------------------------------------------------------

def plot_fig8(results, output_dir):
    """
    Detour factor w vs Recall@10 when varying dataset size.
    Expects a 'detour_factor' key in each result file with structure:
      { "SIFT0.1M": [ {recall, w}, ... ], "SIFT0.2M": [...], ... }

    The detour factor w = len(search_seq) / (len(search_seq) - n_backtracks).
    This data must be produced by a specialized benchmark.
    """
    # Check if any dataset has detour_factor data
    datasets_with_detour = [ds for ds in DATASETS_ORDER if ds in results
                            and "detour_factor" in results[ds]]
    if not datasets_with_detour:
        print("  No detour_factor data — skipping Figure 8.")
        print("    (Run bench_cspg_paper.py with --benchmark detour_factor to generate.)")
        return

    n_ds = len(datasets_with_detour)
    fig, axes = plt.subplots(1, n_ds, figsize=(4.5 * n_ds, 3.8), squeeze=False)
    axes = axes.flatten()

    for idx, ds in enumerate(datasets_with_detour):
        ax = axes[idx]
        data = results[ds]["detour_factor"]

        # data: { "SIFT0.1M": [{recall, w}, ...], ... } or similar
        labels_sorted = sorted(data.keys())
        cmap = ABLATION_CMAP(np.linspace(0, 0.8, max(len(labels_sorted), 1)))

        for i, (label, pts) in enumerate(
                [(l, data[l]) for l in labels_sorted]):
            if not pts:
                continue
            recalls = [p["recall"] for p in pts]
            ws = [p["w"] for p in pts]
            ax.plot(recalls, ws, marker="o", color=cmap[i],
                    label=label, markersize=5, linewidth=1.5)

        ax.set_xlabel("Recall@10")
        ax.set_ylabel("$w$")
        ax.set_title(f"CSPG-HNSW\n{DATASET_LABELS.get(ds, ds)}")
        ax.grid(True, alpha=0.3)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper center", ncol=len(labels_leg),
               fontsize=9, bbox_to_anchor=(0.5, 1.05))

    fig.suptitle("Figure 8: Detour factor when varying the dataset size $n$",
                 fontsize=13, y=1.10)
    fig.tight_layout()
    _save(fig, output_dir, "fig8_detour_factor")


# ---------------------------------------------------------------------------
# Figure 9: Robustness — per-query recall distribution (boxplots)
#   (Paper's Fig 9 shows ANN-Benchmarks. We repurpose the robustness data
#    since ANN-Benchmarks requires external tooling.)
# ---------------------------------------------------------------------------

def plot_fig9(results, output_dir):
    """Robustness boxplots: per-query recall distribution on unseen queries."""
    datasets = [ds for ds in PAPER_CORE_DATASETS if ds in results
                and "robustness" in results[ds] and results[ds]["robustness"]]
    if not datasets:
        print("  No robustness data — skipping Figure 9.")
        return

    fig, ax = plt.subplots(figsize=(max(8, 3 * len(datasets)), 5))
    pos, tick_labels, bxdata, colors = [], [], [], []
    p = 0

    for ds in datasets:
        rob = results[ds]["robustness"]
        ds_label = DATASET_LABELS.get(ds, ds)
        for lbl, stats in rob.items():
            s = _style(lbl)
            bxdata.append({
                "med":    stats.get("median_recall", 0),
                "q1":     stats.get("q25_recall", 0),
                "q3":     stats.get("q75_recall", 0),
                "whislo": stats.get("min_recall", 0),
                "whishi": stats.get("max_recall", 0),
                "label":  s["label"],
            })
            pos.append(p)
            tick_labels.append(f"{ds_label}\n{s['label']}")
            colors.append(s["color"])
            p += 1
        p += 1  # gap between datasets

    if bxdata:
        bxp = ax.bxp(bxdata, positions=pos, widths=0.6,
                      patch_artist=True, showfliers=False)
        for patch, color in zip(bxp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

    ax.set_xticks(pos)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Recall@20")
    ax.set_ylim([0, 1.05])
    ax.set_title("Figure 9: Robustness — Per-Query Recall on Unseen Queries ($k=20$)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "fig9_robustness")


# ---------------------------------------------------------------------------
# Bonus: Construction bar charts (complement to Table 1)
# ---------------------------------------------------------------------------

def plot_construction_bars(results, output_dir):
    """Bar charts of construction time and memory (supplement to Table 1)."""
    datasets = [ds for ds in DATASETS_ORDER if ds in results
                and "construction" in results[ds]]
    if not datasets:
        return

    # Collect all index labels across datasets
    all_cls = []
    for ds in datasets:
        for lbl in results[ds]["construction"]:
            cls = _classify(lbl)
            if cls not in all_cls:
                all_cls.append(cls)

    x = np.arange(len(datasets))
    width = 0.8 / max(len(all_cls), 1)

    for metric, ylabel, stem in [
        ("build_time_s", "Construction Time (s)", "table1_construction_time_bars"),
        ("memory_mb",    "Index Size (MB)",       "table1_index_size_bars"),
    ]:
        fig, ax = plt.subplots(figsize=(max(6, 2 * len(datasets)), 4))

        for i, cls in enumerate(all_cls):
            vals = []
            for ds in datasets:
                con = results[ds]["construction"]
                matched = None
                for lbl in con:
                    if _classify(lbl) == cls:
                        matched = con[lbl]
                        break
                v = matched.get(metric, 0) if matched else 0
                vals.append(max(v, 0.01))

            s = STYLE_MAIN.get(cls, {"color": "#555", "label": cls})
            ax.bar(x + i * width, vals, width,
                   label=s["label"], color=s["color"], alpha=0.85)

        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x + width * (len(all_cls) - 1) / 2)
        ax.set_xticklabels([DATASET_LABELS.get(ds, ds) for ds in datasets])
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"Table 1: {ylabel}")
        fig.tight_layout()
        _save(fig, output_dir, stem)


# ---------------------------------------------------------------------------
# Ablation helper utilities
# ---------------------------------------------------------------------------

def _auto_xlim(ax, data, margin=0.03):
    """Set x-axis limits based on actual recall range in data, with margin."""
    all_recalls = []
    for pts in data.values():
        if pts:
            all_recalls.extend(p["recall"] for p in pts)
    if all_recalls:
        lo = max(0, min(all_recalls) - margin)
        hi = min(1.05, max(all_recalls) + margin)
        ax.set_xlim([lo, hi])
    else:
        ax.set_xlim([0.0, 1.05])


def _ablation_sort_key(label):
    """Sort ablation labels by their numeric parameter value."""
    import re
    nums = re.findall(r"[-+]?\d*\.?\d+", label)
    if nums:
        return float(nums[-1])
    return label


def _extract_param_label(label, param_name):
    """Extract a short display label like 'm=2' from 'CSPG(m=2,λ=0.5)'."""
    import re
    # Try to find param_name=value
    pattern = rf"{re.escape(param_name)}\s*=\s*([-+]?\d*\.?\d+)"
    match = re.search(pattern, label)
    if match:
        return f"${param_name}={match.group(1)}$"

    # For λ (unicode), also try the unicode char
    if param_name == "λ":
        match = re.search(r"[λ\\lambda]\s*=\s*([-+]?\d*\.?\d+)", label)
        if match:
            return f"$\\lambda={match.group(1)}$"

    # Fallback: return the full label
    return label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot CSPG paper figures (NeurIPS 2024)")
    parser.add_argument("--results-dir", default=None,
                        help="Directory containing results_cspg_*.json files")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for plots (default: <results-dir>/plots)")
    parser.add_argument("--figures", nargs="+", default=["all"],
                        choices=["all", "table1", "fig3", "fig4", "fig5",
                                 "fig6", "fig7", "fig8", "fig9"],
                        help="Which figures to generate (default: all)")
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

    print(f"Found datasets: {sorted(results.keys())}")
    print(f"Output: {args.output_dir}\n")

    do_all = "all" in args.figures

    if do_all or "table1" in args.figures:
        print("Table 1: Construction cost…")
        plot_table1(results, args.output_dir)
        plot_construction_bars(results, args.output_dir)

    if do_all or "fig3" in args.figures:
        print("Figure 3: QPS vs Recall@10…")
        plot_fig3(results, args.output_dir)

    if do_all or "fig4" in args.figures:
        print("Figure 4: Varying dataset size n…")
        plot_fig4(results, args.output_dir)

    if do_all or "fig5" in args.figures:
        print("Figure 5: Ablation — m…")
        plot_fig5(results, args.output_dir)

    if do_all or "fig6" in args.figures:
        print("Figure 6: Ablation — λ…")
        plot_fig6(results, args.output_dir)

    if do_all or "fig7" in args.figures:
        print("Figure 7: Ablation — ef1…")
        plot_fig7(results, args.output_dir)

    if do_all or "fig8" in args.figures:
        print("Figure 8: Detour factor…")
        plot_fig8(results, args.output_dir)

    if do_all or "fig9" in args.figures:
        print("Figure 9: Robustness…")
        plot_fig9(results, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
