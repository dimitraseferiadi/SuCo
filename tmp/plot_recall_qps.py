"""
plot_recall_qps.py
==================
Plots Recall@10 vs QPS curves for SuCo, HNSW, IVFFlat, IVFPQ
across Deep1M, Deep10M, GIST1M, and SIFT1M datasets.

All data is hard-coded from the benchmark runs already collected.
Run:  python plot_recall_qps.py
Outputs: recall_qps_<dataset>.png  (individual)
         recall_qps_all.png        (2×2 grid, all datasets)
         suco_param_sweep.png      (alpha / Ns sensitivity per dataset)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# DATA  (recall@10, QPS)  — paste your sweep results here
# ──────────────────────────────────────────────────────────────────────────────

DATA = {

    # ── Deep1M (d=96, LID≈29  →  easy) ──────────────────────────────────────
    "Deep1M": {
        "SuCo": [
            # (R@10,  QPS)   from the alpha/beta/Ns parameter sweep
            (0.9713, 249),   # Ns=8, nc=50, α=0.020, β=0.0020
            (0.9963, 220),   # Ns=8, nc=50, α=0.050, β=0.0050  ← default
            (0.9994, 170),   # Ns=8, nc=50, α=0.100, β=0.0100
            (0.9998, 106),   # Ns=8, nc=50, α=0.200, β=0.0200
            (0.9761, 205),   # Ns=4, nc=50, α=0.050, β=0.0050
            (0.9983, 211),   # Ns=12,nc=50, α=0.050, β=0.0050
            (0.9889, 231),   # Ns=8, nc=25, α=0.050, β=0.0050
            (0.9983, 172),   # Ns=8, nc=100,α=0.050, β=0.0050
        ],
        "HNSW": [
            # efSearch sweep  (M=32, efConstruction=200)
            (0.8172,  31980),
            (0.9111,  23067),
            (0.9661,  14090),
            (0.9901,   8045),
            (0.9969,   4229),
            (0.9993,   2314),
            (0.9999,   1224),
        ],
        "IVFFlat": [
            # nprobe sweep  (nlist=1024)
            (0.4901,  28844),
            (0.6613,  20727),
            (0.7979,  12208),
            (0.8970,   6899),
            (0.9538,   3688),
            (0.9842,   2093),
            (0.9960,   1079),
            (0.9991,    586),
            (0.9996,    303),
            (0.9999,    160),
        ],
        "IVFPQ": [
            # nprobe sweep  (nlist=1024, m=8, nbits=8)
            # recall plateaus early — include all for honest curve
            (0.3870,  30926),
            (0.4871,  28347),
            (0.5509,  20797),
            (0.5840,  14785),
            (0.5963,   8362),
            (0.5997,   4869),
            (0.6009,   2837),
            (0.6009,   1602),
            (0.6009,    803),
            (0.6009,    413),
        ],
        "Flat": [(0.9999, 108)],   # single operating point
    },

    # ── GIST1M (d=960, LID≈70  →  hard) ─────────────────────────────────────
    "GIST1M": {
        "SuCo": [
            # Ns / half_dim / nc  sweep
            (0.8980, 149),   # Ns=8,  hd=60,  nc=50
            (0.9480, 149),   # Ns=16, hd=30,  nc=50
            (0.9630, 137),   # Ns=24, hd=20,  nc=50
            (0.9800,  74),   # Ns=40, hd=12,  nc=50  ← default (stable run)
            (0.9810,  74),   # Ns=60, hd=8,   nc=50
            (0.8610, 172),   # Ns=40, hd=12,  nc=50, α=0.020, β=0.0020
            (0.9980,  57),   # Ns=40, hd=12,  nc=50, α=0.100, β=0.0100
            (0.9990,  31),   # Ns=40, hd=12,  nc=50, α=0.200, β=0.0200
            (0.9480,  89),   # Ns=40, hd=12,  nc=25
            (0.9920,  59),   # Ns=40, hd=12,  nc=100
        ],
        "HNSW": [
            (0.4930,   6551),
            (0.6460,   4646),
            (0.7810,   3021),
            (0.8760,   1834),
            (0.9380,   1019),
            (0.9680,    572),
            (0.9840,    326),
        ],
        "IVFFlat": [
            (0.2800,   2262),
            (0.4330,   1435),
            (0.5770,    873),
            (0.7190,    461),
            (0.8540,    261),
            (0.9230,    138),
            (0.9760,     72),
            (0.9950,     38),
            (0.9990,     20),
            (1.0000,     11),
        ],
        "IVFPQ": [
            (0.2500,   3225),
            (0.3640,   2703),
            (0.4500,   1839),
            (0.5220,   1346),
            (0.5650,   1334),
            (0.5820,    886),
            (0.5800,    686),
            (0.5820,    380),
            (0.5820,    219),
            (0.5820,    134),
        ],
        "Flat": [(1.0000, 20)],
    },

    # ── SIFT1M (d=128, LID≈22  →  easy) ─────────────────────────────────────
    "SIFT1M": {
        "SuCo": [
            (0.9837, 202),   # Ns=4,  hd=16, nc=50
            (0.9965, 201),   # Ns=8,  hd=8,  nc=50, α=0.050, β=0.0050 ← default
            (0.9958, 197),   # Ns=16, hd=4,  nc=50
            (0.9661, 263),   # Ns=8,  hd=8,  nc=50, α=0.020, β=0.0020
            (0.9991, 157),   # Ns=8,  hd=8,  nc=50, α=0.100, β=0.0100
            (0.9998, 101),   # Ns=8,  hd=8,  nc=50, α=0.200, β=0.0200
            (0.9888, 223),   # Ns=8,  hd=8,  nc=25
            (0.9980, 182),   # Ns=8,  hd=8,  nc=100
        ],
        "HNSW": [
            (0.8189,  30142),
            (0.9179,  20995),
            (0.9687,  12382),
            (0.9901,   7054),
            (0.9983,   3990),
            (0.9998,   2086),
            (0.9999,   1095),
        ],
        "IVFFlat": [
            (0.4667,  24950),
            (0.6299,  18597),
            (0.7775,  10233),
            (0.8866,   5861),
            (0.9541,   3114),
            (0.9879,   1627),
            (0.9977,    850),
            (0.9998,    436),
            (1.0000,    228),
            (1.0000,    115),
        ],
        "IVFPQ": [
            (0.4129,  22282),
            (0.5302,  25299),
            (0.6191,  18860),
            (0.6705,  12173),
            (0.6921,   8158),
            (0.6995,   5402),
            (0.7013,   2862),
            (0.7016,   1629),
            (0.7017,    866),
            (0.7017,    452),
        ],
        "Flat": [(1.0000, 108)],
    },

    # ── Deep10M (d=96, nb=10M — SuCo sweep only) ─────────────────────────────
    "Deep10M": {
        "SuCo": [
            (0.9913,  24),   # Ns=8, nc=50, α=0.020, β=0.0020
            (0.9988,  18),   # Ns=8, nc=50, α=0.050, β=0.0050  ← default
            (0.9994,  13),   # Ns=8, nc=50, α=0.100, β=0.0100
            (0.9997,   8),   # Ns=8, nc=50, α=0.200, β=0.0200
            (0.9886,  22),   # Ns=4, nc=50, α=0.050, β=0.0050
            (0.9997,  16),   # Ns=12,nc=50, α=0.050, β=0.0050
            (0.9951,  17),   # Ns=8, nc=25, α=0.050, β=0.0050
            (0.9998,  16),   # Ns=8, nc=100,α=0.050, β=0.0050
        ],
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────────────────────────────────────

STYLE = {
    "SuCo":    dict(color="#e63946", marker="o", linewidth=2.2, markersize=7, zorder=5),
    "HNSW":    dict(color="#457b9d", marker="s", linewidth=2.0, markersize=6, zorder=4),
    "IVFFlat": dict(color="#2a9d8f", marker="^", linewidth=1.8, markersize=6, zorder=3),
    "IVFPQ":   dict(color="#e9c46a", marker="D", linewidth=1.8, markersize=6, zorder=3),
    "Flat":    dict(color="#6c757d", marker="*", linewidth=0,   markersize=10, zorder=2,
                    linestyle="none"),
}

DATASET_META = {
    "Deep1M":  "Deep1M   (d=96,  nb=1M )",
    "Deep10M": "Deep10M  (d=96,  nb=10M — SuCo only)",
    "GIST1M":  "GIST1M   (d=960, nb=1M )",
    "SIFT1M":  "SIFT1M   (d=128, nb=1M )",
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def pareto_front(points):
    """Keep only Pareto-optimal (recall, QPS) points — higher recall AND higher QPS."""
    pts = sorted(points, key=lambda p: p[0])   # sort by recall ascending
    front = []
    best_qps = -1
    for r, q in pts:
        if q > best_qps:
            front.append((r, q))
            best_qps = q
    return front


def plot_dataset(ax, dataset_name, xlim=(0.40, 1.02)):
    methods = DATA[dataset_name]
    ax.set_title(DATASET_META[dataset_name], fontsize=11, pad=8)

    for method, raw_points in methods.items():
        style = STYLE[method].copy()

        if method == "Flat":
            r, q = raw_points[0]
            ax.axvline(r, color=style["color"], linestyle="--", linewidth=1.4,
                       alpha=0.6, label=f"Flat (QPS={q})")
            continue

        front = pareto_front(raw_points)
        recalls = [p[0] for p in front]
        qps     = [p[1] for p in front]

        ls = style.pop("linestyle", "-")
        ln = style.pop("linewidth")
        ax.semilogy(recalls, qps, linestyle=ls, linewidth=ln,
                    label=method, **style)

    ax.set_xlabel("Recall@10", fontsize=10)
    ax.set_ylabel("QPS (log scale)", fontsize=10)
    ax.set_xlim(*xlim)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:.1f}"))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")


# ──────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL PLOTS
# ──────────────────────────────────────────────────────────────────────────────

for ds in ["Deep1M", "Deep10M", "GIST1M", "SIFT1M"]:
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_dataset(ax, ds)
    fig.tight_layout()
    fname = f"recall_qps_{ds}.png"
    fig.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# COMBINED 2×2 GRID
# ──────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, ds in zip(axes.flat, ["Deep1M", "Deep10M", "GIST1M", "SIFT1M"]):
    plot_dataset(ax, ds)

fig.suptitle("Recall@10 vs QPS  —  SuCo vs HNSW, IVFFlat, IVFPQ", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig("recall_qps_all.png", dpi=150, bbox_inches="tight")
print("Saved recall_qps_all.png")
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# SuCo PARAMETER SENSITIVITY  (alpha sweep — fixed Ns=8 or Ns=40 for GIST)
# Shows recall@10 vs QPS for different alpha values on each dataset
# ──────────────────────────────────────────────────────────────────────────────

# alpha sweep points extracted (R@10, QPS)  at fixed Ns / nc
ALPHA_SWEEP = {
    # Deep1M: Ns=8, nc=50, alpha = 0.02 / 0.05 / 0.10 / 0.20
    "Deep1M":  [(0.9713, 249), (0.9963, 220), (0.9994, 170), (0.9998, 106)],
    # Deep10M: Ns=8, nc=50
    "Deep10M": [(0.9913,  24), (0.9988,  18), (0.9994,  13), (0.9997,   8)],
    # GIST1M: Ns=40, nc=50
    "GIST1M":  [(0.8610, 172), (0.9800,  74), (0.9980,  57), (0.9990,  31)],
    # SIFT1M: Ns=8, nc=50
    "SIFT1M":  [(0.9661, 263), (0.9965, 201), (0.9991, 157), (0.9998, 101)],
}
ALPHA_LABELS = ["α=0.02", "α=0.05", "α=0.10", "α=0.20"]
ALPHA_COLORS = ["#1d3557", "#457b9d", "#e63946", "#f4a261"]

# nsubspaces sweep points at fixed alpha=0.05 (R@10, QPS)
NS_SWEEP = {
    "Deep1M":  [(0.9761, 205), (0.9963, 220), (0.9983, 211)],   # Ns=4,8,12
    "Deep10M": [(0.9886,  22), (0.9988,  18), (0.9997,  16)],
    "GIST1M":  [(0.8980, 149), (0.9480, 149), (0.9630, 137), (0.9800, 74), (0.9810, 74)],  # Ns=8,16,24,40,60
    "SIFT1M":  [(0.9837, 202), (0.9965, 201), (0.9958, 197)],   # Ns=4,8,16
}
NS_LABELS = {
    "Deep1M":  ["Ns=4", "Ns=8", "Ns=12"],
    "Deep10M": ["Ns=4", "Ns=8", "Ns=12"],
    "GIST1M":  ["Ns=8", "Ns=16", "Ns=24", "Ns=40", "Ns=60"],
    "SIFT1M":  ["Ns=4", "Ns=8", "Ns=16"],
}
NS_COLORS = ["#e9c46a", "#2a9d8f", "#264653", "#e63946", "#f4a261"]

DATASETS_4 = ["Deep1M", "Deep10M", "GIST1M", "SIFT1M"]

fig, axes = plt.subplots(2, 4, figsize=(22, 9))
fig.suptitle("SuCo Parameter Sensitivity  (Recall@10 vs QPS)", fontsize=13, y=1.01)

for col, ds in enumerate(DATASETS_4):
    # ---- top row: alpha sweep ----
    ax = axes[0, col]
    pts = ALPHA_SWEEP[ds]
    recalls = [p[0] for p in pts]
    qps     = [p[1] for p in pts]
    for i, (r, q, lbl, c) in enumerate(zip(recalls, qps, ALPHA_LABELS, ALPHA_COLORS)):
        ax.scatter(r, q, color=c, s=90, zorder=5)
        ax.annotate(lbl, (r, q), textcoords="offset points",
                    xytext=(5, 4), fontsize=8, color=c)
    ax.plot(recalls, qps, color="#444", linewidth=1.2, linestyle="--", zorder=2)
    ax.set_yscale("log")
    ax.set_title(f"{ds.replace('1M','1M ')} — α sweep", fontsize=10, pad=6)
    ax.set_xlabel("Recall@10", fontsize=9)
    ax.set_ylabel("QPS" if col == 0 else "", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:.1f}"))
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

    # ---- bottom row: Ns sweep ----
    ax = axes[1, col]
    pts  = NS_SWEEP[ds]
    lbls = NS_LABELS[ds]
    cols = NS_COLORS[:len(pts)]
    for (r, q), lbl, c in zip(pts, lbls, cols):
        ax.scatter(r, q, color=c, s=90, zorder=5)
        ax.annotate(lbl, (r, q), textcoords="offset points",
                    xytext=(5, 4), fontsize=8, color=c)
    recalls = [p[0] for p in pts]
    qps     = [p[1] for p in pts]
    ax.plot(sorted(recalls), [q for r, q in sorted(pts)],
            color="#444", linewidth=1.2, linestyle="--", zorder=2)
    ax.set_yscale("log")
    ax.set_title(f"{ds.replace('1M','1M ')} — Ns sweep", fontsize=10, pad=6)
    ax.set_xlabel("Recall@10", fontsize=9)
    ax.set_ylabel("QPS" if col == 0 else "", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:.1f}"))
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

fig.tight_layout()
fig.savefig("suco_param_sweep.png", dpi=150, bbox_inches="tight")
print("Saved suco_param_sweep.png")
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# SCALING FIGURE: SuCo on Deep1M vs Deep10M (same config)
# ──────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_title("SuCo scalability — Deep1M vs Deep10M  (d=96, Ns=8, nc=50)", fontsize=11)

for ds, color, marker in [("Deep1M", "#e63946", "o"), ("Deep10M", "#457b9d", "s")]:
    front = pareto_front(DATA[ds]["SuCo"])
    recalls = [p[0] for p in front]
    qps_vals = [p[1] for p in front]
    ax.semilogy(recalls, qps_vals, color=color, marker=marker,
                linewidth=2.0, markersize=7, label=ds)

ax.set_xlabel("Recall@10", fontsize=10)
ax.set_ylabel("QPS (log scale)", fontsize=10)
ax.set_xlim(0.97, 1.002)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:.1f}"))
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig("suco_scaling.png", dpi=150)
print("Saved suco_scaling.png")
plt.close(fig)