"""
Visualizes SuCo benchmark results from bench_all_20260305_152636.log.
Generates recall-QPS Pareto frontier plots for SIFT1M, GIST1M and Deep1M,
plus supplementary parameter-sweep and build-time charts.

Run:
    python benchs/plot_bench_results.py
Output:  benchs/bench_results.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Raw data extracted from bench_all_20260305_152636.log
# Each entry: (Recall@1, QPS)
# ---------------------------------------------------------------------------

# ── SIFT1M ─────────────────────────────────────────────────────────────────
sift_suco_default  = (0.9892, 234)   # default Ns=8, nc=50, α=0.05
sift_flat           = (0.9914, 106)   # upper-bound reference

# SuCo parameter sweep – (R@1, QPS, label)
sift_suco_sweep = [
    (0.9766, 208,  "Ns=4"),
    (0.9892, 214,  "Ns=8 (default)"),
    (0.9888, 199,  "Ns=16"),
    (0.9596, 262,  "α=0.02"),
    (0.9903, 163,  "α=0.10"),
    (0.9934, 103,  "α=0.20"),
    (0.9811, 226,  "nc=25"),
    (0.9903, 186,  "nc=100"),
]

sift_hnsw = [    # (R@1, QPS)  efSearch = 8,16,32,64,128,256,512
    (0.8114, 35181), (0.9097, 22389), (0.9595, 13401),
    (0.9815,  7405), (0.9897,  4260), (0.9912,  2199), (0.9913,  1134),
]
sift_ivfflat = [  # nprobe = 1,2,4,8,16,32,64,128,256,512
    (0.4633, 27327), (0.6258, 16724), (0.7720,  9480),
    (0.8796,  5637), (0.9460,  3110), (0.9795,  1631),
    (0.9891,   830), (0.9912,   435), (0.9914,   227), (0.9914,   114),
]
sift_ivfpq = [    # nprobe = 1,2,4,8,16,32,64,128,256,512
    (0.2017, 31672), (0.2429, 26016), (0.2651, 19541),
    (0.2781, 13822), (0.2840,  8915), (0.2847,  4622),
    (0.2846,  2594), (0.2848,  1633), (0.2848,   862), (0.2848,   446),
]
sift_opqpq = [
    (0.2112, 26927), (0.2531, 26195), (0.2805, 19036),
    (0.2922, 12658), (0.2983,  8668), (0.3002,  5435),
    (0.3010,  3017), (0.3011,  1650), (0.3011,   870), (0.3011,   445),
]

# ── GIST1M ─────────────────────────────────────────────────────────────────
gist_suco_default = (0.9760, 99)
gist_flat          = (0.9940, 20)

gist_suco_sweep = [
    (0.8890, 117, "Ns=8"),
    (0.9420, 117, "Ns=16"),
    (0.9560, 110, "Ns=24"),
    (0.9760, 100, "Ns=40 (default)"),
    (0.9740,  81, "Ns=60"),
    (0.8530, 173, "α=0.02"),
    (0.9940,  57, "α=0.10"),
    (0.9910,  30, "α=0.20"),
    (0.9400, 105, "nc=25"),
    (0.9820,  72, "nc=100"),
]

gist_hnsw = [
    (0.4860, 6418), (0.6420, 4665), (0.7750, 3280),
    (0.8700, 2029), (0.9320, 1138), (0.9630,  609), (0.9790,  318),
]
gist_ivfflat = [
    (0.2770, 2690), (0.4270, 1688), (0.5710,  919),
    (0.7130,  490), (0.8480,  270), (0.9170,  139),
    (0.9700,   74), (0.9890,   38), (0.9930,   19), (0.9940,   11),
]
gist_ivfpq = [
    (0.1270, 3304), (0.1610, 2738), (0.1860, 2361),
    (0.2010, 1574), (0.2040, 1243), (0.2060,  924),
    (0.2060,  778), (0.2060,  444), (0.2060,  219), (0.2060,  129),
]
gist_opqpq = [
    (0.1810, 2524), (0.2550, 1987), (0.3110, 2059),
    (0.3640, 1420), (0.4140, 1236), (0.4270,  988),
    (0.4340,  691), (0.4340,  379), (0.4360,  217), (0.4370,  122),
]

# ── Deep1M ─────────────────────────────────────────────────────────────────
deep_suco_default = (0.9962, 258)
deep_flat          = (0.9997, 109)

deep_suco_sweep = [
    (0.9711, 303, "α=0.02"),
    (0.9962, 256, "α=0.05 (default)"),
    (0.9992, 200, "α=0.10"),
    (0.9996, 121, "α=0.20"),
    (0.9759, 247, "Ns=4"),
    (0.9962, 256, "Ns=8 (default)"),
    (0.9981, 245, "Ns=12"),
    (0.9887, 268, "nc=25"),
    (0.9982, 208, "nc=100"),
]

deep_hnsw = [
    (0.8200, 38471), (0.9167, 27291), (0.9660, 14067),
    (0.9914,  7733), (0.9968,  4538), (0.9989,  2377), (0.9995,  1217),
]
deep_ivfflat = [
    (0.4899, 33571), (0.6611, 20722), (0.7977, 11353),
    (0.8968,  6236), (0.9536,  3802), (0.9840,  2053),
    (0.9958,  1089), (0.9989,   578), (0.9994,   301), (0.9997,   152),
]
deep_ivfpq = [
    (0.1564, 38586), (0.1857, 33645), (0.1983, 25202),
    (0.2050, 13374), (0.2070,  7881), (0.2071,  5091),
    (0.2071,  2732), (0.2071,  1808), (0.2071,   941), (0.2071,   481),
]
deep_opqpq = [
    (0.1753, 35416), (0.2046, 22979), (0.2243, 22827),
    (0.2311, 12160), (0.2336, 10549), (0.2343,  5963),
    (0.2345,  3175), (0.2345,  1784), (0.2345,   925), (0.2345,   477),
]

# ---------------------------------------------------------------------------
# Build-time data (seconds or minutes where noted)
# ---------------------------------------------------------------------------
build_data = {
    # (index, dataset): build_time_seconds
    ("SuCo",    "SIFT1M"): 2.46,
    ("SuCo",    "GIST1M"): 21.39,
    ("SuCo",    "Deep1M"): 3.76,
    ("HNSW",    "SIFT1M"): 5.3 * 60,    # 5.3 min
    ("HNSW",    "GIST1M"): 18.8 * 60,   # 18.8 min
    ("HNSW",    "Deep1M"): 5.1 * 60,    # 5.1 min
}

# Index size (MiB)
index_size = {
    ("SuCo", "SIFT1M"): 549.6,
    ("SuCo", "GIST1M"): 3968.9,
    ("SuCo", "Deep1M"): 427.6,
}
raw_data_size = {          # nb * d * 4 bytes → MiB
    "SIFT1M": 1e6 * 128 * 4 / 2**20,
    "GIST1M": 1e6 * 960 * 4 / 2**20,
    "Deep1M": 1e6 *  96 * 4 / 2**20,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pareto_frontier(pts):
    """Return Pareto-optimal (recall, QPS) pairs (max recall for given QPS)."""
    pts = sorted(pts, key=lambda p: p[0])
    frontier = []
    best_qps = -1
    for r, q in pts:
        if q > best_qps:
            frontier.append((r, q))
            best_qps = q
    return frontier

def unzip(pts):
    r, q = zip(*[(p[0], p[1]) for p in pts])
    return list(r), list(q)

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("#f8f9fa")

# Row 0: three Recall–QPS Pareto plots
# Row 1: three parameter-sweep scatter plots
# Row 2: build-time bar chart + index-size bar chart + GIST detail
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.32,
                      left=0.07, right=0.97, top=0.93, bottom=0.06)

COLORS = {
    "HNSW":      "#e63946",
    "IVFFlat":   "#457b9d",
    "IVFPQ":     "#a8dadc",
    "OPQ+IVFPQ": "#90be6d",
    "SuCo":      "#f4a261",
    "FlatL2":    "#6c757d",
    "SuCo sweep":"#e76f51",
}
MARKERS = {"HNSW": "^", "IVFFlat": "s", "IVFPQ": "D",
           "OPQ+IVFPQ": "P", "SuCo": "o", "FlatL2": "X"}


def plot_pareto(ax, dataset_label,
                hnsw_pts, ivfflat_pts, ivfpq_pts, opqpq_pts,
                suco_default, flat_pt,
                x_lo=0.3):
    ax.set_facecolor("#ffffff")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlim(x_lo, 1.0)
    ax.set_xlabel("Recall@1", fontsize=11)
    ax.set_ylabel("QPS  (log scale)", fontsize=11)
    ax.set_title(dataset_label, fontsize=13, fontweight="bold")

    def draw(pts, name):
        pf = pareto_frontier(pts)
        r, q = unzip(pf)
        ax.plot(r, q, "-o", markersize=5, linewidth=2,
                color=COLORS[name], marker=MARKERS[name],
                label=name, alpha=0.9, zorder=3)

    draw(hnsw_pts,    "HNSW")
    draw(ivfflat_pts, "IVFFlat")
    draw(ivfpq_pts,   "IVFPQ")
    draw(opqpq_pts,   "OPQ+IVFPQ")

    # SuCo single operating point
    ax.scatter([suco_default[0]], [suco_default[1]],
               s=180, color=COLORS["SuCo"], marker="o",
               zorder=6, label="SuCo (default)", edgecolors="black", linewidths=1.2)

    # FlatL2 reference
    ax.axvline(flat_pt[0], color=COLORS["FlatL2"], linestyle=":", linewidth=1.4, alpha=0.7)
    ax.scatter([flat_pt[0]], [flat_pt[1]],
               s=130, color=COLORS["FlatL2"], marker="X",
               zorder=5, label="FlatL2 (exact)", edgecolors="black", linewidths=0.8)


def plot_sweep(ax, dataset_label, sweep_pts, default_r1, xlo=0.8):
    ax.set_facecolor("#ffffff")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.set_xlabel("Recall@1", fontsize=11)
    ax.set_ylabel("QPS", fontsize=11)
    ax.set_title(f"{dataset_label} — SuCo param sweep", fontsize=12,
                 fontweight="bold")
    ax.set_xlim(xlo, 1.0)

    for r, q, lbl in sweep_pts:
        color = "#f4a261" if "default" in lbl else "#e76f51"
        zorder = 7 if "default" in lbl else 4
        size = 120 if "default" in lbl else 60
        ax.scatter(r, q, s=size, color=color, edgecolors="black",
                   linewidths=1 if "default" in lbl else 0.5,
                   zorder=zorder)
        offset = (0.001, 8)
        ax.annotate(lbl, (r, q),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=7.5, color="#333333")


# ── Row 0: Pareto plots ────────────────────────────────────────────────────
ax00 = fig.add_subplot(gs[0, 0])
plot_pareto(ax00, "SIFT1M  (d=128, 1M vecs, 10K queries)",
            sift_hnsw, sift_ivfflat, sift_ivfpq, sift_opqpq,
            sift_suco_default, sift_flat, x_lo=0.20)
ax00.legend(fontsize=8, loc="lower right")

ax01 = fig.add_subplot(gs[0, 1])
plot_pareto(ax01, "GIST1M  (d=960, 1M vecs, 1K queries)",
            gist_hnsw, gist_ivfflat, gist_ivfpq, gist_opqpq,
            gist_suco_default, gist_flat, x_lo=0.10)
ax01.legend(fontsize=8, loc="lower right")

ax02 = fig.add_subplot(gs[0, 2])
plot_pareto(ax02, "Deep1M  (d=96, 1M vecs, 10K queries)",
            deep_hnsw, deep_ivfflat, deep_ivfpq, deep_opqpq,
            deep_suco_default, deep_flat, x_lo=0.15)
ax02.legend(fontsize=8, loc="lower right")

# ── Row 1: Parameter sweeps ────────────────────────────────────────────────
ax10 = fig.add_subplot(gs[1, 0])
plot_sweep(ax10, "SIFT1M", sift_suco_sweep, sift_suco_default[0], xlo=0.93)

ax11 = fig.add_subplot(gs[1, 1])
plot_sweep(ax11, "GIST1M", gist_suco_sweep, gist_suco_default[0], xlo=0.82)

ax12 = fig.add_subplot(gs[1, 2])
plot_sweep(ax12, "Deep1M", deep_suco_sweep, deep_suco_default[0], xlo=0.93)

# ── Row 2: Build time & index size ────────────────────────────────────────
ax20 = fig.add_subplot(gs[2, 0])
ax20.set_facecolor("#ffffff")
ax20.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
datasets = ["SIFT1M", "GIST1M", "Deep1M"]
x = np.arange(len(datasets))
w = 0.35
suco_bt = [build_data[("SuCo",  d)] for d in datasets]
hnsw_bt = [build_data[("HNSW",  d)] for d in datasets]
b1 = ax20.bar(x - w/2, suco_bt, w, label="SuCo",  color=COLORS["SuCo"],
              edgecolor="black", linewidth=0.7)
b2 = ax20.bar(x + w/2, hnsw_bt, w, label="HNSW",  color=COLORS["HNSW"],
              edgecolor="black", linewidth=0.7)
for bar in b1:
    ax20.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
              f"{bar.get_height():.1f}s", ha="center", va="bottom", fontsize=8)
for bar in b2:
    ax20.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
              f"{bar.get_height()/60:.1f}min", ha="center", va="bottom", fontsize=8)
ax20.set_xticks(x)
ax20.set_xticklabels(datasets, fontsize=10)
ax20.set_ylabel("Build time  (seconds)", fontsize=11)
ax20.set_title("Build Time: SuCo vs HNSW", fontsize=12, fontweight="bold")
ax20.legend(fontsize=9)

# Index size vs raw data size
ax21 = fig.add_subplot(gs[2, 1])
ax21.set_facecolor("#ffffff")
ax21.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
suco_sz = [index_size[("SuCo", d)] for d in datasets]
raw_sz  = [raw_data_size[d] for d in datasets]
b3 = ax21.bar(x - w/2, suco_sz, w, label="SuCo index", color=COLORS["SuCo"],
              edgecolor="black", linewidth=0.7)
b4 = ax21.bar(x + w/2, raw_sz,  w, label="Raw float32",  color="#ccc",
              edgecolor="black", linewidth=0.7)
for bar in b3:
    ax21.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
              f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7.5)
for bar in b4:
    ax21.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
              f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7.5)
ax21.set_xticks(x)
ax21.set_xticklabels(datasets, fontsize=10)
ax21.set_ylabel("Memory  (MiB)", fontsize=11)
ax21.set_title("Index Size vs Raw Data", fontsize=12, fontweight="bold")
ax21.legend(fontsize=9)

# SuCo speed-up over FlatL2
ax22 = fig.add_subplot(gs[2, 2])
ax22.set_facecolor("#ffffff")
ax22.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
speedup_data = {
    "SIFT1M": {
        "SuCo":     sift_suco_default[1] / sift_flat[1],
        "HNSW@~R1": 2199 / sift_flat[1],        # efSearch=256, R@1≈0.991
        "IVFFlat@~R1": 435 / sift_flat[1],      # nprobe=128, R@1≈0.991
    },
    "GIST1M": {
        "SuCo":     gist_suco_default[1] / gist_flat[1],
        "HNSW@~R1": 318 / gist_flat[1],         # efSearch=512, R@1≈0.979
        "IVFFlat@~R1": 74 / gist_flat[1],       # nprobe=64, R@1≈0.970
    },
    "Deep1M": {
        "SuCo":     deep_suco_default[1] / deep_flat[1],
        "HNSW@~R1": 4538 / deep_flat[1],        # efSearch=128, R@1≈0.997
        "IVFFlat@~R1": 1089 / deep_flat[1],     # nprobe=64, R@1≈0.996
    },
}
methods = ["SuCo", "HNSW@~R1", "IVFFlat@~R1"]
mcolors = [COLORS["SuCo"], COLORS["HNSW"], COLORS["IVFFlat"]]
n_methods = len(methods)
xm = np.arange(len(datasets))
bar_w = 0.25
for i, (m, c) in enumerate(zip(methods, mcolors)):
    vals = [speedup_data[d][m] for d in datasets]
    bars = ax22.bar(xm + (i - 1) * bar_w, vals, bar_w, label=m,
                    color=c, edgecolor="black", linewidth=0.7)
    for bar, v in zip(bars, vals):
        ax22.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                  f"{v:.1f}×", ha="center", va="bottom", fontsize=7.5)
ax22.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax22.set_xticks(xm)
ax22.set_xticklabels(datasets, fontsize=10)
ax22.set_ylabel("Speed-up over FlatL2  (×)", fontsize=11)
ax22.set_title("QPS Speed-up vs Exact Search\n(at matching recall level)", fontsize=11.5,
               fontweight="bold")
ax22.legend(fontsize=8.5)

# ── Super-title ────────────────────────────────────────────────────────────
fig.suptitle(
    "SuCo  (Subspace Collision)  —  Benchmark Results  [2026-03-05]\n"
    "SIFT1M · GIST1M · Deep1M   |   nb = 1 M vectors",
    fontsize=14, fontweight="bold", y=0.97
)

out = "/Users/dhm/Documents/SuCo/benchs/bench_results.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved → {out}")
