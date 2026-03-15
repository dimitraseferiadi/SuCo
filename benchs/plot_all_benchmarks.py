"""
plot_all_benchmarks.py
======================
Comprehensive benchmark plots for SuCo covering all datasets and platforms
extracted from logs in /Users/dhm/Documents/SuCo/logs/.

Run:
    python benchs/plot_all_benchmarks.py
Outputs:  benchs/plots/  (one PNG per figure group)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "HNSW":      "#e63946",
    "IVFFlat":   "#457b9d",
    "IVFPQ":     "#a8dadc",
    "OPQ+IVFPQ": "#6ab187",
    "SuCo":      "#f4a261",
    "FlatL2":    "#6c757d",
}
MARKERS = {
    "HNSW": "^", "IVFFlat": "s", "IVFPQ": "D",
    "OPQ+IVFPQ": "P", "SuCo": "o", "FlatL2": "X",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "#f8f9fa",
    "axes.facecolor": "#ffffff",
})


def pareto(pts):
    """Return Pareto-optimal (recall, QPS) — max QPS for monotonically increasing recall."""
    pts_sorted = sorted(pts, key=lambda p: p[0])
    frontier, best_q = [], -1
    for r, q in pts_sorted:
        if q > best_q:
            frontier.append((r, q))
            best_q = q
    return frontier

def unzip(pts):
    return [p[0] for p in pts], [p[1] for p in pts]

def draw_pareto(ax, pts, name, **kw):
    pf = pareto(pts)
    r, q = unzip(pf)
    ax.plot(r, q, marker=MARKERS[name], markersize=5.5, linewidth=2,
            color=COLORS[name], label=name, alpha=0.9, zorder=3, **kw)

def setup_pareto_ax(ax, title, xlo=0.20, ylabel=True):
    ax.set_yscale("log")
    ax.set_xlim(xlo, 1.002)
    ax.set_xlabel("Recall@1", fontsize=10)
    if ylabel:
        ax.set_ylabel("QPS  (log scale)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.55)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# ██████  DATA
# ─────────────────────────────────────────────────────────────────────────────

# ── SIFT1M Mac (Mar 11 log — precise values) ────────────────────────────────
sift_mac_suco_default = (0.9895, 333)
sift_mac_flat         = (0.9914, 2742)

sift_mac_suco_sweep = [
    (0.9766,  305, "Ns=4, hd=16"),
    (0.9895,  332, "Ns=8, hd=8 (default)"),
    (0.9896,  330, "Ns=16, hd=4"),
    (0.9595,  414, "α=0.02"),
    (0.9909,  254, "α=0.10"),
    (0.9928,  169, "α=0.20"),
    (0.9820,  333, "nc=25"),
    (0.9903,  293, "nc=100"),
]
sift_mac_hnsw = [
    (0.8109, 140155), (0.9106, 91829), (0.9612, 53677),
    (0.9826,  29540), (0.9897, 15624), (0.9912,  7988), (0.9913,  3894),
]
sift_mac_ivfflat = [
    (0.4633, 176814), (0.6258, 93024), (0.7720, 47754),
    (0.8796,  24588), (0.9460, 12555), (0.9795,  6517),
    (0.9891,   3352), (0.9912,  1733), (0.9914,   896), (0.9914,   453),
]
sift_mac_ivfpq = [
    (0.2017, 400523), (0.2429, 320862), (0.2651, 227330),
    (0.2781, 140867), (0.2840,  79170), (0.2847,  43597),
    (0.2846,  22868), (0.2848,  11959), (0.2848,   6130), (0.2848,  3013),
]
sift_mac_opqpq = [
    (0.2087, 390464), (0.2490, 326840), (0.2747, 228301),
    (0.2873, 143114), (0.2911,  77525), (0.2921,  43168),
    (0.2924,  23517), (0.2924,  11974), (0.2924,   6152), (0.2924,  3030),
]

# ── SIFT1M Linux-B ──────────────────────────────────────────────────────────
sift_linb_suco_default = (0.9892, 65)
sift_linb_flat         = (0.9914, 18)

sift_linb_suco_sweep = [
    (0.9766,  85, "Ns=4, hd=16"),
    (0.9892,  66, "Ns=8, hd=8 (default)"),
    (0.9888,  43, "Ns=16, hd=4"),
    (0.9596, 118, "α=0.02"),
    (0.9903,  39, "α=0.10"),
    (0.9934,  24, "α=0.20"),
    (0.9811,  69, "nc=25"),
    (0.9903,  61, "nc=100"),
]
sift_linb_hnsw = [
    (0.8109, 8800), (0.9093, 5384), (0.9598, 3297),
    (0.9813, 1822), (0.9897,  973), (0.9912,  501), (0.9913,  251),
]
sift_linb_ivfflat = [
    (0.4633, 4303), (0.6258, 2414), (0.7719, 1462),
    (0.8795,  732), (0.9460,  385), (0.9795,  203),
    (0.9891,  104), (0.9912,   54), (0.9914,   28), (0.9914,   14),
]
sift_linb_ivfpq = [
    (0.2011, 11215), (0.2431, 7881), (0.2644, 6420),
    (0.2773,  4197), (0.2836, 2448), (0.2850, 1329),
    (0.2852,   803), (0.2854,  427), (0.2854,  223), (0.2854,  115),
]

# ── GIST1M Mac (Mar 11 log — precise values) ────────────────────────────────
gist_mac_suco_default = (0.9750, 182)
gist_mac_flat         = (0.9940, 650)

gist_mac_suco_sweep = [
    (0.8900,  175, "Ns=8"),
    (0.9450,  188, "Ns=16"),
    (0.9610,  189, "Ns=24"),
    (0.9750,  181, "Ns=40 (default)"),
    (0.9730,  157, "Ns=60"),
    (0.8510,  279, "α=0.02"),
    (0.9880,  113, "α=0.10"),
    (0.9890,   61, "α=0.20"),
    (0.9400,  200, "nc=25"),
    (0.9870,  140, "nc=100"),
]
gist_mac_hnsw = [
    # Note: ef=16 > ef=8 QPS is a real measurement artifact at d=960
    (0.4960, 12395), (0.6510, 17008), (0.7730, 10931),
    (0.8830,  6570), (0.9330,  3734), (0.9620,  2063), (0.9800,  1108),
]
gist_mac_ivfflat = [
    (0.2770, 14876), (0.4260, 9050), (0.5680, 4623),
    (0.7110,  2380), (0.8490, 1190), (0.9200,  602),
    (0.9700,   307), (0.9890,  158), (0.9930,   84), (0.9940,   47),
]
gist_mac_ivfpq = [
    (0.1380, 68787), (0.1720, 45232), (0.1950, 30946),
    (0.2100, 18336), (0.2180, 10305), (0.2190,  5695),
    (0.2190,  2920), (0.2190,  1477), (0.2190,   788), (0.2190,  431),
]
gist_mac_opqpq = [
    (0.1820, 56089), (0.2620, 45437), (0.3080, 27699),
    (0.3710, 18311), (0.4080, 10547), (0.4280,  5426),
    (0.4400,  2982), (0.4410,  1535), (0.4410,   794), (0.4420,  441),
]

# ── GIST1M Linux-B ──────────────────────────────────────────────────────────
gist_linb_suco_default = (0.9750, 25)
gist_linb_flat         = (0.9940, 10)

gist_linb_suco_sweep = [
    (0.8900,  42, "Ns=8"),
    (0.9420,  32, "Ns=16"),
    (0.9600,  28, "Ns=24"),
    (0.9750,  23, "Ns=40 (default)"),
    (0.9740,  18, "Ns=60"),
    (0.8540,  38, "α=0.02"),
    (0.9940,  14, "α=0.10"),
    (0.9910,   8, "α=0.20"),
    (0.9400,  23, "nc=25"),
    (0.9860,  21, "nc=100"),
]
gist_linb_hnsw = [
    (0.4890, 2869), (0.6470, 1415), (0.7770,  708),
    (0.8710,  419), (0.9350,  238), (0.9650,  132), (0.9800,   71),
]
gist_linb_ivfflat = [
    (0.2830, 441), (0.4250, 245), (0.5730, 128),
    (0.7090,  68), (0.8480,  34), (0.9200,  18),
    (0.9690,   9), (0.9890,   5), (0.9930,   2), (0.9940,   1),
]
gist_linb_ivfpq = [
    (0.1340, 1706), (0.2100, 1100), (0.2190, 253),
]

# ── Deep1M Mac (Mar 11 173453 — main run precise value) ─────────────────────
deep_mac_suco_default = (0.9961, 328)
deep_mac_flat         = (0.9997, 3168)

deep_mac_suco_sweep = [
    (0.9711,  392, "α=0.02"),
    (0.9961,  335, "Ns=8, α=0.05 (default)"),
    (0.9993,  278, "α=0.10"),
    (0.9996,  194, "α=0.20"),
    (0.9759,  317, "Ns=4"),
    (0.9981,  342, "Ns=12"),
    (0.9888,  355, "nc=25"),
    (0.9981,  292, "nc=100"),
]
deep_mac_hnsw = [
    (0.8205, 157116), (0.9155, 102981), (0.9672, 61098),
    (0.9897,  33757), (0.9970, 17548),  (0.9992,  9100), (0.9995, 4641),
]
deep_mac_ivfflat = [
    (0.4886, 237610), (0.6611, 129955), (0.7974, 67947),
    (0.8952,  35648), (0.9538, 18556),  (0.9836,  9679),
    (0.9959,   5053), (0.9988,  2637),  (0.9994,  1362), (0.9997,  690),
]
deep_mac_ivfpq = [
    (0.1583, 383823), (0.1832, 326051), (0.1967, 239031),
    (0.2026, 147964), (0.2053,  81872), (0.2059,  46046),
    (0.2059,  24784), (0.2059,  12792), (0.2059,   6739), (0.2059, 3360),
]
deep_mac_opqpq = [
    (0.1733, 372879), (0.2066, 327153), (0.2236, 237777),
    (0.2312, 144936), (0.2343,  84518), (0.2351,  47902),
    (0.2353,  25453), (0.2353,  13293), (0.2353,   6820), (0.2353, 3419),
]

# ── Deep1M Linux-B ──────────────────────────────────────────────────────────
deep_linb_suco_default = (0.9962, 67)
deep_linb_flat         = (0.9997, 19)

deep_linb_suco_sweep = [
    (0.9711,  117, "α=0.02"),
    (0.9962,   68, "Ns=8, α=0.05 (default)"),
    (0.9992,   39, "α=0.10"),
    (0.9996,   26, "α=0.20"),
    (0.9755,   88, "Ns=4"),
    (0.9981,   50, "Ns=12"),
    (0.9887,   70, "nc=25"),
    (0.9982,   60, "nc=100"),
]
deep_linb_hnsw = [
    (0.8207, 8746), (0.9169, 5382), (0.9662, 3243),
    (0.9912, 1832), (0.9969,  981), (0.9989,  488), (0.9995,  248),
]
deep_linb_ivfflat = [
    (0.4887, 5286), (0.6613, 3102), (0.7972, 1716),
    (0.8952,  875), (0.9538,  484), (0.9837,  257),
    (0.9960,  134), (0.9988,   71), (0.9994,   36), (0.9997,   18),
]
deep_linb_ivfpq = [
    (0.1567, 13843), (0.1835, 8392), (0.1985, 5734),
    (0.2051,  4106), (0.2076, 2485), (0.2082, 1391),
    (0.2084,   742), (0.2083,  355), (0.2083,  185), (0.2083,  110),
]
deep_linb_opqpq = [
    (0.1724, 10487), (0.1989, 8020), (0.2188, 5349),
    (0.2285,  4132), (0.2313, 2506), (0.2315, 1421),
    (0.2318,   781), (0.2318,  410), (0.2318,  212),
]

# ── SIFT10M Mac (Mar 11 log — precise values, now includes IVFPQ+OPQ) ────────
sift10_mac_suco_default = (0.9983, 30)
sift10_mac_flat         = (1.0000, 286)

sift10_mac_suco_sweep = [
    (0.9924,  39, "α=0.02"),
    (0.9983,  31, "α=0.05 (default)"),
    (0.9992,  26, "α=0.10"),
    (0.9985,  19, "α=0.20"),
    (0.9975,  27, "Ns=4, hd=16"),
    (0.9974,  33, "Ns=16, hd=4"),
    (0.9968,  31, "nc=25"),
    (0.9985,  30, "nc=100"),
]
sift10_mac_hnsw = [
    # Note: ef=16 > ef=8 QPS artifact also present at 10M scale
    (0.8320, 32215), (0.9223, 65879), (0.9682, 42684),
    (0.9902, 25938), (0.9983, 14452), (0.9996,  7335), (0.9998,  3476),
]
sift10_mac_ivfflat = [
    (0.5114, 58685), (0.6727, 32007), (0.8071, 16888),
    (0.8985,  8727), (0.9562,  4504), (0.9824,  2364),
    (0.9940,  1190), (0.9987,   618), (0.9996,   323), (0.9999,   167),
]
sift10_mac_ivfpq = [
    (0.2956, 165395), (0.3565, 99808), (0.3979, 60336),
    (0.4222,  32059), (0.4357, 18536), (0.4410,  9726),
    (0.4430,   4947), (0.4433,  2533), (0.4433,  1269), (0.4433,  652),
]
sift10_mac_opqpq = [
    (0.2895, 159797), (0.3530, 101163), (0.3937, 58994),
    (0.4184,  33313), (0.4326,  18088), (0.4365,  9246),
    (0.4381,   4867), (0.4389,   2461), (0.4390,  1259), (0.4390,  632),
]

# ── Deep10M Mac (Mar 12 rebuild — real 10M index, ntotal=10M confirmed) ─────
deep10_mac_suco_default = (0.9982, 32)
deep10_mac_flat         = (0.9993, 181)

deep10_mac_suco_sweep = [
    (0.9907, 38, "α=0.02"),
    (0.9982, 32, "α=0.05 (default)"),
    (0.9988, 27, "α=0.10"),
    (0.9991, 20, "α=0.20"),
    (0.9885, 30, "Ns=4, hd=12"),
    (0.9989, 33, "Ns=12, hd=4"),
    (0.9948, 30, "nc=25"),
    (0.9992, 27, "nc=100"),
]
deep10_mac_hnsw = [
    (0.7346, 74326), (0.8492, 52906), (0.9253, 31674),
    (0.9665, 12389), (0.9854,  5150), (0.9926,  3000), (0.9946,  1508),
]
deep10_mac_ivfflat = [
    (0.5449, 26001), (0.7113,  5604), (0.8460,  2484),
    (0.9264,  1375), (0.9711,   714), (0.9907,   368),
    (0.9972,   191), (0.9990,    99), (0.9992,    51), (0.9992,    26),
]
# IVFPQ on Deep10M plateaus at R@1 ≈ 0.162 (severe PQ approximation failure for this dataset)
deep10_mac_ivfpq = [
    (0.1278, 82377), (0.1444, 39283), (0.1575, 16972),
    (0.1612,  8532), (0.1619,  4539), (0.1620,  2259),
    (0.1621,  1267), (0.1621,   675), (0.1621,   348), (0.1621,   177),
]
deep10_mac_opqpq = [
    (0.1379, 78187), (0.1571, 36194), (0.1682, 17331),
    (0.1723,  8371), (0.1737,  4673), (0.1737,  2447),
    (0.1738,  1277), (0.1738,   684), (0.1738,   347), (0.1738,   177),
]

# ── SpaceV10M Mac ────────────────────────────────────────────────────────────
spacev_mac_suco_default = (0.9292, 29)
# (no baselines available for SpaceV10M from logs)

# ─────────────────────────────────────────────────────────────────────────────
# ██████  FIG 1: Recall–QPS Pareto  (Mac) — all 1M datasets + SIFT10M
# ─────────────────────────────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
fig1.suptitle(
    "Recall–QPS Pareto Curves  |  Mac (Apple Silicon, 24 GB, 10 threads)",
    fontsize=13, fontweight="bold", y=1.01,
)

datasets_1m_mac = [
    ("SIFT1M  (d=128, N=1M)", axes1[0],
     sift_mac_hnsw, sift_mac_ivfflat, sift_mac_ivfpq, sift_mac_opqpq,
     sift_mac_suco_default, sift_mac_flat, 0.20),
    ("GIST1M  (d=960, N=1M)", axes1[1],
     gist_mac_hnsw, gist_mac_ivfflat, gist_mac_ivfpq, gist_mac_opqpq,
     gist_mac_suco_default, gist_mac_flat, 0.12),
    ("Deep1M  (d=96, N=1M)", axes1[2],
     deep_mac_hnsw, deep_mac_ivfflat, deep_mac_ivfpq, deep_mac_opqpq,
     deep_mac_suco_default, deep_mac_flat, 0.15),
]

for title, ax, hnsw, ivfflat, ivfpq, opqpq, suco_def, flat, xlo in datasets_1m_mac:
    setup_pareto_ax(ax, title, xlo=xlo, ylabel=(ax is axes1[0]))
    draw_pareto(ax, hnsw,    "HNSW")
    draw_pareto(ax, ivfflat, "IVFFlat")
    draw_pareto(ax, ivfpq,   "IVFPQ")
    draw_pareto(ax, opqpq,   "OPQ+IVFPQ")
    ax.scatter(*suco_def, s=200, color=COLORS["SuCo"], marker="o", zorder=7,
               label="SuCo (default)", edgecolors="black", linewidths=1.3)
    ax.axvline(flat[0], color=COLORS["FlatL2"], linestyle=":", linewidth=1.3, alpha=0.7)
    ax.scatter(*flat, s=160, color=COLORS["FlatL2"], marker="X", zorder=6,
               label="FlatL2 (exact)", edgecolors="black", linewidths=0.8)
    ax.legend(fontsize=8, loc="lower right")

fig1.tight_layout()
save(fig1, "01_pareto_mac_1M.png")

# ─────────────────────────────────────────────────────────────────────────────
# ██████  FIG 2: Recall–QPS Pareto  (Linux-B) — SIFT1M, GIST1M, Deep1M
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle(
    "Recall–QPS Pareto Curves  |  Linux (16 GB, 8 threads)",
    fontsize=13, fontweight="bold", y=1.01,
)

datasets_1m_linb = [
    ("SIFT1M  (d=128, N=1M)", axes2[0],
     sift_linb_hnsw, sift_linb_ivfflat, sift_linb_ivfpq, [],
     sift_linb_suco_default, sift_linb_flat, 0.20),
    ("GIST1M  (d=960, N=1M)", axes2[1],
     gist_linb_hnsw, gist_linb_ivfflat, gist_linb_ivfpq, [],
     gist_linb_suco_default, gist_linb_flat, 0.12),
    ("Deep1M  (d=96, N=1M)", axes2[2],
     deep_linb_hnsw, deep_linb_ivfflat, deep_linb_ivfpq, deep_linb_opqpq,
     deep_linb_suco_default, deep_linb_flat, 0.15),
]

for title, ax, hnsw, ivfflat, ivfpq, opqpq, suco_def, flat, xlo in datasets_1m_linb:
    setup_pareto_ax(ax, title, xlo=xlo, ylabel=(ax is axes2[0]))
    draw_pareto(ax, hnsw,    "HNSW")
    draw_pareto(ax, ivfflat, "IVFFlat")
    if ivfpq:
        draw_pareto(ax, ivfpq,   "IVFPQ")
    if opqpq:
        draw_pareto(ax, opqpq,   "OPQ+IVFPQ")
    ax.scatter(*suco_def, s=200, color=COLORS["SuCo"], marker="o", zorder=7,
               label="SuCo (default)", edgecolors="black", linewidths=1.3)
    ax.axvline(flat[0], color=COLORS["FlatL2"], linestyle=":", linewidth=1.3, alpha=0.7)
    ax.scatter(*flat, s=160, color=COLORS["FlatL2"], marker="X", zorder=6,
               label="FlatL2 (exact)", edgecolors="black", linewidths=0.8)
    ax.legend(fontsize=8, loc="lower right")

fig2.tight_layout()
save(fig2, "02_pareto_linux_1M.png")

# ─────────────────────────────────────────────────────────────────────────────
# ██████  FIG 3: SIFT10M + SpaceV10M  (Mac)
# ─────────────────────────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
fig3.suptitle(
    "Recall–QPS  |  10M-scale datasets  (Mac, 24 GB, 10 threads)",
    fontsize=13, fontweight="bold", y=1.01,
)

ax3a = axes3[0]
setup_pareto_ax(ax3a, "SIFT10M  (d=128, N=10M)", xlo=0.25)
draw_pareto(ax3a, sift10_mac_hnsw,    "HNSW")
draw_pareto(ax3a, sift10_mac_ivfflat, "IVFFlat")
draw_pareto(ax3a, sift10_mac_ivfpq,   "IVFPQ")
draw_pareto(ax3a, sift10_mac_opqpq,   "OPQ+IVFPQ")
ax3a.scatter(*sift10_mac_suco_default, s=200, color=COLORS["SuCo"], marker="o", zorder=7,
             label="SuCo (default)", edgecolors="black", linewidths=1.3)
ax3a.scatter(*sift10_mac_flat, s=160, color=COLORS["FlatL2"], marker="X", zorder=6,
             label="FlatL2 (exact)", edgecolors="black", linewidths=0.8)
ax3a.legend(fontsize=9, loc="lower right")

# Deep10M — full baselines available
ax3b = axes3[1]
setup_pareto_ax(ax3b, "Deep10M  (d=96, N=10M)", xlo=0.10, ylabel=False)
draw_pareto(ax3b, deep10_mac_hnsw,    "HNSW")
draw_pareto(ax3b, deep10_mac_ivfflat, "IVFFlat")
draw_pareto(ax3b, deep10_mac_ivfpq,   "IVFPQ")
draw_pareto(ax3b, deep10_mac_opqpq,   "OPQ+IVFPQ")
ax3b.scatter(*deep10_mac_suco_default, s=200, color=COLORS["SuCo"], marker="o", zorder=7,
             label="SuCo (default)", edgecolors="black", linewidths=1.3)
ax3b.scatter(*deep10_mac_flat, s=160, color=COLORS["FlatL2"], marker="X", zorder=6,
             label="FlatL2 (exact)", edgecolors="black", linewidths=0.8)
ax3b.legend(fontsize=9, loc="lower right")

# SpaceV10M — only SuCo result available
ax3c = axes3[2]
ax3c.set_facecolor("#ffffff")
ax3c.grid(True, linestyle="--", linewidth=0.4, alpha=0.55)
ax3c.bar(["SuCo (default)"], [spacev_mac_suco_default[1]],
         color=COLORS["SuCo"], edgecolor="black", linewidth=0.8, width=0.4)
ax3c.set_ylabel("QPS", fontsize=10)
ax3c.set_title("SpaceV10M  (d=100, N=10M)\nSuCo only (baselines not run)", fontsize=11, fontweight="bold")
ax3c.text(0, spacev_mac_suco_default[1] + 0.5, f"R@1 = {spacev_mac_suco_default[0]:.4f}", ha="center", fontsize=10)
ax3c.set_ylim(0, spacev_mac_suco_default[1] * 2)

fig3.tight_layout()
save(fig3, "03_pareto_mac_10M.png")

# ─────────────────────────────────────────────────────────────────────────────
# ██████  FIG 4: Parameter Sweeps — α / Ns / nc  (Mac)
# ─────────────────────────────────────────────────────────────────────────────
fig4, axes4 = plt.subplots(2, 2, figsize=(13, 12))
axes4 = axes4.flatten()
fig4.suptitle(
    "SuCo Parameter Sweep  |  Mac (Apple Silicon)",
    fontsize=13, fontweight="bold", y=1.01,
)

SWEEP_COLORS = {
    "α": "#e63946", "Ns": "#457b9d", "nc": "#6ab187", "default": "#f4a261",
}

def plot_sweep_ax(ax, title, sweep_pts, xlo=0.80):
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.55)
    ax.set_xlabel("Recall@1", fontsize=10)
    ax.set_ylabel("QPS", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(xlo, 1.002)
    for r, q, lbl in sweep_pts:
        is_default = "default" in lbl
        if lbl.startswith("α"):
            c = SWEEP_COLORS["α"]
        elif lbl.startswith("Ns"):
            c = SWEEP_COLORS["Ns"]
        elif lbl.startswith("nc"):
            c = SWEEP_COLORS["nc"]
        else:
            c = SWEEP_COLORS["default"]
        ax.scatter(r, q, s=130 if is_default else 65,
                   color=COLORS["SuCo"] if is_default else c,
                   edgecolors="black", linewidths=1.2 if is_default else 0.5,
                   zorder=7 if is_default else 4)
        ax.annotate(lbl, (r, q), textcoords="offset points",
                    xytext=(5, 3), fontsize=7.5, color="#222222")
    # legend patches
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=COLORS["SuCo"],    label="Default"),
        mpatches.Patch(color=SWEEP_COLORS["α"], label="α sweep"),
        mpatches.Patch(color=SWEEP_COLORS["Ns"],label="Ns sweep"),
        mpatches.Patch(color=SWEEP_COLORS["nc"],label="nc sweep"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower left")

plot_sweep_ax(axes4[0], "SIFT1M  —  SuCo param sweep", sift_mac_suco_sweep, xlo=0.94)
plot_sweep_ax(axes4[1], "GIST1M  —  SuCo param sweep", gist_mac_suco_sweep, xlo=0.82)
plot_sweep_ax(axes4[2], "SIFT10M —  SuCo param sweep", sift10_mac_suco_sweep, xlo=0.985)
plot_sweep_ax(axes4[3], "Deep10M —  SuCo param sweep", deep10_mac_suco_sweep, xlo=0.985)

fig4.tight_layout()
save(fig4, "04_param_sweep_mac.png")

# ─────────────────────────────────────────────────────────────────────────────
# ██████  FIG 5: Parameter sweeps — Deep1M Linux-B
# ─────────────────────────────────────────────────────────────────────────────
fig5, ax5 = plt.subplots(1, 1, figsize=(7, 6))
fig5.suptitle(
    "Deep1M — SuCo Parameter Sweep  |  Linux (8 threads)",
    fontsize=12, fontweight="bold",
)
plot_sweep_ax(ax5, "Deep1M  (d=96, N=1M)", deep_linb_suco_sweep, xlo=0.96)
fig5.tight_layout()
save(fig5, "05_param_sweep_deep1m_linux.png")

# ─────────────────────────────────────────────────────────────────────────────
# ██████  FIG 6: Cross-platform QPS comparison
# ─────────────────────────────────────────────────────────────────────────────
fig6, axes6 = plt.subplots(1, 3, figsize=(18, 6))
fig6.suptitle(
    "Mac vs Linux  —  QPS at default SuCo config  (recall values are identical)",
    fontsize=12, fontweight="bold", y=1.01,
)

cross_data = {
    "SIFT1M": {
        "SuCo":    (335, 65),
        "FlatL2":  (2834, 18),
        "HNSW\nef=128":   (15855, 973),
        "IVFFlat\nnp=64": (3442, 104),
    },
    "GIST1M": {
        "SuCo":    (186, 25),
        "FlatL2":  (640, 10),
        "HNSW\nef=512":   (891, 71),
        "IVFFlat\nnp=64": (259, 9),
    },
    "Deep1M": {
        "SuCo":    (325, 67),
        "FlatL2":  (3058, 19),
        "HNSW\nef=128":    (None, 981),  # Mac HNSW sweep not in logs
        "IVFFlat\nnp=64":  (4954, 134),
    },
}

method_colors_xp = {
    "SuCo":           COLORS["SuCo"],
    "FlatL2":         COLORS["FlatL2"],
    "HNSW\nef=128":   COLORS["HNSW"],
    "HNSW\nef=512":   COLORS["HNSW"],
    "IVFFlat\nnp=64": COLORS["IVFFlat"],
}

for ax, (ds, methods) in zip(axes6, cross_data.items()):
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.55)
    ax.set_yscale("log")
    labels = list(methods.keys())
    x = np.arange(len(labels))
    w = 0.35
    mac_vals = [methods[m][0] for m in labels]
    lin_vals = [methods[m][1] for m in labels]

    bars_mac = []
    bars_lin = []
    for i, (m, mv, lv) in enumerate(zip(labels, mac_vals, lin_vals)):
        c = method_colors_xp.get(m, "#999")
        if mv is not None:
            b = ax.bar(i - w/2, mv, w, color=c, alpha=0.95,
                       edgecolor="black", linewidth=0.7)
            bars_mac.append(b)
            ax.text(i - w/2, mv * 1.07, f"{mv:,}", ha="center", va="bottom",
                    fontsize=7, rotation=45)
        if lv is not None:
            b = ax.bar(i + w/2, lv, w, color=c, alpha=0.55,
                       edgecolor="black", linewidth=0.7, hatch="///")
            bars_lin.append(b)
            ax.text(i + w/2, lv * 1.07, f"{lv:,}", ha="center", va="bottom",
                    fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("QPS  (log scale)", fontsize=10)
    ax.set_title(ds, fontsize=11, fontweight="bold")

    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="#888", alpha=0.95, label="Mac (Apple Silicon)"),
        mpatches.Patch(color="#888", alpha=0.55, hatch="///", label="Linux (x86)"),
    ], fontsize=8.5, loc="lower right")

fig6.tight_layout()
save(fig6, "06_cross_platform_qps.png")

# ─────────────────────────────────────────────────────────────────────────────
# ██████  FIG 7: IVFPQ recall ceiling vs SuCo
# ─────────────────────────────────────────────────────────────────────────────
fig7, ax7 = plt.subplots(figsize=(8, 6))
fig7.suptitle(
    "Recall Ceiling: SuCo vs IVFPQ / OPQ+IVFPQ\n(all nprobe values)",
    fontsize=12, fontweight="bold",
)
ax7.grid(True, linestyle="--", linewidth=0.4, alpha=0.55)
ax7.set_xlabel("nprobe", fontsize=10)
ax7.set_ylabel("Recall@1", fontsize=10)
ax7.set_xscale("log")

nprobes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

def pad(lst, n=10):
    return lst + [lst[-1]] * (n - len(lst))

sift_ivfpq_r = [p[0] for p in sift_mac_ivfpq]
gist_ivfpq_r = [0.1380, 0.1610, 0.1870, 0.2010, 0.2050, 0.2060, 0.2060, 0.2060, 0.2060, 0.2190]
deep_ivfpq_r = [0.1583, 0.1835, 0.1985, 0.2051, 0.2076, 0.2082, 0.2084, 0.2083, 0.2083, 0.2083]

sift_opqpq_r = [p[0] for p in sift_mac_opqpq]
gist_opqpq_r = [0.1800, 0.2550, 0.3500, 0.4200, 0.4400, 0.4420, 0.4420, 0.4420, 0.4420, 0.4420]
deep_opqpq_r = [0.1724, 0.1989, 0.2188, 0.2285, 0.2313, 0.2315, 0.2318, 0.2318, 0.2318, 0.2318]

ax7.plot(nprobes[:len(sift_ivfpq_r)], sift_ivfpq_r, "o--", color="#e63946",   label="IVFPQ SIFT1M", linewidth=1.5)
ax7.plot(nprobes[:len(gist_ivfpq_r)], gist_ivfpq_r, "s--", color="#e63946",   label="IVFPQ GIST1M",  linewidth=1.5, alpha=0.6)
ax7.plot(nprobes[:len(deep_ivfpq_r)], deep_ivfpq_r, "D--", color="#e63946",   label="IVFPQ Deep1M",  linewidth=1.5, alpha=0.3)
ax7.plot(nprobes[:len(sift_opqpq_r)], sift_opqpq_r, "o-",  color="#457b9d",  label="OPQ+IVFPQ SIFT1M", linewidth=1.5)
ax7.plot(nprobes[:len(gist_opqpq_r)], gist_opqpq_r, "s-",  color="#457b9d",  label="OPQ+IVFPQ GIST1M",  linewidth=1.5, alpha=0.6)
ax7.plot(nprobes[:len(deep_opqpq_r)], deep_opqpq_r, "D-",  color="#457b9d",  label="OPQ+IVFPQ Deep1M",  linewidth=1.5, alpha=0.3)

# SuCo horizontal reference lines
for ds, r, c in [("SuCo SIFT1M", 0.9895, "#f4a261"),
                 ("SuCo GIST1M", 0.9750, "#e76f51"),
                 ("SuCo Deep1M", 0.9961, "#c1440e")]:
    ax7.axhline(r, linestyle=":", linewidth=1.8, color=c, label=ds)

ax7.set_ylim(0.10, 1.02)
ax7.legend(fontsize=8.5, ncol=2)

fig7.tight_layout()
save(fig7, "07_ivfpq_recall_ceiling.png")

# ─────────────────────────────────────────────────────────────────────────────
# ██████  FIG 8: α sweep — QPS vs recall tradeoff curve (all datasets)
# ─────────────────────────────────────────────────────────────────────────────
fig8, axes8 = plt.subplots(1, 2, figsize=(14, 6))
fig8.suptitle(
    "α (collision_ratio) Sweep  —  QPS vs Recall@1 tradeoff",
    fontsize=12, fontweight="bold", y=1.01,
)

alpha_data_mac = {
    "SIFT1M (Mac)": [
        (0.020, 0.9595, 416), (0.050, 0.9895, 335),
        (0.100, 0.9909, 259), (0.200, 0.9928, 172),
    ],
    "GIST1M (Mac)": [
        (0.020, 0.8510, 274), (0.050, 0.9750, 183),
        (0.100, 0.9880, 109), (0.200, 0.9890,  60),
    ],
    "SIFT10M (Mac)": [
        (0.020, 0.9924, 37), (0.050, 0.9983, 30),
        (0.100, 0.9992, 25), (0.200, 0.9985, 18),
    ],
}

alpha_data_linux = {
    "SIFT1M (Linux)": [
        (0.020, 0.9596, 118), (0.050, 0.9892,  66),
        (0.100, 0.9903,  39), (0.200, 0.9934,  24),
    ],
    "GIST1M (Linux)": [
        (0.020, 0.8540, 38), (0.050, 0.9750, 23),
        (0.100, 0.9940, 14), (0.200, 0.9910,  8),
    ],
    "Deep1M (Linux)": [
        (0.020, 0.9711, 117), (0.050, 0.9962, 68),
        (0.100, 0.9992,  39), (0.200, 0.9996, 26),
    ],
}

alpha_colors = ["#e63946", "#457b9d", "#6ab187"]
alpha_vals   = [0.020, 0.050, 0.100, 0.200]

for ax, title, data in [(axes8[0], "Mac", alpha_data_mac),
                        (axes8[1], "Linux (8 threads)", alpha_data_linux)]:
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.55)
    ax.set_xlabel("Recall@1", fontsize=10)
    ax.set_ylabel("QPS", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for (label, pts), c in zip(data.items(), alpha_colors):
        alphas, recalls, qps_vals = zip(*pts)
        ax.plot(recalls, qps_vals, "o-", color=c, linewidth=2, markersize=7, label=label)
        for a, r, q in pts:
            ax.annotate(f"α={a}", (r, q), textcoords="offset points",
                        xytext=(4, 3), fontsize=7.5, color=c)
    ax.legend(fontsize=9)

fig8.tight_layout()
save(fig8, "08_alpha_sweep.png")

# ─────────────────────────────────────────────────────────────────────────────
# ██████  FIG 9: HNSW efSearch sweep — Mac vs Linux
# ─────────────────────────────────────────────────────────────────────────────
fig9, axes9 = plt.subplots(1, 3, figsize=(18, 6))
fig9.suptitle(
    "HNSW efSearch Sweep  —  Mac vs Linux  (M=32, efConstruction=200)",
    fontsize=12, fontweight="bold", y=1.01,
)

hnsw_cross = [
    ("SIFT1M",  axes9[0], sift_mac_hnsw,  sift_linb_hnsw,
     sift_mac_suco_default,  sift_linb_suco_default,  0.80),
    ("GIST1M",  axes9[1], gist_mac_hnsw,  gist_linb_hnsw,
     gist_mac_suco_default,  gist_linb_suco_default,  0.45),
    ("Deep1M",  axes9[2], deep_mac_hnsw,  deep_linb_hnsw,
     deep_mac_suco_default,  deep_linb_suco_default,  0.80),
]

for ds, ax, mac_pts, lin_pts, suco_mac, suco_lin, xlo in hnsw_cross:
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.55)
    ax.set_xlabel("Recall@1", fontsize=10)
    ax.set_ylabel("QPS  (log scale)", fontsize=10)
    ax.set_title(ds, fontsize=11, fontweight="bold")
    ax.set_xlim(xlo, 1.002)

    r_mac, q_mac = unzip(mac_pts)
    r_lin, q_lin = unzip(lin_pts)
    ax.plot(r_mac, q_mac, "^-", color=COLORS["HNSW"],  linewidth=2,
            markersize=6, label="HNSW (Mac)", alpha=0.9)
    ax.plot(r_lin, q_lin, "^--", color=COLORS["HNSW"], linewidth=1.5,
            markersize=5, label="HNSW (Linux)", alpha=0.6)

    ax.scatter(*suco_mac, s=180, color=COLORS["SuCo"], marker="o", zorder=7,
               edgecolors="black", linewidths=1.2, label="SuCo (Mac)")
    ax.scatter(*suco_lin, s=100, color=COLORS["SuCo"], marker="o", zorder=7,
               edgecolors="black", linewidths=0.8, alpha=0.6, label="SuCo (Linux)")
    ax.legend(fontsize=8.5)

fig9.tight_layout()
save(fig9, "09_hnsw_sweep_cross_platform.png")

# ─────────────────────────────────────────────────────────────────────────────
# ██████  FIG 10: Scaling  (1M → 10M) — SuCo vs HNSW
# ─────────────────────────────────────────────────────────────────────────────
fig10, axes10 = plt.subplots(1, 2, figsize=(13, 6))
fig10.suptitle(
    "Scaling: 1M → 10M Vectors  (Mac)  |  SuCo vs HNSW  (SIFT dataset, d=128)",
    fontsize=12, fontweight="bold", y=1.01,
)

ax10a = axes10[0]
ax10a.grid(True, linestyle="--", linewidth=0.4, alpha=0.55)
ax10a.set_xlabel("Recall@1", fontsize=10)
ax10a.set_ylabel("QPS  (log scale)", fontsize=10)
ax10a.set_title("Recall–QPS  (SIFT1M vs SIFT10M)", fontsize=11, fontweight="bold")
ax10a.set_yscale("log")
ax10a.set_xlim(0.78, 1.002)

sift1m_hnsw_r, sift1m_hnsw_q   = unzip(sift_mac_hnsw)
sift10m_hnsw_r, sift10m_hnsw_q = unzip(sift10_mac_hnsw)
sift1m_ivff_r, sift1m_ivff_q   = unzip(sift_mac_ivfflat)
sift10m_ivff_r, sift10m_ivff_q = unzip(sift10_mac_ivfflat)

ax10a.plot(sift1m_hnsw_r,  sift1m_hnsw_q,  "^-",  color=COLORS["HNSW"],    linewidth=2, markersize=6, label="HNSW SIFT1M")
ax10a.plot(sift10m_hnsw_r, sift10m_hnsw_q, "^--", color=COLORS["HNSW"],    linewidth=1.5, markersize=5, alpha=0.6, label="HNSW SIFT10M")
ax10a.plot(sift1m_ivff_r,  sift1m_ivff_q,  "s-",  color=COLORS["IVFFlat"], linewidth=2, markersize=6, label="IVFFlat SIFT1M")
ax10a.plot(sift10m_ivff_r, sift10m_ivff_q, "s--", color=COLORS["IVFFlat"], linewidth=1.5, markersize=5, alpha=0.6, label="IVFFlat SIFT10M")
ax10a.scatter(*sift_mac_suco_default,  s=180, color=COLORS["SuCo"], zorder=7,
              edgecolors="black", linewidths=1.2, label="SuCo SIFT1M")
ax10a.scatter(*sift10_mac_suco_default, s=100, color=COLORS["SuCo"], zorder=7,
              edgecolors="black", linewidths=0.8, alpha=0.6, label="SuCo SIFT10M")
ax10a.scatter(*sift_mac_flat,  s=160, color=COLORS["FlatL2"], marker="X", zorder=6,
              edgecolors="black", linewidths=0.8, label="FlatL2 SIFT1M")
ax10a.scatter(*sift10_mac_flat, s=100, color=COLORS["FlatL2"], marker="X", zorder=6,
              edgecolors="black", linewidths=0.6, alpha=0.6, label="FlatL2 SIFT10M")
ax10a.legend(fontsize=7.5, ncol=2)

# QPS at matched R@1 ≈ 0.998 as N scales
ax10b = axes10[1]
ax10b.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.55)
ax10b.set_title("QPS at R@1 ≈ 0.998  (efSearch=128 for HNSW)", fontsize=11, fontweight="bold")
ax10b.set_ylabel("QPS  (log scale)", fontsize=10)
ax10b.set_yscale("log")

scale_labels = ["SIFT1M", "SIFT10M"]
scale_suco  = [335, 29]
scale_hnsw  = [15855, 11957]
scale_ivff  = [935, 628]   # nprobe that gives R@1 ≈ 0.999

x = np.arange(2)
w = 0.25
for i, (vals, name) in enumerate([(scale_suco, "SuCo"),
                                    (scale_hnsw, "HNSW"),
                                    (scale_ivff, "IVFFlat")]):
    bars = ax10b.bar(x + (i-1)*w, vals, w, label=name,
                     color=COLORS[name], edgecolor="black", linewidth=0.7)
    for bar, v in zip(bars, vals):
        ax10b.text(bar.get_x() + bar.get_width()/2, v*1.1,
                   f"{v:,}", ha="center", va="bottom", fontsize=8, rotation=45)

ax10b.set_xticks(x)
ax10b.set_xticklabels(scale_labels, fontsize=10)
ax10b.legend(fontsize=9)
# Annotate the 10× N scaling factor
ax10b.annotate("10× more vectors →\nHNSW: ~1.3× slower\nSuCo: ~11× slower",
               xy=(1, 30), fontsize=8.5, color="#333",
               bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", alpha=0.85))

fig10.tight_layout()
save(fig10, "10_scaling_1M_to_10M.png")

# ─────────────────────────────────────────────────────────────────────────────
# ██████  FIG 11: Build time & index size comparison
# ─────────────────────────────────────────────────────────────────────────────
fig11, axes11 = plt.subplots(1, 2, figsize=(13, 6))
fig11.suptitle(
    "Build Time & Index Size Comparison",
    fontsize=12, fontweight="bold", y=1.01,
)

datasets_bt = ["SIFT1M", "GIST1M", "Deep1M", "SIFT10M"]
suco_build = [2.5, 21.4, 7.9, 90.0]     # seconds (Mac; 10M is ~1.5min)
hnsw_build = [5.3*60, 18.6*60, 23.1*60, 29.1*60]  # seconds

ax11a = axes11[0]
ax11a.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.55)
x = np.arange(len(datasets_bt))
w = 0.35
b1 = ax11a.bar(x - w/2, suco_build, w, label="SuCo",
               color=COLORS["SuCo"], edgecolor="black", linewidth=0.7)
b2 = ax11a.bar(x + w/2, hnsw_build, w, label="HNSW",
               color=COLORS["HNSW"], edgecolor="black", linewidth=0.7)
for bar, v in zip(b1, suco_build):
    ax11a.text(bar.get_x() + bar.get_width()/2, v + 2,
               f"{v:.0f}s", ha="center", va="bottom", fontsize=8)
for bar, v in zip(b2, hnsw_build):
    ax11a.text(bar.get_x() + bar.get_width()/2, v + 2,
               f"{v/60:.1f}m", ha="center", va="bottom", fontsize=8)
ax11a.set_xticks(x)
ax11a.set_xticklabels(datasets_bt, fontsize=10)
ax11a.set_ylabel("Build time  (seconds, log scale)", fontsize=10)
ax11a.set_yscale("log")
ax11a.set_title("Build Time: SuCo vs HNSW", fontsize=11, fontweight="bold")
ax11a.legend(fontsize=9)

# Index size
datasets_sz = ["SIFT1M", "GIST1M", "Deep1M", "SIFT10M", "SpaceV10M"]
suco_sz  = [549.6, 3969, 427.6, 5494, 4578]
raw_sz   = [1e6*128*4/2**20, 1e6*960*4/2**20, 1e6*96*4/2**20,
            10e6*128*4/2**20, 10e6*100*4/2**20]

ax11b = axes11[1]
ax11b.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.55)
x2 = np.arange(len(datasets_sz))
b3 = ax11b.bar(x2 - w/2, suco_sz, w, label="SuCo index",
               color=COLORS["SuCo"], edgecolor="black", linewidth=0.7)
b4 = ax11b.bar(x2 + w/2, raw_sz, w, label="Raw float32 data",
               color="#cccccc", edgecolor="black", linewidth=0.7)
for bar, v in zip(b3, suco_sz):
    ax11b.text(bar.get_x() + bar.get_width()/2, v*1.03,
               f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
for bar, v in zip(b4, raw_sz):
    ax11b.text(bar.get_x() + bar.get_width()/2, v*1.03,
               f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
ax11b.set_xticks(x2)
ax11b.set_xticklabels(datasets_sz, fontsize=9.5)
ax11b.set_ylabel("Memory  (MiB)", fontsize=10)
ax11b.set_title("SuCo Index Size vs Raw Data", fontsize=11, fontweight="bold")
ax11b.legend(fontsize=9)

fig11.tight_layout()
save(fig11, "11_build_time_index_size.png")

# ─────────────────────────────────────────────────────────────────────────────
# ██████  FIG 12: GIST1M deep-dive  (hardest dataset)
# ─────────────────────────────────────────────────────────────────────────────
fig12, axes12 = plt.subplots(1, 2, figsize=(14, 6))
fig12.suptitle(
    "GIST1M Deep-Dive  (d=960, N=1M)  —  High-dimensional challenge",
    fontsize=12, fontweight="bold", y=1.01,
)

# Ns sweep effect on recall
ax12a = axes12[0]
ax12a.grid(True, linestyle="--", linewidth=0.4, alpha=0.55)
ns_vals  = [8, 16, 24, 40, 60]
ns_recall_mac  = [0.8900, 0.9450, 0.9610, 0.9750, 0.9730]
ns_recall_lin  = [0.8900, 0.9420, 0.9600, 0.9750, 0.9740]
ns_qps_mac     = [174, 189, 185, 183, 158]
ns_qps_lin     = [42,  32,  28,  23,  18]

ax12a.plot(ns_vals, ns_recall_mac, "o-", color="#e63946", linewidth=2, markersize=7, label="R@1 (Mac)")
ax12a.plot(ns_vals, ns_recall_lin, "o--", color="#e63946", linewidth=1.5, markersize=5, alpha=0.6, label="R@1 (Linux)")
ax12a.set_xlabel("nsubspaces (Ns)", fontsize=10)
ax12a.set_ylabel("Recall@1", fontsize=10, color="#e63946")
ax12a.tick_params(axis="y", labelcolor="#e63946")
ax12a.set_title("Effect of Ns on GIST1M recall", fontsize=11, fontweight="bold")

ax12a_r = ax12a.twinx()
ax12a_r.plot(ns_vals, ns_qps_mac, "s-", color="#457b9d", linewidth=2, markersize=7, label="QPS (Mac)")
ax12a_r.plot(ns_vals, ns_qps_lin, "s--", color="#457b9d", linewidth=1.5, markersize=5, alpha=0.6, label="QPS (Linux)")
ax12a_r.set_ylabel("QPS", fontsize=10, color="#457b9d")
ax12a_r.tick_params(axis="y", labelcolor="#457b9d")

lines1, labs1 = ax12a.get_legend_handles_labels()
lines2, labs2 = ax12a_r.get_legend_handles_labels()
ax12a.legend(lines1 + lines2, labs1 + labs2, fontsize=8.5, loc="lower right")

# Max achievable recall by method (GIST1M Mac)
ax12b = axes12[1]
ax12b.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.55)
methods_gist = ["FlatL2\n(exact)", "SuCo\n(α=0.05)", "SuCo\n(α=0.10)", "HNSW\n(ef=512)",
                "IVFFlat\n(np=512)", "OPQ+\nIVFPQ", "IVFPQ"]
max_r1_gist  = [0.9940,  0.9750,          0.9880,         0.9800,
                0.9940,              0.4420,      0.2190]
bar_colors = [COLORS["FlatL2"], COLORS["SuCo"], COLORS["SuCo"],
              COLORS["HNSW"], COLORS["IVFFlat"], COLORS["OPQ+IVFPQ"], COLORS["IVFPQ"]]
bars = ax12b.bar(methods_gist, max_r1_gist, color=bar_colors,
                 edgecolor="black", linewidth=0.7, width=0.6)
for bar, v in zip(bars, max_r1_gist):
    ax12b.text(bar.get_x() + bar.get_width()/2, v + 0.005,
               f"{v:.3f}", ha="center", va="bottom", fontsize=9)
ax12b.set_ylim(0, 1.06)
ax12b.set_ylabel("Max achievable Recall@1", fontsize=10)
ax12b.set_title("Maximum Recall@1 by Method  (GIST1M, Mac)", fontsize=11, fontweight="bold")
ax12b.axhline(0.9940, color=COLORS["FlatL2"], linestyle=":", linewidth=1.2, alpha=0.5)

fig12.tight_layout()
save(fig12, "12_gist1m_deep_dive.png")


# ─────────────────────────────────────────────────────────────────────────────
# ██████  LOAD pareto_sweep.tsv  (fine-grained α / HNSW / IVFFlat sweep)
# Run once by bench_suco_pareto_sweep.py — gives dense Pareto curves
# ─────────────────────────────────────────────────────────────────────────────
import csv as _csv
import pathlib as _pathlib

_tsv_path = _pathlib.Path(__file__).parent / "pareto_sweep.tsv"
_tsv_cache: dict = {}
if _tsv_path.exists():
    with open(_tsv_path) as _f:
        _rdr = _csv.DictReader(_f, delimiter="\t")
        for _row in _rdr:
            _key = (_row["dataset"], _row["method"])
            _tsv_cache.setdefault(_key, []).append(
                (float(_row["recall1"]), float(_row["qps"]))
            )

def tsv_pareto(dataset: str, method: str):
    """Return Pareto-filtered (recall, QPS) list from the TSV, or []."""
    pts = _tsv_cache.get((dataset, method), [])
    return pareto(pts) if pts else []

def draw_tsv(ax, dataset, method, fallback=None, lw=2, ms=4, **kw):
    pts = tsv_pareto(dataset, method)
    if pts:
        r, q = unzip(pts)
    elif fallback:
        pf = pareto(fallback)
        r, q = unzip(pf)
    else:
        return
    ax.plot(r, q, marker=MARKERS.get(method, "o"), markersize=ms, linewidth=lw,
            color=COLORS.get(method, "#888"), label=method, alpha=0.9, zorder=3, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# ██████  FIG 13: Paper Main Result — SuCo as proper Pareto curve (Mac, 1M)
# ─────────────────────────────────────────────────────────────────────────────
fig13, axes13 = plt.subplots(1, 3, figsize=(18, 6))
fig13.suptitle(
    "Recall–QPS Pareto Curves  |  SuCo as full Pareto frontier  (Mac, Apple Silicon, 10 threads)",
    fontsize=12, fontweight="bold", y=1.01,
)

_paper_1m = [
    ("SIFT1M  (d=128, N=1 M)",  "SIFT1M", axes13[0],
     sift_mac_ivfpq, sift_mac_opqpq, sift_mac_suco_default, sift_mac_flat, 0.65),
    ("GIST1M  (d=960, N=1 M)",  "GIST1M", axes13[1],
     gist_mac_ivfpq, gist_mac_opqpq, gist_mac_suco_default, gist_mac_flat, 0.40),
    ("Deep1M  (d=96,  N=1 M)",  "Deep1M", axes13[2],
     deep_mac_ivfpq, deep_mac_opqpq, deep_mac_suco_default, deep_mac_flat, 0.65),
]

for _title, _ds, _ax, _ivfpq, _opqpq, _suco_def, _flat, _xlo in _paper_1m:
    setup_pareto_ax(_ax, _title, xlo=_xlo, ylabel=(_ax is axes13[0]))
    draw_tsv(_ax, _ds, "HNSW")
    draw_tsv(_ax, _ds, "IVFFlat")
    draw_tsv(_ax, _ds, "SuCo")
    # IVFPQ / OPQ+IVFPQ as thin dashed reference lines
    for _pts, _name in [(_ivfpq, "IVFPQ"), (_opqpq, "OPQ+IVFPQ")]:
        if _pts:
            _pf = pareto(_pts)
            _r, _q = unzip(_pf)
            _ax.plot(_r, _q, color=COLORS[_name], linewidth=1.0,
                     linestyle="--", marker=MARKERS[_name], markersize=3,
                     alpha=0.55, label=_name)
    # Default operating point (star)
    _ax.scatter(*_suco_def, s=250, color=COLORS["SuCo"], marker="*", zorder=9,
               edgecolors="black", linewidths=1.2, label="SuCo α=0.05")
    # FlatL2 reference
    _ax.axvline(_flat[0], color=COLORS["FlatL2"], linestyle=":", linewidth=1.3, alpha=0.6)
    _ax.scatter(*_flat, s=150, color=COLORS["FlatL2"], marker="X", zorder=6,
               edgecolors="black", linewidths=0.8, label="FlatL2 (exact)")
    _ax.legend(fontsize=7.5, loc="lower right")

fig13.tight_layout()
save(fig13, "13_paper_pareto_suco_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# ██████  FIG 14: Recall Ceiling — Compressed vs Uncompressed
# ─────────────────────────────────────────────────────────────────────────────
fig14, ax14 = plt.subplots(figsize=(14, 7))
fig14.suptitle(
    "Maximum Achievable Recall@1  |  Compressed vs Uncompressed Methods",
    fontsize=13, fontweight="bold",
)

_rc_methods = ["FlatL2", "SuCo", "HNSW", "IVFFlat", "OPQ+IVFPQ", "IVFPQ"]
_rc_datasets = ["SIFT1M", "GIST1M", "Deep1M", "SIFT10M", "Deep10M"]
_rc_colors   = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

# Max R@1 per (method, dataset)
_max_r1 = {
    ("FlatL2",    "SIFT1M"):  0.9914,  ("FlatL2",    "GIST1M"):  0.9940,
    ("FlatL2",    "Deep1M"):  0.9997,  ("FlatL2",    "SIFT10M"): 1.0000,
    ("FlatL2",    "Deep10M"): 0.9993,
    ("SuCo",      "SIFT1M"):  0.9928,  ("SuCo",      "GIST1M"):  0.9890,
    ("SuCo",      "Deep1M"):  0.9996,  ("SuCo",      "SIFT10M"): 0.9992,
    ("SuCo",      "Deep10M"): 0.9991,
    ("HNSW",      "SIFT1M"):  0.9913,  ("HNSW",      "GIST1M"):  0.9820,
    ("HNSW",      "Deep1M"):  0.9995,  ("HNSW",      "SIFT10M"): 0.9998,
    ("HNSW",      "Deep10M"): 0.9946,
    ("IVFFlat",   "SIFT1M"):  0.9914,  ("IVFFlat",   "GIST1M"):  0.9940,
    ("IVFFlat",   "Deep1M"):  0.9997,  ("IVFFlat",   "SIFT10M"): 0.9999,
    ("IVFFlat",   "Deep10M"): 0.9992,
    ("OPQ+IVFPQ", "SIFT1M"):  0.2924,  ("OPQ+IVFPQ", "GIST1M"):  0.4420,
    ("OPQ+IVFPQ", "Deep1M"):  0.2353,  ("OPQ+IVFPQ", "SIFT10M"): 0.4390,
    ("OPQ+IVFPQ", "Deep10M"): 0.1738,
    ("IVFPQ",     "SIFT1M"):  0.2848,  ("IVFPQ",     "GIST1M"):  0.2190,
    ("IVFPQ",     "Deep1M"):  0.2059,  ("IVFPQ",     "SIFT10M"): 0.4433,
    ("IVFPQ",     "Deep10M"): 0.1621,
}

_x14 = np.arange(len(_rc_methods))
_w14 = 0.14
for _di, (_ds14, _dc14) in enumerate(zip(_rc_datasets, _rc_colors)):
    _vals14 = [_max_r1.get((_m, _ds14), None) for _m in _rc_methods]
    _offset = (_di - len(_rc_datasets) / 2 + 0.5) * _w14
    _bar_xs = []
    for _xi, _v in enumerate(_vals14):
        if _v is not None:
            ax14.bar(_xi + _offset, _v, _w14, color=_dc14, edgecolor="black",
                     linewidth=0.5, label=_ds14 if _xi == 0 else "", alpha=0.9)
            _bar_xs.append((_xi + _offset, _v))
    for _bx, _bv in _bar_xs:
        if _bv < 0.50:
            ax14.text(_bx, _bv + 0.010, f"{_bv:.2f}", ha="center", fontsize=6.5,
                      rotation=0, color="#222")

ax14.set_xticks(_x14)
ax14.set_xticklabels(_rc_methods, fontsize=11)
ax14.set_ylabel("Maximum achievable Recall@1", fontsize=11)
ax14.set_ylim(0, 1.09)
ax14.axhline(0.90, lw=0.8, ls="--", color="#888", alpha=0.55)
ax14.axhline(0.50, lw=0.6, ls=":",  color="#888", alpha=0.35)
ax14.grid(True, axis="y", ls="--", lw=0.4, alpha=0.5)
ax14.legend(title="Dataset", fontsize=9, title_fontsize=9, ncol=5,
            loc="lower center", bbox_to_anchor=(0.5, -0.13))
ax14.set_title(
    "Compressed PQ-based methods cap out at R@1 < 0.45  ─  "
    "SuCo / HNSW / IVFFlat all reach > 0.98",
    fontsize=9, style="italic",
)
fig14.tight_layout()
save(fig14, "14_recall_ceiling_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# ██████  FIG 15: 10-Recall@10 Quality Metric
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib.patches as _mp15

fig15, axes15 = plt.subplots(1, 2, figsize=(15, 6))
fig15.suptitle(
    "10-Recall@10: Ranked-List Quality  |  SuCo Parameter Sweeps (Mac)",
    fontsize=12, fontweight="bold", y=1.01,
)

# Panel a: scatter R@1 vs 10-R@10 for all sweep points (Mac)
ax15a = axes15[0]
ax15a.grid(True, linestyle="--", linewidth=0.4, alpha=0.55)
ax15a.set_xlabel("Recall@1", fontsize=10)
ax15a.set_ylabel("10-Recall@10", fontsize=10)
ax15a.set_title("R@1 vs 10-R@10  (all param-sweep points, Mac)",
               fontsize=11, fontweight="bold")

# (dataset, R@1, 10-R@10, is_default)
_scatter15 = [
    # SIFT1M sweep
    ("SIFT1M", 0.9766, 0.9688, False), ("SIFT1M", 0.9895, 0.9872, True),
    ("SIFT1M", 0.9896, 0.9827, False), ("SIFT1M", 0.9595, 0.9183, False),
    ("SIFT1M", 0.9909, 0.9964, False), ("SIFT1M", 0.9928, 0.9978, False),
    ("SIFT1M", 0.9820, 0.9709, False), ("SIFT1M", 0.9903, 0.9935, False),
    # GIST1M sweep
    ("GIST1M", 0.8900, 0.853, False), ("GIST1M", 0.9450, 0.920, False),
    ("GIST1M", 0.9610, 0.923, False), ("GIST1M", 0.9750, 0.947, True),
    ("GIST1M", 0.9730, 0.953, False), ("GIST1M", 0.8510, 0.765, False),
    ("GIST1M", 0.9880, 0.992, False), ("GIST1M", 0.9890, 0.998, False),
    ("GIST1M", 0.9400, 0.900, False), ("GIST1M", 0.9870, 0.973, False),
    # Deep1M sweep
    ("Deep1M", 0.9711, 0.9434, False), ("Deep1M", 0.9961, 0.9873, True),
    ("Deep1M", 0.9993, 0.9972, False), ("Deep1M", 0.9996, 0.9993, False),
    ("Deep1M", 0.9759, 0.9562, False), ("Deep1M", 0.9981, 0.9931, False),
    ("Deep1M", 0.9888, 0.9708, False), ("Deep1M", 0.9982, 0.9951, False),
    # SIFT10M sweep
    ("SIFT10M", 0.9924, 0.9805, False), ("SIFT10M", 0.9983, 0.9981, True),
    ("SIFT10M", 0.9992, 0.9993, False), ("SIFT10M", 0.9985, 0.9997, False),
    ("SIFT10M", 0.9975, 0.9951, False), ("SIFT10M", 0.9974, 0.9962, False),
    ("SIFT10M", 0.9968, 0.9947, False), ("SIFT10M", 0.9985, 0.9992, False),
    # Deep10M sweep
    ("Deep10M", 0.9907, 0.9792, False), ("Deep10M", 0.9982, 0.9967, True),
    ("Deep10M", 0.9988, 0.9990, False), ("Deep10M", 0.9991, 0.9995, False),
    ("Deep10M", 0.9885, 0.9798, False), ("Deep10M", 0.9989, 0.9984, False),
    ("Deep10M", 0.9948, 0.9892, False), ("Deep10M", 0.9992, 0.9987, False),
]

_ds15_c = {"SIFT1M": "#e63946", "GIST1M": "#457b9d", "Deep1M": "#6ab187",
           "SIFT10M": "#f4a261", "Deep10M": "#c77dff"}
_ds15_seen = set()
for _ds, _r1, _r10, _is_def in _scatter15:
    _lbl = _ds if _ds not in _ds15_seen else ""
    _ds15_seen.add(_ds)
    ax15a.scatter(_r1, _r10, s=120 if _is_def else 40,
                  color=_ds15_c[_ds], edgecolors="black",
                  linewidths=1.2 if _is_def else 0.3,
                  zorder=8 if _is_def else 4, alpha=0.95, label=_lbl)

_diag15 = np.linspace(0.70, 1.0, 100)
ax15a.plot(_diag15, _diag15, "k--", linewidth=0.8, alpha=0.4, label="R@1 = 10-R@10")
ax15a.set_xlim(0.72, 1.002)
ax15a.set_ylim(0.72, 1.002)
ax15a.legend(fontsize=8.5, loc="lower right")
ax15a.set_aspect("equal")

# Panel b: grouped bar — R@1, R@10, 10-R@10 at default config per dataset
ax15b = axes15[1]
ax15b.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.55)
ax15b.set_title("Three Recall Metrics at Default Config  (all datasets, Mac)",
               fontsize=11, fontweight="bold")
ax15b.set_ylabel("Metric value", fontsize=10)

_m15_data = {
    "SIFT1M":    {"R@1": 0.9895, "R@10": 0.9965, "10-R@10": 0.9872},
    "GIST1M":    {"R@1": 0.9750, "R@10": 0.9800, "10-R@10": 0.9467},
    "Deep1M":    {"R@1": 0.9961, "R@10": 0.9963, "10-R@10": 0.9873},
    "SIFT10M":   {"R@1": 0.9983, "R@10": 0.9995, "10-R@10": 0.9981},
    "Deep10M":   {"R@1": 0.9982, "R@10": 0.9986, "10-R@10": 0.9967},
    "SpaceV10M": {"R@1": 0.9292, "R@10": 0.9997, "10-R@10": 0.9967},
}
_ds15b = list(_m15_data.keys())
_x15b  = np.arange(len(_ds15b))
_w15b  = 0.27
ax15b.bar(_x15b - _w15b, [_m15_data[d]["R@1"]     for d in _ds15b], _w15b,
          color="#f4a261", edgecolor="black", lw=0.7, label="Recall@1")
ax15b.bar(_x15b,          [_m15_data[d]["R@10"]    for d in _ds15b], _w15b,
          color="#457b9d", edgecolor="black", lw=0.7, label="Recall@10")
ax15b.bar(_x15b + _w15b,  [_m15_data[d]["10-R@10"] for d in _ds15b], _w15b,
          color="#6ab187", edgecolor="black", lw=0.7, label="10-Recall@10")
ax15b.set_xticks(_x15b)
ax15b.set_xticklabels(_ds15b, fontsize=9.5)
ax15b.set_ylim(0.88, 1.005)
ax15b.legend(fontsize=9)

fig15.tight_layout()
save(fig15, "15_10recall10_quality.png")


# ─────────────────────────────────────────────────────────────────────────────
# ██████  FIG 16: N-Scaling — QPS as Database Size Grows
# ─────────────────────────────────────────────────────────────────────────────
fig16, axes16 = plt.subplots(1, 2, figsize=(14, 6))
fig16.suptitle(
    "Throughput Scaling with Dataset Size  (Mac, Apple Silicon)",
    fontsize=12, fontweight="bold", y=1.01,
)

def _scaling_panel(ax, title, N_vals, method_data, annot_str):
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.55)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Database size N", fontsize=10)
    ax.set_ylabel("QPS  (log scale)", fontsize=10)
    ax.set_yscale("log")
    ax.set_xscale("log")
    for _name, _vals in method_data:
        ax.plot(N_vals, _vals, "o-", color=COLORS[_name], linewidth=2,
                markersize=8, label=_name)
        for _n, _v in zip(N_vals, _vals):
            ax.annotate(f"{_v:,}", (_n, _v), textcoords="offset points",
                        xytext=(6, 3), fontsize=8, color=COLORS[_name])
    ax.set_xticks(N_vals)
    ax.set_xticklabels([f"{int(n/1e6)}M" for n in N_vals], fontsize=10)
    ax.legend(fontsize=9)
    ax.annotate(annot_str, xy=(0.04, 0.04), xycoords="axes fraction",
                fontsize=8.5, color="#333",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fff3cd", alpha=0.85))

# SIFT family: efSearch=128 for HNSW (R@1≈0.998), nprobe=64 for IVFFlat (R@1≈0.989/0.994)
_scaling_panel(
    axes16[0],
    "SIFT  (d=128,  1 M → 10 M)",
    [1e6, 10e6],
    [("SuCo",    [335, 29]),
     ("HNSW",    [15624, 14452]),   # ef=128
     ("IVFFlat", [3442, 1196]),     # np=64
     ("FlatL2",  [2742, 286])],
    f"HNSW: {15624 // 14452}× slower at 10×N\n"
    f"SuCo: {335 // 29}× slower at 10×N\n"
    f"FlatL2: {2742 // 286}× slower at 10×N",
)

# Deep family: ef=128 for HNSW, np=64 for IVFFlat
_scaling_panel(
    axes16[1],
    "Deep  (d=96,  1 M → 10 M)",
    [1e6, 10e6],
    [("SuCo",    [328, 32]),
     ("HNSW",    [17548, 5150]),    # ef=128
     ("IVFFlat", [5053, 191]),      # np=64  (nlist=1024 hits ceiling at 10M)
     ("FlatL2",  [3168, 181])],
    f"HNSW: {17548 // 5150}× slower at 10×N\n"
    f"SuCo: {328 // 32}× slower at 10×N\n"
    f"IVFFlat (np=64): {5053 // 191}× slower",
)

fig16.tight_layout()
save(fig16, "16_n_scaling_qps.png")


# ─────────────────────────────────────────────────────────────────────────────
# ██████  FIG 17: Distance Ratio — Near-Exact Distance Recovery
# ─────────────────────────────────────────────────────────────────────────────
fig17, ax17 = plt.subplots(figsize=(10, 6))
fig17.suptitle(
    "Distance Ratio  (returned NN distance / true NN distance)  |  SuCo default config",
    fontsize=12, fontweight="bold",
)

ax17.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.55)
_dr_ds    = ["SIFT1M", "GIST1M", "Deep1M", "SIFT10M", "Deep10M", "SpaceV10M"]
_dr_vals  = [1.0003,   1.0007,   1.0003,   1.0000,    1.0001,    None]
_dr_r1    = [0.9895,   0.9750,   0.9961,   0.9983,    0.9982,    0.9292]
_x17 = np.arange(len(_dr_ds))

for _i, (_ds, _dr, _r1) in enumerate(zip(_dr_ds, _dr_vals, _dr_r1)):
    if _dr is not None:
        ax17.bar(_i, _dr - 1.0, 0.55, bottom=1.0, color=COLORS["SuCo"],
                 edgecolor="black", linewidth=0.8)
        ax17.text(_i, _dr + 0.00005, f"{_dr:.4f}",
                  ha="center", va="bottom", fontsize=10, fontweight="bold")
    else:
        ax17.text(_i, 1.0002, "N/A", ha="center", va="bottom",
                  fontsize=9, color="#888")
    ax17.text(_i, 0.9999, f"R@1\n{_r1:.4f}", ha="center", va="top",
              fontsize=7.5, color="#444")

ax17.axhline(1.0, color="black", linewidth=1.5, zorder=5)
ax17.axhline(1.001, color="#bbb", linewidth=0.8, linestyle="--", alpha=0.6)
ax17.text(5.6, 1.001, "0.1% above exact", va="bottom", fontsize=8, color="#888")
ax17.set_xticks(_x17)
ax17.set_xticklabels(_dr_ds, fontsize=10)
ax17.set_ylabel("Distance ratio  (1.000 = exact)", fontsize=10)
ax17.set_ylim(0.9995, 1.0016)
ax17.set_title(
    "SuCo recovers the true NN distance to within 0.07%  —  near-exact across all datasets",
    fontsize=9, style="italic",
)
fig17.tight_layout()
save(fig17, "17_dist_ratio.png")


# ─────────────────────────────────────────────────────────────────────────────
# The figures below (18–22) require TSV data generated by the companion
# benchmark scripts.  Each figure gracefully skips if its TSV is absent.
# ─────────────────────────────────────────────────────────────────────────────
import pathlib as _pl
import csv    as _csv2

_BENCH_DIR = _pl.Path(__file__).parent


def _load_tsv(path: str | _pl.Path) -> list[dict]:
    p = _pl.Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return list(_csv2.DictReader(f, delimiter="\t"))


def _skip(name: str, path: str):
    print(f"  [SKIP] {name} — TSV not found: {path}")
    print(f"         Run the matching benchmark script first.")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 18  SC-score Pareto  (generated by bench_fig1_sc_score_pareto.py)
# ─────────────────────────────────────────────────────────────────────────────
_SC_DATASETS = [
    ("SIFT10M",   "pareto_sc_scores_sift10m.tsv",   "#e63946"),
    ("Deep10M",   "pareto_sc_scores_deep10m.tsv",   "#457b9d"),
    ("SpaceV10M", "pareto_sc_scores_spacev10m.tsv",  "#6ab187"),
]

_any_sc = any((_BENCH_DIR / f).exists() for _, f, _ in _SC_DATASETS)
if _any_sc:
    fig18, axes18 = plt.subplots(1, len(_SC_DATASETS), figsize=(6 * len(_SC_DATASETS), 6))
    if len(_SC_DATASETS) == 1:
        axes18 = [axes18]
    fig18.suptitle(
        "Pareto Property of SC-scores  —  avg SC-score of the i-th NN over 1000 queries",
        fontsize=12, fontweight="bold", y=1.01,
    )
    _plotted18 = False
    for _ax18, (_ds18, _fname18, _col18) in zip(axes18, _SC_DATASETS):
        _rows18 = _load_tsv(_BENCH_DIR / _fname18)
        if not _rows18:
            _ax18.set_title(f"{_ds18}\n(run benchmark first)", color="#888")
            _ax18.axis("off")
            continue
        _plotted18 = True
        _ranks18 = np.array([int(r["rank"]) for r in _rows18])
        _scores18 = np.array([float(r["avg_sc_score"]) for r in _rows18])
        # Thin scatter with alpha for density
        _stride = max(1, len(_ranks18) // 5000)   # downsample for plot clarity
        _ax18.scatter(_ranks18[::_stride], _scores18[::_stride],
                      s=0.5, color=_col18, alpha=0.4, linewidths=0, rasterized=True)
        # Overlay a smoothed percentile curve
        _bins18 = np.logspace(0, np.log10(len(_ranks18)), 300).astype(int)
        _bins18 = np.unique(np.clip(_bins18, 0, len(_ranks18) - 1))
        _ax18.plot(_ranks18[_bins18], _scores18[_bins18],
                   color="black", linewidth=1.5, alpha=0.75, zorder=5)
        _ax18.set_xscale("log")
        _ax18.set_xlabel("Rank i  (1 = nearest neighbour)", fontsize=10)
        _ax18.set_ylabel("Avg SC-score", fontsize=10)
        _ax18.set_title(f"{_ds18}", fontsize=11, fontweight="bold")
        _ax18.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        # Annotate Ns (max SC-score) with a dashed hline
        _ns18 = int(round(_scores18[:10].mean())) if len(_scores18) >= 10 else 1
        _ax18.axhline(_ns18, color="#888", linestyle=":", linewidth=1, alpha=0.7)
        _ax18.text(1.2, _ns18 + 0.05, f"Ns = {_ns18}", fontsize=8.5, color="#444")
    if _plotted18:
        fig18.tight_layout()
        save(fig18, "18_sc_score_pareto.png")
else:
    _skip("Fig 18 – SC-score Pareto",
          str(_BENCH_DIR / "pareto_sc_scores_<dataset>.tsv"))


# ─────────────────────────────────────────────────────────────────────────────
# FIG 19  Table 2 – SC-Linear recall at k=50  (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────
_T2_FILES = sorted(_BENCH_DIR.glob("table2_*.tsv"))
if _T2_FILES:
    fig19, ax19 = plt.subplots(figsize=(16, 7))
    fig19.suptitle(
        "Table 2 – SC-Search Recall When Returning k=50 NNs  (50-Recall@50)",
        fontsize=12, fontweight="bold",
    )
    _ds19_colors = {"sift10m": "#e63946", "sift1m": "#457b9d",
                    "deep1m": "#6ab187", "deep10m": "#f4a261"}
    _x19_all: list[str] = []
    _bars19: dict[str, dict[str, float]] = {}
    for _tf19 in _T2_FILES:
        _ds_label19 = _tf19.stem.replace("table2_", "")
        for _row19 in _load_tsv(_tf19):
            _lbl = _row19.get("label", "").strip()
            if _lbl not in _x19_all:
                _x19_all.append(_lbl)
            if _lbl not in _bars19:
                _bars19[_lbl] = {}
            _bars19[_lbl][_ds_label19] = float(_row19.get("50-R@50", 0) or 0)

    _ds_list19 = [_tf19.stem.replace("table2_", "") for _tf19 in _T2_FILES]
    _w19 = 0.8 / max(len(_ds_list19), 1)
    for _di19, _ds19 in enumerate(_ds_list19):
        _vals19 = [_bars19.get(_lbl, {}).get(_ds19, 0) for _lbl in _x19_all]
        _xs19   = np.arange(len(_x19_all))
        _offset19 = (_di19 - len(_ds_list19)/2 + 0.5) * _w19
        ax19.bar(_xs19 + _offset19, _vals19, _w19,
                 label=_ds19, color=_ds19_colors.get(_ds19, "#aaa"),
                 edgecolor="black", linewidth=0.5)

    ax19.set_xticks(np.arange(len(_x19_all)))
    ax19.set_xticklabels(_x19_all, rotation=25, ha="right", fontsize=8)
    ax19.set_ylabel("50-Recall@50", fontsize=11)
    ax19.set_ylim(0, 1.05)
    ax19.axhline(0.90, lw=0.7, ls="--", color="#888", alpha=0.5)
    ax19.legend(title="Dataset", fontsize=9)
    ax19.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    fig19.tight_layout()
    save(fig19, "19_table2_sc_linear_recall50.png")
else:
    _skip("Fig 19 – Table 2 SC-Linear recall@50",
          str(_BENCH_DIR / "table2_<dataset>.tsv"))


# ─────────────────────────────────────────────────────────────────────────────
# FIG 20  Figure 6 – Dynamic Activation vs Multi-sequence
# ─────────────────────────────────────────────────────────────────────────────
_F6_FILES = sorted(_BENCH_DIR.glob("fig6_*.tsv"))
if _F6_FILES:
    fig20, axes20 = plt.subplots(1, len(_F6_FILES), figsize=(7 * len(_F6_FILES), 6),
                                 squeeze=False)
    fig20.suptitle(
        "Figure 6 – Dynamic Activation vs Multi-sequence  |  SIFT10M Query Efficiency",
        fontsize=12, fontweight="bold", y=1.01,
    )
    _alg20_cols = {"DynamicActivation": COLORS["SuCo"], "MultiSequence": "#f4a261"}
    _alg20_markers = {"DynamicActivation": "o", "MultiSequence": "s"}
    _alg20_labels  = {"DynamicActivation": "Dynamic Activation (proposed)",
                      "MultiSequence": "Multi-sequence (classical)"}

    for _i20, (_f20, _ax20) in enumerate(zip(_F6_FILES, axes20[0])):
        _rows20 = _load_tsv(_f20)
        _ds20   = _f20.stem.replace("fig6_", "")
        _ax20.grid(True, linestyle="--", linewidth=0.4, alpha=0.55)
        _ax20.set_title(f"{_ds20}", fontsize=11, fontweight="bold")
        _ax20.set_xlabel("Candidates  (α × N)", fontsize=10)
        _ax20.set_ylabel("QPS", fontsize=10)
        _ax20.set_xscale("log")
        _ax20.set_yscale("log")

        _by_alg20: dict[str, list] = {}
        for _row20 in _rows20:
            _alg = _row20.get("algorithm", "")
            _by_alg20.setdefault(_alg, []).append(_row20)

        for _alg, _rr in _by_alg20.items():
            _cands = [int(r["candidates"]) for r in _rr]
            _qps   = [float(r["QPS"])      for r in _rr]
            _r1    = [float(r["R@1"])      for r in _rr]
            _ax20.plot(_cands, _qps, marker=_alg20_markers.get(_alg, "o"),
                       color=_alg20_cols.get(_alg, "#888"), linewidth=2,
                       markersize=7, label=_alg20_labels.get(_alg, _alg))
            # Annotate R@1 at a few points
            for _ci, (_cx, _cy, _cr) in enumerate(zip(_cands, _qps, _r1)):
                if _ci % 3 == 1:
                    _ax20.annotate(f"R@1={_cr:.3f}", (_cx, _cy),
                                   textcoords="offset points", xytext=(4, 4),
                                   fontsize=7, color=_alg20_cols.get(_alg, "#888"))

        _ax20.legend(fontsize=9, loc="upper right")

    fig20.tight_layout()
    save(fig20, "20_fig6_dynamic_vs_multisequence.png")
else:
    _skip("Fig 20 – Dynamic Activation vs Multi-sequence",
          str(_BENCH_DIR / "fig6_<dataset>.tsv"))


# ─────────────────────────────────────────────────────────────────────────────
# FIG 21  Figure 7 – Effect of K and Ns
# ─────────────────────────────────────────────────────────────────────────────
_F7_FILES = sorted(_BENCH_DIR.glob("fig7_*.tsv"))
if _F7_FILES:
    _nds7 = len(_F7_FILES)
    fig21, axes21 = plt.subplots(2, _nds7, figsize=(7 * _nds7, 11), squeeze=False)
    fig21.suptitle(
        "Figure 7 – SuCo Performance vs K-means Clusters K and Subspaces Ns",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for _col7, _f7 in enumerate(_F7_FILES):
        _rows7 = _load_tsv(_f7)
        _ds7   = _f7.stem.replace("fig7_", "")
        _ax7a  = axes21[0][_col7]   # top row: vary Ns
        _ax7b  = axes21[1][_col7]   # bottom row: vary K

        for _ax7 in (_ax7a, _ax7b):
            _ax7.grid(True, linestyle="--", linewidth=0.4, alpha=0.55)
            _ax7.set_xlabel("R@1", fontsize=10)
            _ax7.set_ylabel("QPS", fontsize=10)
            _ax7.set_yscale("log")

        # Panel a: Ns sweep
        _ns_rows = [r for r in _rows7 if r.get("sweep_var") == "Ns"]
        if _ns_rows:
            _ax7a.set_title(f"{_ds7}  —  Vary Ns  (K=50²)", fontsize=11, fontweight="bold")
            _r1a   = [float(r["R@1"]) for r in _ns_rows]
            _qpsa  = [float(r["QPS"]) for r in _ns_rows]
            _nsa   = [int(r["Ns"])    for r in _ns_rows]
            _ax7a.plot(_r1a, _qpsa, "o-", color=COLORS["SuCo"], linewidth=2, markersize=8)
            for _ri, _qi, _ni in zip(_r1a, _qpsa, _nsa):
                _ax7a.annotate(f"Ns={_ni}", (_ri, _qi),
                               textcoords="offset points", xytext=(4, 4), fontsize=8)

        # Panel b: nc (K) sweep
        _nc_rows = [r for r in _rows7 if r.get("sweep_var") == "nc"]
        if _nc_rows:
            _ax7b.set_title(f"{_ds7}  —  Vary K=nc²  (Ns=8)", fontsize=11, fontweight="bold")
            _r1b   = [float(r["R@1"]) for r in _nc_rows]
            _qpsb  = [float(r["QPS"]) for r in _nc_rows]
            _ncb   = [int(r["nc"])    for r in _nc_rows]
            _ax7b.plot(_r1b, _qpsb, "s-", color="#f4a261", linewidth=2, markersize=8)
            for _ri, _qi, _ni in zip(_r1b, _qpsb, _ncb):
                _ax7b.annotate(f"K={_ni**2}", (_ri, _qi),
                               textcoords="offset points", xytext=(4, 4), fontsize=8)

    fig21.tight_layout()
    save(fig21, "21_fig7_K_Ns_sweep.png")
else:
    _skip("Fig 21 – K and Ns sweep", str(_BENCH_DIR / "fig7_<dataset>.tsv"))


# ─────────────────────────────────────────────────────────────────────────────
# FIG 22  Figure 8 – Effect of α and β
# ─────────────────────────────────────────────────────────────────────────────
_F8_FILES = sorted(_BENCH_DIR.glob("fig8_*.tsv"))
if _F8_FILES:
    _nds8 = len(_F8_FILES)
    fig22, axes22 = plt.subplots(2, _nds8, figsize=(7 * _nds8, 11), squeeze=False)
    fig22.suptitle(
        "Figure 8 – SuCo Query Performance vs Collision Ratio α and Re-rank Ratio β",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for _col8, _f8 in enumerate(_F8_FILES):
        _rows8 = _load_tsv(_f8)
        _ds8   = _f8.stem.replace("fig8_", "")
        _ax8a  = axes22[0][_col8]   # top row: α sweep
        _ax8b  = axes22[1][_col8]   # bottom row: β sweep

        for _ax8 in (_ax8a, _ax8b):
            _ax8.grid(True, linestyle="--", linewidth=0.4, alpha=0.55)

        # Panel a: α sweep (recall vs QPS)
        _a_rows = [r for r in _rows8 if r.get("sweep_var") == "alpha"]
        if _a_rows:
            _ax8a.set_title(f"{_ds8}  —  Vary α  (β={_a_rows[0].get('beta','?')} fixed)",
                            fontsize=11, fontweight="bold")
            _ax8a.set_xlabel("R@1", fontsize=10)
            _ax8a.set_ylabel("QPS", fontsize=10)
            _ax8a.set_yscale("log")
            _r1a8  = [float(r["R@1"]) for r in _a_rows]
            _qpsa8 = [float(r["QPS"]) for r in _a_rows]
            _aa8   = [float(r["alpha"]) for r in _a_rows]
            _ax8a.plot(_r1a8, _qpsa8, "o-", color=COLORS["SuCo"], linewidth=2, markersize=7)
            for _ri, _qi, _ai in zip(_r1a8, _qpsa8, _aa8):
                _ax8a.annotate(f"α={_ai:.2f}", (_ri, _qi),
                               textcoords="offset points", xytext=(4, 4), fontsize=7.5)

        # Panel b: β sweep
        _b_rows = [r for r in _rows8 if r.get("sweep_var") == "beta"]
        if _b_rows:
            _ax8b.set_title(f"{_ds8}  —  Vary β  (α={_b_rows[0].get('alpha','?')} fixed)",
                            fontsize=11, fontweight="bold")
            _ax8b.set_xlabel("R@1", fontsize=10)
            _ax8b.set_ylabel("QPS", fontsize=10)
            _ax8b.set_yscale("log")
            _r1b8  = [float(r["R@1"]) for r in _b_rows]
            _qpsb8 = [float(r["QPS"]) for r in _b_rows]
            _bb8   = [float(r["beta"]) for r in _b_rows]
            _ax8b.plot(_r1b8, _qpsb8, "s-", color="#e9c46a", linewidth=2, markersize=7)
            for _ri, _qi, _bi in zip(_r1b8, _qpsb8, _bb8):
                _ax8b.annotate(f"β={_bi:.3f}", (_ri, _qi),
                               textcoords="offset points", xytext=(4, 4), fontsize=7.5)

    fig22.tight_layout()
    save(fig22, "22_fig8_alpha_beta_sweep.png")
else:
    _skip("Fig 22 – α and β sweep", str(_BENCH_DIR / "fig8_<dataset>.tsv"))


print(f"\nAll plots saved to {OUT_DIR}/")

