#!/usr/bin/env python3
"""
Generate conceptual SuCo paper diagrams that are not produced by the
benchmark-result plotting scripts.

This script creates illustrative (not benchmark-derived) versions of:
- Figure 3: SuCo workflow overview
- Figure 4: IVF vs IMI 2D clustering illustration
- Figure 5: Dynamic Activation traversal illustration

Outputs are saved under benchs/plots/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> None:
    out = OUT_DIR / filename
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_fig3_workflow() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.02,rounding_size=0.02")

    def add_box(x, y, w, h, text, color):
        rect = patches.FancyBboxPatch((x, y), w, h, **box_style,
                                      linewidth=1.4, edgecolor="#2f3e46",
                                      facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=10.5, color="#1d3557", fontweight="bold")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#344e41"))

    # Offline/index build path.
    add_box(0.06, 0.70, 0.17, 0.18, "Training Data\n(xt)", "#d8f3dc")
    add_box(0.30, 0.70, 0.20, 0.18, "Subspace Split\n+ K-means", "#bee1e6")
    add_box(0.57, 0.70, 0.18, 0.18, "Build Ns IMIs\nPer Subspace", "#fefae0")
    add_box(0.80, 0.70, 0.14, 0.18, "Store\nBuckets", "#ffd6a5")

    arrow(0.23, 0.79, 0.30, 0.79)
    arrow(0.50, 0.79, 0.57, 0.79)
    arrow(0.75, 0.79, 0.80, 0.79)

    # Online/query path.
    add_box(0.06, 0.30, 0.17, 0.18, "Query q", "#ffddd2")
    add_box(0.30, 0.30, 0.20, 0.18, "Compute SC-Score\nAcross Ns", "#e0fbfc")
    add_box(0.57, 0.30, 0.18, 0.18, "Dynamic Activation\n(Top-candidates)", "#fff3b0")
    add_box(0.80, 0.30, 0.14, 0.18, "Re-rank\nOutput k-NN", "#cdeac0")

    arrow(0.23, 0.39, 0.30, 0.39)
    arrow(0.50, 0.39, 0.57, 0.39)
    arrow(0.75, 0.39, 0.80, 0.39)

    # Link index artifacts into query path.
    ax.annotate("", xy=(0.66, 0.49), xytext=(0.66, 0.69),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#6c757d", linestyle="--"))
    ax.text(0.675, 0.59, "Use IMI buckets", fontsize=9, color="#495057", va="center")

    # Phase labels.
    ax.text(0.02, 0.89, "Offline Index Construction", fontsize=11.5,
            color="#2b2d42", fontweight="bold")
    ax.text(0.02, 0.49, "Online Query Processing", fontsize=11.5,
            color="#2b2d42", fontweight="bold")

    ax.set_title("Figure 3 (illustration): Overview of the SuCo workflow",
                 fontsize=13, fontweight="bold", pad=12)
    _save(fig, "23_fig3_suco_workflow_diagram.png")


def plot_fig4_ivf_vs_imi() -> None:
    rng = np.random.default_rng(7)
    pts = np.vstack([
        rng.normal(loc=(-1.2, -1.1), scale=0.35, size=(110, 2)),
        rng.normal(loc=(1.0, -0.8), scale=0.33, size=(110, 2)),
        rng.normal(loc=(-0.8, 1.1), scale=0.32, size=(110, 2)),
        rng.normal(loc=(1.15, 1.0), scale=0.30, size=(110, 2)),
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)
    fig.patch.set_facecolor("#f8f9fa")

    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.scatter(pts[:, 0], pts[:, 1], s=8, c="#457b9d", alpha=0.35, linewidths=0)
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.4)
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_xlabel("subspace u")
    axes[0].set_ylabel("subspace v")

    # Left: IVF-style Voronoi-like coarse partition with 4 centroids.
    c_ivf = np.array([[-1.1, -1.1], [1.0, -0.9], [-0.8, 1.0], [1.1, 1.0]])
    axes[0].scatter(c_ivf[:, 0], c_ivf[:, 1], s=90, marker="X", c="#e63946",
                    edgecolors="#1d3557", linewidths=1.0, zorder=5)
    for x in [-0.05, 0.15]:
        axes[0].axvline(x, color="#e76f51", lw=1.0, alpha=0.6)
    for y in [-0.1, 0.2]:
        axes[0].axhline(y, color="#e76f51", lw=1.0, alpha=0.6)
    axes[0].set_title("(a) Inverted Index (single-space K-means)",
                      fontsize=10.5, fontweight="bold")

    # Right: IMI-style product partition from two independent 1D quantizers.
    for x in [-1.4, -0.2, 0.8, 1.6]:
        axes[1].axvline(x, color="#2a9d8f", lw=1.0, alpha=0.8)
    for y in [-1.5, -0.4, 0.5, 1.4]:
        axes[1].axhline(y, color="#2a9d8f", lw=1.0, alpha=0.8)
    axes[1].set_title("(b) Inverted Multi-Index (product cells)",
                      fontsize=10.5, fontweight="bold")

    fig.suptitle("Figure 4 (illustration): IVF vs IMI partitioning in 2D",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "24_fig4_ivf_vs_imi_diagram.png")


def plot_fig5_dynamic_activation() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    n = 7
    rng = np.random.default_rng(11)
    scores = rng.uniform(0.0, 1.0, size=(n, n))
    # Make a few high-score cells to mimic likely activations.
    for r, c, s in [(3, 3, 0.98), (3, 4, 0.92), (2, 3, 0.89), (4, 3, 0.88), (2, 4, 0.84)]:
        scores[r, c] = s

    im = ax.imshow(scores, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Activation priority")

    # Candidate visitation order example.
    path = [(3, 3), (3, 4), (2, 3), (4, 3), (2, 4), (4, 4)]
    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    ax.plot(xs, ys, "-o", color="#1d3557", linewidth=2.0, markersize=5, zorder=5)
    for i, (r, c) in enumerate(path, start=1):
        ax.text(c, r, str(i), ha="center", va="center", fontsize=8,
                color="white", fontweight="bold")

    # Query center marker.
    ax.scatter([3], [3], marker="*", s=220, c="#264653", edgecolors="white",
               linewidths=1.0, zorder=6)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xlabel("Centroid id in subspace A")
    ax.set_ylabel("Centroid id in subspace B")
    ax.set_title("Figure 5 (illustration): Dynamic Activation traversal over IMI cells",
                 fontsize=12.5, fontweight="bold")

    # Mark a cutoff to illustrate stopping criterion.
    cutoff_step = 5
    ax.text(0.02, -0.10,
            f"Example stopping criterion: stop after top {cutoff_step} activated cells",
            transform=ax.transAxes, fontsize=9, color="#495057")

    fig.tight_layout()
    _save(fig, "25_fig5_dynamic_activation_diagram.png")


def main() -> None:
    print("Generating conceptual paper diagrams (Figures 3-5 illustrations)...")
    plot_fig3_workflow()
    plot_fig4_ivf_vs_imi()
    plot_fig5_dynamic_activation()
    print(f"Done. Files are in {OUT_DIR}")


if __name__ == "__main__":
    main()
