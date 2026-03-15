#!/usr/bin/env python3
"""
Build benchmark figures directly from bench_all logs.

This script is log-driven (no hardcoded benchmark arrays) and focuses on:
- Mac-only benchmark figures from a bench_all log
- Optional Mac-vs-Linux comparison figures when a Linux log is provided

Usage:
  python benchs/plot_benchmarks_from_logs.py \
      --mac-log logs/bench_all_20260313_142528.log

  python benchs/plot_benchmarks_from_logs.py \
      --mac-log logs/bench_all_20260313_142528.log \
      --linux-log logs/bench_all_linux_20260312_101010.log
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "HNSW": "#e63946",
    "IVFFlat": "#457b9d",
    "IVFPQ": "#a8dadc",
    "OPQ+IVFPQ": "#6ab187",
    "SuCo": "#f4a261",
    "FlatL2": "#6c757d",
}

MARKERS = {
    "HNSW": "^",
    "IVFFlat": "s",
    "IVFPQ": "D",
    "OPQ+IVFPQ": "P",
    "SuCo": "o",
    "FlatL2": "X",
}

METHOD_ORDER = ["HNSW", "IVFFlat", "IVFPQ", "OPQ+IVFPQ"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mac-log", required=True, help="Path to Mac bench_all log")
    p.add_argument("--linux-log", default=None, help="Optional Linux bench_all log")
    p.add_argument("--out-dir", default="benchs/plots", help="Directory for output PNG files")
    return p.parse_args()


def dataset_key(raw: str) -> str:
    x = raw.strip().lower()
    if "sift10" in x:
        return "SIFT10M"
    if "sift1" in x:
        return "SIFT1M"
    if "gist1" in x:
        return "GIST1M"
    if "deep10" in x:
        return "Deep10M"
    if "deep1" in x:
        return "Deep1M"
    if "spacev10" in x:
        return "SpaceV10M"
    return raw.strip()


def new_dataset_entry() -> dict:
    return {
        "suco_default": None,
        "flat": None,
        "sweep": [],
        "HNSW": [],
        "IVFFlat": [],
        "IVFPQ": [],
        "OPQ+IVFPQ": [],
    }


def parse_log(log_path: Path) -> dict[str, dict]:
    text = log_path.read_text(errors="replace")
    lines = text.splitlines()

    data: dict[str, dict] = {}
    current_dataset: str | None = None
    mode: str | None = None

    i = 0
    while i < len(lines):
        ln = lines[i].rstrip("\n")

        m_sec = re.search(r"##\s+([A-Za-z0-9?]+)", ln)
        if m_sec:
            maybe_ds = dataset_key(m_sec.group(1))
            if maybe_ds in {"SIFT1M", "GIST1M", "Deep1M", "SIFT10M", "Deep10M", "SpaceV10M"}:
                current_dataset = maybe_ds
                data.setdefault(current_dataset, new_dataset_entry())

        m_bench = re.search(r"IndexSuCo benchmark\s+\(([^,\)]+)", ln)
        if m_bench:
            current_dataset = dataset_key(m_bench.group(1))
            data.setdefault(current_dataset, new_dataset_entry())

        if "Parameter Sweep" in ln:
            mode = "sweep"
            i += 1
            continue
        if "HNSW efSearch sweep" in ln:
            mode = "HNSW"
            i += 1
            continue
        if "IVFFlat nprobe sweep" in ln:
            mode = "IVFFlat"
            i += 1
            continue
        if "IVFPQ nprobe sweep" in ln and "OPQ+" not in ln:
            mode = "IVFPQ"
            i += 1
            continue
        if "OPQ+IVFPQ nprobe sweep" in ln:
            mode = "OPQ+IVFPQ"
            i += 1
            continue

        if "Baseline: IndexFlatL2" in ln:
            j = i + 1
            qps = None
            r1 = None
            while j < len(lines) and "IndexSuCo benchmark" not in lines[j] and ">>>" not in lines[j]:
                mq = re.search(r"QPS\s*=\s*([0-9.]+)", lines[j])
                mr = re.search(r"Recall@1\s*=\s*([0-9.]+)", lines[j])
                if mq:
                    qps = float(mq.group(1))
                if mr:
                    r1 = float(mr.group(1))
                j += 1
            if current_dataset and qps is not None and r1 is not None:
                data.setdefault(current_dataset, new_dataset_entry())
                data[current_dataset]["flat"] = (r1, qps)
            i = j
            continue

        if "Results:" in ln and current_dataset:
            j = i + 1
            qps = None
            r1 = None
            while j < len(lines):
                t = lines[j]
                if not t.strip():
                    break
                if t.strip().startswith("===") or t.strip().startswith(">>>"):
                    break
                mq = re.search(r"QPS\s*=\s*([0-9.]+)", t)
                mr = re.search(r"Recall@1\s*=\s*([0-9.]+)", t)
                if mq:
                    qps = float(mq.group(1))
                if mr:
                    r1 = float(mr.group(1))
                j += 1
            if qps is not None and r1 is not None:
                data.setdefault(current_dataset, new_dataset_entry())
                data[current_dataset]["suco_default"] = (r1, qps)
            i = j
            continue

        if mode == "sweep" and current_dataset:
            # Ns hd nc alpha beta ms/q QPS R@1 R@10 10R@10 ratio
            m = re.match(
                r"\s*(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
                ln,
            )
            if m:
                ns = int(m.group(1))
                nc = int(m.group(3))
                alpha = float(m.group(4))
                qps = float(m.group(7))
                r1 = float(m.group(8))
                label = f"Ns={ns}, nc={nc}, a={alpha:.3g}"
                data[current_dataset]["sweep"].append((r1, qps, label))

        if mode in {"HNSW", "IVFFlat", "IVFPQ", "OPQ+IVFPQ"} and current_dataset:
            # ef/nprobe, ms/q, QPS, R@1, R@10
            m = re.match(r"\s*(\d+)\s+[0-9.]+\s+([0-9.]+)\s+([0-9.]+)\s+[0-9.]+", ln)
            if m:
                qps = float(m.group(2))
                r1 = float(m.group(3))
                data[current_dataset][mode].append((r1, qps))

        if ln.strip().startswith(">>>"):
            mode = None

        i += 1

    return data


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []
    pts = sorted(points, key=lambda x: x[0])
    out: list[tuple[float, float]] = []
    best_q = -1.0
    for r, q in pts:
        if q > best_q:
            out.append((r, q))
            best_q = q
    return out


def draw_pareto(ax: plt.Axes, points: list[tuple[float, float]], name: str) -> None:
    if not points:
        return
    pf = pareto_frontier(points)
    rs = [x[0] for x in pf]
    qs = [x[1] for x in pf]
    ax.plot(
        rs,
        qs,
        marker=MARKERS[name],
        markersize=5.5,
        linewidth=2,
        color=COLORS[name],
        label=name,
        alpha=0.9,
    )


def setup_ax(ax: plt.Axes, title: str, xlo: float, ylabel: bool = True) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.55)
    ax.set_yscale("log")
    ax.set_xlim(xlo, 1.002)
    ax.set_xlabel("Recall@1")
    if ylabel:
        ax.set_ylabel("QPS (log scale)")
    ax.set_title(title, fontsize=11, fontweight="bold")


def plot_mac_1m(mac: dict[str, dict], out_dir: Path) -> None:
    datasets = [
        ("SIFT1M", 0.20),
        ("GIST1M", 0.12),
        ("Deep1M", 0.15),
    ]
    available = [(d, xlo) for d, xlo in datasets if d in mac]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 6))
    if len(available) == 1:
        axes = [axes]

    fig.suptitle("Recall-QPS Pareto Curves | Mac (from log)", fontsize=13, fontweight="bold", y=1.02)

    for idx, ((ds, xlo), ax) in enumerate(zip(available, axes)):
        entry = mac[ds]
        setup_ax(ax, f"{ds}", xlo=xlo, ylabel=(idx == 0))

        for method in METHOD_ORDER:
            draw_pareto(ax, entry[method], method)

        if entry["suco_default"]:
            ax.scatter(
                *entry["suco_default"],
                s=180,
                color=COLORS["SuCo"],
                marker="o",
                edgecolors="black",
                linewidths=1.1,
                zorder=7,
                label="SuCo (default)",
            )

        if entry["flat"]:
            ax.scatter(
                *entry["flat"],
                s=145,
                color=COLORS["FlatL2"],
                marker="X",
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
                label="FlatL2 (exact)",
            )

        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out = out_dir / "01_pareto_mac_1m_from_log.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_mac_sweeps(mac: dict[str, dict], out_dir: Path) -> None:
    candidates = ["SIFT1M", "GIST1M", "Deep1M", "SIFT10M", "Deep10M", "SpaceV10M"]
    avail = [d for d in candidates if d in mac and mac[d]["sweep"]]
    if not avail:
        return

    n = len(avail)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 5 * rows))
    axes_arr = np.atleast_1d(axes).flatten()

    fig.suptitle("SuCo Parameter Sweeps | Mac (from log)", fontsize=13, fontweight="bold", y=1.01)

    for ax in axes_arr[n:]:
        ax.axis("off")

    for i, ds in enumerate(avail):
        ax = axes_arr[i]
        pts = mac[ds]["sweep"]
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.55)
        ax.set_xlabel("Recall@1")
        ax.set_ylabel("QPS")
        ax.set_title(ds, fontsize=11, fontweight="bold")

        for r1, qps, lbl in pts:
            is_default = "a=0.05" in lbl and "nc=50" in lbl
            ax.scatter(
                r1,
                qps,
                s=120 if is_default else 42,
                color=COLORS["SuCo"] if is_default else "#e76f51",
                edgecolors="black",
                linewidths=1.0 if is_default else 0.35,
            )
        if mac[ds]["suco_default"]:
            r, q = mac[ds]["suco_default"]
            ax.scatter(r, q, s=180, marker="*", color="#111111", label="default")
            ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    out = out_dir / "02_param_sweeps_mac_from_log.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def plot_cross_platform_defaults(mac: dict[str, dict], linux: dict[str, dict], out_dir: Path) -> None:
    common = [d for d in ["SIFT1M", "GIST1M", "Deep1M", "SIFT10M", "Deep10M"] if d in mac and d in linux]
    common = [d for d in common if mac[d]["suco_default"] and linux[d]["suco_default"]]
    if not common:
        return

    fig, axes = plt.subplots(1, len(common), figsize=(6 * len(common), 6))
    if len(common) == 1:
        axes = [axes]

    fig.suptitle("Mac vs Linux | Default QPS comparison", fontsize=13, fontweight="bold", y=1.02)

    for ax, ds in zip(axes, common):
        methods = ["SuCo", "FlatL2"]
        vals_mac = []
        vals_lin = []
        labels = []

        suco_m = mac[ds]["suco_default"]
        suco_l = linux[ds]["suco_default"]
        if suco_m and suco_l:
            labels.append("SuCo")
            vals_mac.append(suco_m[1])
            vals_lin.append(suco_l[1])

        flat_m = mac[ds]["flat"]
        flat_l = linux[ds]["flat"]
        if flat_m and flat_l:
            labels.append("FlatL2")
            vals_mac.append(flat_m[1])
            vals_lin.append(flat_l[1])

        if not labels:
            ax.axis("off")
            continue

        x = np.arange(len(labels))
        w = 0.35

        ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.55)
        ax.set_yscale("log")
        bm = ax.bar(x - w / 2, vals_mac, w, color="#457b9d", edgecolor="black", linewidth=0.7, label="Mac")
        bl = ax.bar(x + w / 2, vals_lin, w, color="#e76f51", edgecolor="black", linewidth=0.7, label="Linux")

        for b, v in zip(bm, vals_mac):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.06, f"{int(v):,}", ha="center", va="bottom", fontsize=8)
        for b, v in zip(bl, vals_lin):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.06, f"{int(v):,}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(ds, fontsize=11, fontweight="bold")
        ax.set_ylabel("QPS (log scale)")
        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out = out_dir / "03_cross_platform_defaults_from_logs.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mac = parse_log(Path(args.mac_log))
    linux = parse_log(Path(args.linux_log)) if args.linux_log else None

    plot_mac_1m(mac, out_dir)
    plot_mac_sweeps(mac, out_dir)

    if linux is not None:
        plot_cross_platform_defaults(mac, linux, out_dir)

    print(f"Done. Figures are in {out_dir}")


if __name__ == "__main__":
    main()
