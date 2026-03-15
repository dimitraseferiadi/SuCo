#!/usr/bin/env python3
"""Plot SIFT1M/GIST1M SuCo sweep metrics from a run_suco_only log.

Usage:
  python benchs/plot_suco_only_log.py \
      --log logs/bench_suco_only_20260313_132331.log

Outputs:
  benchs/plots/30_sift1m_suco_sweep_from_log.png
  benchs/plots/30_gist1m_suco_sweep_from_log.png
  benchs/plots/30_sift1m_gist1m_defaults_from_log.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="Path to run_suco_only log file")
    p.add_argument("--out-dir", default="benchs/plots", help="Output directory")
    return p.parse_args()


def parse_default_metrics(text: str, dataset: str) -> dict[str, float] | None:
    pat = re.compile(
        rf"IndexSuCo benchmark\s+\({dataset},.*?Results:\n(.*?)(?:\n\n=+)",
        re.S,
    )
    m = pat.search(text)
    if not m:
        return None
    blk = m.group(1)

    def get(name: str) -> float:
        mm = re.search(rf"{name}\s*=\s*([0-9.]+)", blk)
        if not mm:
            raise ValueError(f"Missing metric {name} for {dataset}")
        return float(mm.group(1))

    return {
        "qps": get("QPS"),
        "r1": get("Recall@1"),
        "r10": get("Recall@10"),
        "dr": get("Dist ratio"),
    }


def parse_sweep_rows(text: str, dataset: str) -> list[dict[str, float]]:
    pat = re.compile(
        rf"Parameter Sweep\s+\({dataset},[^\n]*\)\n=+\n.*?\n-+\n(.*?)(?:IndexSuCo::read_index|\n\(done in)",
        re.S,
    )
    m = pat.search(text)
    if not m:
        return []

    rows: list[dict[str, float]] = []
    for ln in m.group(1).splitlines():
        ln = ln.strip()
        if not ln or not re.match(r"^\d+", ln):
            continue
        parts = ln.split()
        if len(parts) < 11:
            continue
        rows.append(
            {
                "Ns": int(parts[0]),
                "hd": int(parts[1]),
                "nc": int(parts[2]),
                "alpha": float(parts[3]),
                "beta": float(parts[4]),
                "msq": float(parts[5]),
                "qps": float(parts[6]),
                "r1": float(parts[7]),
                "r10": float(parts[8]),
                "r10rec10": float(parts[9]),
                "ratio": float(parts[10]),
            }
        )
    return rows


def plot_sweep(
    dataset: str,
    rows: list[dict[str, float]],
    default_metrics: dict[str, float] | None,
    out_dir: Path,
) -> None:
    colors = {"SIFT1M": "#457b9d", "GIST1M": "#e76f51"}
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    x = [r["r1"] for r in rows]
    y = [r["qps"] for r in rows]
    labels = [f"Ns={r['Ns']},nc={r['nc']},a={r['alpha']:.2f}" for r in rows]

    ax.scatter(x, y, s=58, color=colors[dataset], alpha=0.88)
    for xi, yi, lbl in zip(x, y, labels):
        ax.annotate(lbl, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=7)

    if default_metrics:
        ax.scatter(
            [default_metrics["r1"]],
            [default_metrics["qps"]],
            s=180,
            marker="*",
            color="#111111",
            label="default",
            zorder=5,
        )

    ax.set_xlabel("Recall@1")
    ax.set_ylabel("QPS")
    ax.set_title(f"{dataset} SuCo Sweep (from log)", fontweight="bold")
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = out_dir / f"30_{dataset.lower()}_suco_sweep_from_log.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


def plot_defaults(defaults: dict[str, dict[str, float]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ds = ["SIFT1M", "GIST1M"]
    qps = [defaults[d]["qps"] for d in ds]
    r1 = [defaults[d]["r1"] for d in ds]

    axes[0].bar(ds, qps, color=["#457b9d", "#e76f51"])
    axes[0].set_title("Default QPS")
    axes[0].set_ylabel("QPS")

    axes[1].bar(ds, r1, color=["#457b9d", "#e76f51"])
    axes[1].set_title("Default Recall@1")
    axes[1].set_ylim(0.85, 1.0)

    fig.suptitle("SuCo defaults from run_suco_only log", fontweight="bold")
    fig.tight_layout()
    out = out_dir / "30_sift1m_gist1m_defaults_from_log.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = log_path.read_text()

    defaults = {}
    for ds in ("SIFT1M", "GIST1M"):
        m = parse_default_metrics(text, ds)
        if m:
            defaults[ds] = m

    for ds in ("SIFT1M", "GIST1M"):
        rows = parse_sweep_rows(text, ds)
        if rows:
            plot_sweep(ds, rows, defaults.get(ds), out_dir)

    if "SIFT1M" in defaults and "GIST1M" in defaults:
        plot_defaults(defaults, out_dir)


if __name__ == "__main__":
    main()
