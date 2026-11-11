"""Plot pooled_rho and mean class rho vs alpha from alpha_ablation_summary.csv.

Usage:
    cd /scratch/rnair21/CARDS
    /scratch/rnair21/envs/cards_env/bin/python \\
        scripts/coco_stuff/analysis/plot_alpha_ablation.py

Output:
    results/coco_stuff/alpha_ablation_rho_trend.png
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

SUMMARY_CSV = Path("results/coco_stuff/alpha_ablation_summary.csv")
OUT_PATH    = Path("results/coco_stuff/alpha_ablation_rho_trend.png")


def main() -> None:
    # Collect per (strategy, alpha): pooled_rho and per-class rhos
    pooled_rho:  dict[tuple[str, float], float]       = {}
    class_rhos:  dict[tuple[str, float], list[float]] = {}

    with open(SUMMARY_CSV, newline="") as f:
        for row in csv.DictReader(f):
            strategy = row["strategy"]
            alpha    = float(row["alpha"])
            pr       = row["pooled_rho"]
            cr       = row["spearman_rho"]
            key      = (strategy, alpha)
            if key not in pooled_rho and pr not in ("", "nan"):
                pooled_rho[key] = float(pr)
                class_rhos[key] = []
            if cr not in ("", "nan"):
                class_rhos.setdefault(key, []).append(float(cr))

    strategies = sorted({k[0] for k in pooled_rho})
    alphas     = sorted({k[1] for k in pooled_rho})

    fig, ax = plt.subplots(figsize=(8, 4))

    markers = {"black_mask": "o", "black_bbox": "s"}
    colors  = {"black_mask": "#1f77b4", "black_bbox": "#ff7f0e"}

    for strategy in strategies:
        pr_vals   = [pooled_rho.get((strategy, a), float("nan")) for a in alphas]
        mean_vals = [
            float(np.nanmean(class_rhos.get((strategy, a), [float("nan")])))
            for a in alphas
        ]
        m = markers.get(strategy, "o")
        c = colors.get(strategy, None)
        ax.plot(alphas, pr_vals,   marker=m,   color=c, linewidth=2,
                label=f"{strategy} — pooled ρ")
        ax.plot(alphas, mean_vals, marker=m,   color=c, linewidth=1,
                linestyle="--", alpha=0.6,
                label=f"{strategy} — mean class ρ")

    ax.set_xlabel("Alpha (z-score threshold)", fontsize=12)
    ax.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_title("CARDS faithfulness vs. localization threshold α (COCO-Stuff)", fontsize=13)
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"{a:.2f}" for a in alphas], rotation=45)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
