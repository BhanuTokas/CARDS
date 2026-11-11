"""Plot histograms of delta_p for both masking strategies.

Reads only the two delta_p columns from the faithfulness CSV to keep
memory low. Saves two PNGs to results/coco_stuff/.

Usage:
  python scripts/coco_stuff/analysis/plot_delta_p_histogram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR      = Path("results/coco_stuff")
FAITH_CSV        = RESULTS_DIR / "coco_stuff_faithfulness.csv"
COUNTS_CSV       = RESULTS_DIR / "coco_stuff_faithfulness_detection_counts.csv"
STRATEGIES       = ("black_mask", "black_bbox")
MIN_FAITH_IMAGES = 100


def _load_balanced_concepts(counts_csv: Path, min_images: int) -> set[str]:
    balanced: set[str] = set()
    with open(counts_csv, newline="") as f:
        import csv
        for row in csv.DictReader(f):
            if int(row["n_images"]) >= min_images:
                balanced.add(row["concept_name"])
    return balanced


def main() -> None:
    balanced = _load_balanced_concepts(COUNTS_CSV, MIN_FAITH_IMAGES)
    print(f"Balanced concepts: {len(balanced)}", flush=True)

    cols = ["concept_name"] + [f"delta_p_{s}" for s in STRATEGIES]
    print(f"Reading {FAITH_CSV} ...", flush=True)
    df = pd.read_csv(FAITH_CSV, usecols=cols)
    df = df[df["concept_name"].isin(balanced)].drop(columns=["concept_name"])
    print(f"Rows after filtering to {len(balanced)} concepts: {len(df):,}", flush=True)

    for strategy in STRATEGIES:
        col  = f"delta_p_{strategy}"
        vals = df[col].dropna()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(vals, bins=200, color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvline(0, color="red", linewidth=1.2, linestyle="--", label="delta_p = 0")
        ax.set_xlabel("delta_p  (p_orig − p_masked)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"delta_p distribution — {strategy}\n"
                     f"mean={vals.mean():.4f}  std={vals.std():.4f}  "
                     f"median={vals.median():.4f}", fontsize=11)
        ax.legend()
        out = RESULTS_DIR / f"delta_p_histogram_{strategy}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}", flush=True)
        print(f"  [{strategy}] mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"median={vals.median():.4f}  "
              f"pct>0={100*(vals>0).mean():.1f}%", flush=True)


if __name__ == "__main__":
    main()
