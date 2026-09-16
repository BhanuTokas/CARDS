"""Builds top-5 (not top-10), Attractive-task-only local-attribution grids
for two paper-figure concepts (Eyeglasses, Black_Hair), reusing the same
scored CSV as the top-10 galleries -- no recompute, just a different
top-K/task filter for a more compact, side-by-side paper figure.

The original per-concept galleries (`rebuild_top10_grids_corrected_gt_with_
ground_truth.py`) pooled BOTH tasks' rows together for a given concept
before ranking (each (image, concept) pair contributes one row per task).
Since the paper is dropping Young from this experiment, this script filters
to target_task == "Attractive" BEFORE ranking, so the qualitative figure
matches the quantitative (Attractive-only) table exactly.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from local_attribution_plot_utils import build_image_grid_clean

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "local_attribution_top5_attractive_only"
CONCEPTS = ["Eyeglasses", "Black_Hair"]
TOP_K = 5


def main():
    rows = []
    with open(RESULTS_DIR / "local_attribution_celeba_pairs_corrected_gt.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["target_task"] != "Attractive":
                continue
            rows.append({
                "image": row["image"],
                "concept_name": row["concept_name"],
                "gt_delta_p": float(row["gt_delta_p"]),
                "hybrid_score": float(row["hybrid_score"]),
                "tcav_score": float(row["tcav_score"]),
            })
    print(f"Loaded {len(rows)} Attractive-only scored rows.", flush=True)

    by_concept: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_concept[r["concept_name"]].append(r)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for concept_name in CONCEPTS:
        concept_rows = by_concept[concept_name]
        rows_for_grid = [
            ("ground_truth", sorted([(r["image"], r["gt_delta_p"]) for r in concept_rows], key=lambda kv: -abs(kv[1]))[:TOP_K]),
            ("hybrid", sorted([(r["image"], r["hybrid_score"]) for r in concept_rows], key=lambda kv: -abs(kv[1]))[:TOP_K]),
            ("tcav", sorted([(r["image"], r["tcav_score"]) for r in concept_rows], key=lambda kv: -abs(kv[1]))[:TOP_K]),
        ]
        build_image_grid_clean(rows_for_grid, OUT_DIR / f"{concept_name}.png")
        print(f"  built {concept_name}.png ({len(concept_rows)} Attractive-only pairs available)", flush=True)

    print(f"Saved {len(CONCEPTS)} grids to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
