"""Rebuilds the CORRECTED-ground-truth top-10 grids with `ground_truth`
(real gt_delta_p) replacing `baseline` -- carries forward the same
substitution already applied to the original-ground-truth grids
(`rebuild_top10_grids_with_ground_truth_celeba.py`, prompted "could you
replace the baseline with ground truth masking results?"), which the
newer `local_attribution_comparison_celeba_corrected_gt.py` run
reverted to `baseline` by oversight (caught directly: "Shouldn't the
images be with gt not baseline?").

Reuses `results/local_attribution_celeba_pairs_corrected_gt.csv`
directly -- no recompute. Also carries forward the two most recent
requests, applied to all 3 rows: ranked by |magnitude| (not raw value),
sign shown via score-text color (green=positive, red=negative).
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from local_attribution_comparison_celeba_corrected_gt import build_image_grid

RESULTS_DIR = Path("results")
TOP10_DIR = RESULTS_DIR / "local_attribution_top10_corrected_gt_with_gt"


def main():
    rows = []
    with open(RESULTS_DIR / "local_attribution_celeba_pairs_corrected_gt.csv", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "image": row["image"],
                "concept_name": row["concept_name"],
                "gt_delta_p": float(row["gt_delta_p"]),
                "hybrid_score": float(row["hybrid_score"]),
                "tcav_score": float(row["tcav_score"]),
            })
    print(f"Loaded {len(rows)} scored rows.", flush=True)

    by_concept: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_concept[r["concept_name"]].append(r)

    TOP10_DIR.mkdir(parents=True, exist_ok=True)
    for concept_name, concept_rows in by_concept.items():
        rows_for_grid = [
            ("ground_truth", sorted([(r["image"], r["gt_delta_p"]) for r in concept_rows], key=lambda kv: -abs(kv[1]))[:10]),
            ("hybrid", sorted([(r["image"], r["hybrid_score"]) for r in concept_rows], key=lambda kv: -abs(kv[1]))[:10]),
            ("tcav", sorted([(r["image"], r["tcav_score"]) for r in concept_rows], key=lambda kv: -abs(kv[1]))[:10]),
        ]
        build_image_grid(rows_for_grid, TOP10_DIR / f"{concept_name}.png")

    print(f"Saved {len(by_concept)} grids to {TOP10_DIR}/", flush=True)


if __name__ == "__main__":
    main()
