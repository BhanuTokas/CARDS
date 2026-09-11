"""Rebuilds the local-attribution top-10 grids with the `baseline` row
replaced by REAL ground-truth masking results (gt_delta_p), prompted
directly ("could you replace the baseline with ground truth masking
results?") -- a more informative comparison than method-vs-method: does
each method's own top-10 picks actually match what real masking-based
faithfulness says matters most for that concept, not just how the three
methods compare to each other.

Reuses `results/local_attribution_celeba_pairs.csv` directly (already
has gt_delta_p, hybrid_score, tcav_score per row from
`local_attribution_comparison_celeba.py`) -- no model recompute needed,
pure re-aggregation, same as the normalization ablation. Rows with both
tasks present are pooled together per concept (matching the original
grid's own convention), so an image can appear under both its own
Attractive and Young gt_delta_p/scores independently.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from local_attribution_comparison_celeba import build_image_grid

RESULTS_DIR = Path("results")
TOP10_DIR = RESULTS_DIR / "local_attribution_top10_with_gt"


def main():
    rows = []
    with open(RESULTS_DIR / "local_attribution_celeba_pairs.csv", newline="") as f:
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
            ("ground_truth", sorted([(r["image"], r["gt_delta_p"]) for r in concept_rows], key=lambda kv: -kv[1])[:10]),
            ("hybrid", sorted([(r["image"], r["hybrid_score"]) for r in concept_rows], key=lambda kv: -kv[1])[:10]),
            ("tcav", sorted([(r["image"], r["tcav_score"]) for r in concept_rows], key=lambda kv: -kv[1])[:10]),
        ]
        build_image_grid(rows_for_grid, TOP10_DIR / f"{concept_name}.png")

    print(f"Saved {len(by_concept)} grids to {TOP10_DIR}/", flush=True)


if __name__ == "__main__":
    main()
