"""Builds top-10, Attractive-task-only local-attribution grids for all 26
concepts (appendix material), reusing the same scored CSV as the other
local-attribution galleries -- no recompute, just a task filter + full
concept sweep. Supersedes `local_attribution_top10_corrected_gt_with_gt/`
(which pooled both Attractive and Young rows per concept) now that Young
is dropped from this experiment for consistency with the main-text table
and figure.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from local_attribution_plot_utils import build_image_grid_clean

from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS  # noqa: E402  (after sys.path insert)

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "local_attribution_top10_attractive_only"
TOP_K = 10


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
    for concept_name in GROUNDABLE_CONCEPTS:
        concept_rows = by_concept.get(concept_name, [])
        if not concept_rows:
            print(f"  SKIPPED {concept_name} (no Attractive-task rows found)", flush=True)
            continue
        rows_for_grid = [
            ("ground_truth", sorted([(r["image"], r["gt_delta_p"]) for r in concept_rows], key=lambda kv: -abs(kv[1]))[:TOP_K]),
            ("hybrid", sorted([(r["image"], r["hybrid_score"]) for r in concept_rows], key=lambda kv: -abs(kv[1]))[:TOP_K]),
            ("tcav", sorted([(r["image"], r["tcav_score"]) for r in concept_rows], key=lambda kv: -abs(kv[1]))[:TOP_K]),
        ]
        build_image_grid_clean(rows_for_grid, OUT_DIR / f"{concept_name}.png")
        print(f"  built {concept_name}.png ({len(concept_rows)} Attractive-only pairs available)", flush=True)

    print(f"Saved grids to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
