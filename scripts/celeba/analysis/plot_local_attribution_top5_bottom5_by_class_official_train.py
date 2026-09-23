"""Per-CLASS (task) counterpart of `plot_local_attribution_top5_
bottom5_combined_official_train.py` -- same combined present+absent
|magnitude| retrieval grid, but ranked SEPARATELY per target task
(Attractive, Male) instead of pooling both into one ranking per
concept. Prompted directly ("Can you show the top and bottom 5
examples now for each class?"), the natural follow-up once pooling
Attractive+Male was shown to be confounded (score_local_attribution_
present_vs_complete_official_train.py's own class-pooled-rho check --
Male's hybrid_score has ~2.2x the spread of Attractive's, so a POOLED
magnitude ranking would be structurally biased toward surfacing Male
rows regardless of true concept relevance, the same kind of between-
task artifact already flagged for the correlation number).

**Top-3/bottom-3, SIDE BY SIDE in one row per method** (not top-5/
bottom-5 stacked as two separate rows) -- confirmed directly ("can we
get the top and bottom 3 results, but side by side instead of on top
of each other per method??"). Each method now gets exactly one row:
3 top-ranked thumbnails, a vertical divider, then 3 bottom-ranked
thumbnails -- 2 rows total per grid (hybrid, tcav) instead of 4.

Both classes (Attractive AND Male) are already covered here -- TASKS
includes both, one grid saved per (concept, task) pair, `{concept}_
{task}.png`, so Male's own results are produced by this same run, not
a separate pass.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from local_attribution_plot_utils import build_presence_grid_clean

RESULTS_DIR = Path("results")
PRESENT_CSV = RESULTS_DIR / "local_attribution_celeba_official_train_pairs.csv"
ABSENT_CSV = RESULTS_DIR / "local_attribution_celeba_official_train_absent_pairs.csv"
OUT_DIR = RESULTS_DIR / "local_attribution_top3_bottom3_by_class_official_train"
METHODS = ["hybrid", "tcav"]
METHOD_DISPLAY_NAMES = {"hybrid": "ConceptMask (Ours)", "tcav": "TCAV"}
TASKS = ["Attractive", "Male"]
N = 3


def main():
    # candidates[concept][task][method] = [(image, score, is_present), ...]
    candidates: dict[str, dict[str, dict[str, list[tuple[str, float, bool]]]]] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    with open(PRESENT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            for method_name in METHODS:
                candidates[row["concept_name"]][row["target_task"]][method_name].append(
                    (row["image"], float(row[f"{method_name}_score"]), True)
                )
    with open(ABSENT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            for method_name in METHODS:
                candidates[row["concept_name"]][row["target_task"]][method_name].append(
                    (row["image"], float(row[f"{method_name}_score"]), False)
                )

    print(f"{len(candidates)} concepts found, split by class ({TASKS}).", flush=True)
    n_saved = 0
    for concept_name in sorted(candidates):
        for task_name in TASKS:
            rows_for_grid = []
            for method_name in METHODS:
                ranked = sorted(candidates[concept_name][task_name][method_name], key=lambda kv: -abs(kv[1]))
                top_n = [(img, present) for img, _score, present in ranked[:N]]
                bottom_n = [(img, present) for img, _score, present in ranked[-N:]]
                rows_for_grid.append((METHOD_DISPLAY_NAMES[method_name], top_n, bottom_n))
            title = f"{concept_name.replace('_', ' ')} — {task_name}"
            build_presence_grid_clean(title, rows_for_grid, OUT_DIR / f"{concept_name}_{task_name}.png", n_per_block=N)
            n_saved += 1
    print(f"Saved {n_saved} per-(concept,class) grids to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
