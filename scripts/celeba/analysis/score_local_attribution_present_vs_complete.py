"""Compares local-attribution Spearman rho on the PRESENT-only set
(`local_attribution_celeba_pairs_corrected_gt.csv`, the original v112
result) against the COMPLETE set (present + the newly-scored absent
images from `score_local_attribution_absent_images.py`), per task per
method. Prompted directly as the natural next step once absent images
had REAL (not assumed-zero) hybrid/TCAV scores: "we can compare
correlation over the complete set."

`baseline` is dropped here (present in the present-only CSV, absent
scores were never computed for it -- out of scope, matching the earlier
decision to exclude it from the top-5/bottom-5 grids too).
"""

from __future__ import annotations

import csv
from pathlib import Path

from scipy.stats import spearmanr

RESULTS_DIR = Path("results")
PRESENT_CSV = RESULTS_DIR / "local_attribution_celeba_pairs_corrected_gt.csv"
ABSENT_CSV = RESULTS_DIR / "local_attribution_celeba_absent_pairs.csv"
METHODS = ["hybrid", "tcav"]
TASKS = ["Attractive", "Young"]


def load_rows(path: Path, presence: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "target_task": row["target_task"],
                "gt_delta_p": float(row["gt_delta_p"]),
                "hybrid_score": float(row["hybrid_score"]),
                "tcav_score": float(row["tcav_score"]),
                "presence": presence,
            })
    return rows


def main():
    present_rows = load_rows(PRESENT_CSV, "present")
    absent_rows = load_rows(ABSENT_CSV, "absent")
    print(f"{len(present_rows)} present rows, {len(absent_rows)} absent rows.", flush=True)

    all_rows = present_rows + absent_rows

    print("\n=== rho: present-only vs. complete set (present + absent), per task per method ===")
    print(f"{'task':<12s}{'method':<10s}{'n_present':>10s}{'rho_present':>13s}{'n_complete':>11s}{'rho_complete':>13s}")
    for task_name in TASKS:
        task_present = [r for r in present_rows if r["target_task"] == task_name]
        task_all = [r for r in all_rows if r["target_task"] == task_name]
        for method_name in METHODS:
            col = f"{method_name}_score"
            gt_p = [r["gt_delta_p"] for r in task_present]
            sc_p = [r[col] for r in task_present]
            rho_p, _ = spearmanr(gt_p, sc_p)

            gt_a = [r["gt_delta_p"] for r in task_all]
            sc_a = [r[col] for r in task_all]
            rho_a, _ = spearmanr(gt_a, sc_a)

            print(f"{task_name:<12s}{method_name:<10s}{len(task_present):>10d}{rho_p:>+13.4f}"
                  f"{len(task_all):>11d}{rho_a:>+13.4f}")


if __name__ == "__main__":
    main()
