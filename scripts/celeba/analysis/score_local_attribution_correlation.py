"""Adds Pearson r alongside the existing Spearman rho for the LOCAL
(per-image, per-concept) attribution comparison -- the raw scored-pairs
CSVs already exist (`local_attribution_celeba_pairs.csv`/`_corrected_gt.csv`,
written by `local_attribution_comparison_celeba.py`/`_corrected_gt.py`), so
this reads them directly rather than re-running the expensive TCAV/
embedding pipeline that produced them. Mirrors the aggregate-level
`score_methods_against_attribute_conditioned_gt.py`'s own Pearson addition,
at the individual-pair level instead of the per-(concept,class) mean level.
"""

from __future__ import annotations

import csv
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

RESULTS_DIR = Path("results")
TARGET_CLASSES = ["Attractive", "Young"]
METHOD_COLUMNS = ["baseline_score", "hybrid_score", "tcav_score"]


def report(label: str, csv_name: str) -> None:
    print(f"\n############## {label} ({csv_name}) ##############", flush=True)
    with open(RESULTS_DIR / csv_name, newline="") as f:
        rows = list(csv.DictReader(f))
    for task_name in TARGET_CLASSES:
        task_rows = [r for r in rows if r["target_task"] == task_name]
        gt = [float(r["gt_delta_p"]) for r in task_rows]
        for method in METHOD_COLUMNS:
            scores = [float(r[method]) for r in task_rows]
            rho, sp_p = spearmanr(gt, scores)
            r, pe_p = pearsonr(gt, scores)
            print(f"  [{task_name}] {method:<14s} n={len(task_rows)} rho={rho:+.4f} (p={sp_p:.4g})  "
                  f"pearson_r={r:+.4f} (p={pe_p:.4g})", flush=True)


def main():
    report("ORIGINAL ground truth", "local_attribution_celeba_pairs.csv")
    report("CORRECTED ground truth (attribute-conditioned)", "local_attribution_celeba_pairs_corrected_gt.csv")


if __name__ == "__main__":
    main()
