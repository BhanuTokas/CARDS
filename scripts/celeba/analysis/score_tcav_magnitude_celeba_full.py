"""Rescores TCAV using its own `mean_magnitude` (the raw signed mean
directional derivative, already computed and saved by run_tcav_celeba_
full.py -- captum's TCAV.interpret() computes both `sign_count` (fraction
of images with a positive dot product) and `magnitude` (mean of the
signed dot product itself) as a side effect of the same call, but only
sign_count was ever used for scoring) instead of `mean_sign_count` (a
coarse [0,1] fraction) for the Spearman rho comparison, prompted directly
("for TCAV ... it also calculates the gradient. Should we compare the
gradient for rho?"). No new compute -- magnitude was already sitting in
`tcav_celeba_full_scores.csv`.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS, TARGET_CLASSES
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

RESULTS_DIR = Path("results")
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}


def load_records_by_task() -> dict[str, list[FaithfulnessResult]]:
    by_task: dict[str, list[FaithfulnessResult]] = {t: [] for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "celeba_full_faithfulness.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]].append(FaithfulnessResult(
                image=row["image"], concept_number=CONCEPT_TO_IDX[row["concept_name"]], category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return by_task


def main():
    records_by_task = load_records_by_task()

    sign_count_by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    magnitude_by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "tcav_celeba_full_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            key = (CONCEPT_TO_IDX[row["concept_name"]], 1)
            sign_count_by_task[row["target_task"]][key] = float(row["mean_sign_count"])
            magnitude_by_task[row["target_task"]][key] = float(row["mean_magnitude"])

    for task_name in TARGET_CLASSES:
        print(f"\n=== {task_name} ===")
        rho_sc = score_method_agreement(records_by_task[task_name], sign_count_by_task[task_name])
        sign_sc = score_sign_agreement(records_by_task[task_name], sign_count_by_task[task_name], method_threshold=0.5)
        print(f"TCAV sign_count (current): n={rho_sc.n_pairs} rho={rho_sc.spearman_rho:+.4f} p={rho_sc.spearman_p:.4g} "
              f"| sign={sign_sc.agreement_frac:.1%} ({sign_sc.n_agree}/{sign_sc.n_pairs}) p={sign_sc.binom_p:.4g}")

        rho_mag = score_method_agreement(records_by_task[task_name], magnitude_by_task[task_name])
        sign_mag = score_sign_agreement(records_by_task[task_name], magnitude_by_task[task_name], method_threshold=0.0)
        print(f"TCAV magnitude (new):      n={rho_mag.n_pairs} rho={rho_mag.spearman_rho:+.4f} p={rho_mag.spearman_p:.4g} "
              f"| sign={sign_mag.agreement_frac:.1%} ({sign_mag.n_agree}/{sign_mag.n_pairs}) p={sign_mag.binom_p:.4g}")


if __name__ == "__main__":
    main()
