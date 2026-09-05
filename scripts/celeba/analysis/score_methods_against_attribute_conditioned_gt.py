"""Re-scores the masking hybrid (v96 best config), TCAV, and PCBM against
the corrected, attribute-conditioned ground truth
(`celeba_full_faithfulness_attribute_conditioned.csv` -- both label
directions pooled, candidates filtered by the concept's own attribute
value, N_PER_ATTRIBUTE=90, see run_celeba_full_faithfulness_attribute_
conditioned.py's own docstring for the full rationale), prompted
directly ("Yes, please score them against the corrected ground truth
please!").

No method needs recomputing -- every method's own score is already a
per-(concept,class) scalar, independent of which images ground truth
was built from (score_method_agreement/score_sign_agreement's own
documented design). This just re-aggregates the NEW ground truth and
re-correlates the SAME already-computed score tables against it,
printed side by side with the ORIGINAL ground truth's own numbers for
direct comparison.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")  # needed to unpickle the saved PosthocLinearCBM checkpoint

from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS, TARGET_CLASSES
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

RESULTS_DIR = Path("results")
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}


def load_records_by_task(csv_name: str) -> dict[str, list[FaithfulnessResult]]:
    by_task: dict[str, list[FaithfulnessResult]] = {t: [] for t in TARGET_CLASSES}
    with open(RESULTS_DIR / csv_name, newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]].append(FaithfulnessResult(
                image=row["image"], concept_number=CONCEPT_TO_IDX[row["concept_name"]], category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return by_task


def load_hybrid_scores() -> dict[str, dict[tuple[int, int], float]]:
    by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_threshold_zscore_orthogonalize_ablation_raw_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            if float(row["k"]) == 1.0:  # this CSV predates the k->alpha rename; real header is still "k"
                by_task[row["target_task"]][(CONCEPT_TO_IDX[row["concept_name"]], 1)] = float(row["hybrid_raw_score"])
    return by_task


def load_tcav_scores() -> dict[str, dict[tuple[int, int], float]]:
    by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "tcav_celeba_full_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]][(CONCEPT_TO_IDX[row["concept_name"]], 1)] = float(row["mean_sign_count"])
    return by_task


def load_hybrid_official_val_scores() -> dict[str, dict[tuple[int, int], float]]:
    by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_official_val_k1_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]][(CONCEPT_TO_IDX[row["concept_name"]], 1)] = float(row["hybrid_raw_score"])
    return by_task


def load_pcbm_scores() -> dict[str, dict[tuple[int, int], float]]:
    by_task: dict[str, dict[tuple[int, int], float]] = {}
    for task_name in TARGET_CLASSES:
        ckpt_path = (
            Path("trained_models_new/celeba_full/celeba_attractive_young")
            / f"pcbm_celeba_full__celeba_attractive_young__{task_name.lower()}__surrogate__seed_42__linear.ckpt"
        )
        posthoc_layer = torch.load(ckpt_path, weights_only=False)
        weight = posthoc_layer.classifier.weight.detach().cpu().numpy()  # (1, n_concepts) binary-task row
        names = posthoc_layer.names
        by_task[task_name] = {(CONCEPT_TO_IDX[name], 1): float(weight[0, i]) for i, name in enumerate(names)}
    return by_task


def report(label: str, records_by_task, scores_by_task, method_threshold: float = 0.0):
    print(f"\n=== {label} ===", flush=True)
    for task_name in TARGET_CLASSES:
        rho_r = score_method_agreement(records_by_task[task_name], scores_by_task[task_name])
        sign_r = score_sign_agreement(records_by_task[task_name], scores_by_task[task_name], method_threshold=method_threshold)
        if rho_r is None:
            print(f"  [{task_name}] too few pairs")
            continue
        print(f"  [{task_name}] n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} (p={rho_r.spearman_p:.4g})  "
              f"sign={sign_r.agreement_frac:.1%} ({sign_r.n_agree}/{sign_r.n_pairs}, p={sign_r.binom_p:.4g})", flush=True)


def main():
    old_records = load_records_by_task("celeba_full_faithfulness.csv")
    new_records = load_records_by_task("celeba_full_faithfulness_attribute_conditioned.csv")

    hybrid_scores = load_hybrid_scores()
    hybrid_official_val_scores = load_hybrid_official_val_scores()
    tcav_scores = load_tcav_scores()
    pcbm_scores = load_pcbm_scores()

    print("############## ORIGINAL ground truth (region-filtered only, target-positive only) ##############")
    report("Masking hybrid, HQ-val (alpha=1.0, orth=True, K=50, SigLIP)", old_records, hybrid_scores)
    report("Masking hybrid, OFFICIAL-val (same config)", old_records, hybrid_official_val_scores)
    report("TCAV (mean_sign_count)", old_records, tcav_scores, method_threshold=0.5)
    report("PCBM (CAV-based, region-crop bank)", old_records, pcbm_scores)

    print("\n############## CORRECTED ground truth (attribute-conditioned, both label directions) ##############")
    report("Masking hybrid, HQ-val (alpha=1.0, orth=True, K=50, SigLIP)", new_records, hybrid_scores)
    report("Masking hybrid, OFFICIAL-val (same config)", new_records, hybrid_official_val_scores)
    report("TCAV (mean_sign_count)", new_records, tcav_scores, method_threshold=0.5)
    report("PCBM (CAV-based, region-crop bank)", new_records, pcbm_scores)


if __name__ == "__main__":
    main()
