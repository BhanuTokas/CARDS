"""Compares the masking hybrid (v81-v83) against TCAV and PCBM, prompted
directly ("Can we compare TCAV and PCBM with it?"). All methods scored
against the SAME full 26-concept x 2-task real-mask ground truth
(`celeba_full_faithfulness.csv`) already used for the v72 headline
comparison and every hybrid result since -- no new compute, just
compiling existing outputs into one table: TCAV/PCBM/plain-CARDS scores
were already saved from the v72 full-scale run; the hybrid's own scores
are v82's SigLIP 7-strategy full run
(`cards_celeba_masking_hybrid_scores_best_of_family_full.csv`).
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


def load_csv_scores(filename: str, score_field: str) -> dict[str, dict[tuple[int, int], float]]:
    by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / filename, newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]][(CONCEPT_TO_IDX[row["concept_name"]], 1)] = float(row[score_field])
    return by_task


def load_pcbm_scores() -> dict[str, dict[tuple[int, int], float]]:
    by_task: dict[str, dict[tuple[int, int], float]] = {}
    for task_name in TARGET_CLASSES:
        ckpt_path = (
            Path("trained_models_new/celeba_full/celeba_attractive_young")
            / f"pcbm_celeba_full__celeba_attractive_young__{task_name.lower()}__surrogate__seed_42__linear.ckpt"
        )
        posthoc_layer = torch.load(ckpt_path, weights_only=False)
        weight = posthoc_layer.classifier.weight.detach().cpu().numpy()  # (1, n_concepts)
        names = posthoc_layer.names
        by_task[task_name] = {(CONCEPT_TO_IDX[name], 1): float(weight[0, i]) for i, name in enumerate(names)}
    return by_task


def evaluate(label, task_name, records, scores, threshold=0.0):
    rho_result = score_method_agreement(records, scores)
    sign_result = score_sign_agreement(records, scores, method_threshold=threshold)
    if rho_result is not None:
        print(f"{label:<28s} {task_name:<12s} n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} "
              f"p={rho_result.spearman_p:.4g} | sign={sign_result.agreement_frac:.1%} "
              f"({sign_result.n_agree}/{sign_result.n_pairs}) binom_p={sign_result.binom_p:.4g}", flush=True)
    else:
        print(f"{label:<28s} {task_name:<12s} too few pairs", flush=True)


def main():
    records_by_task = load_records_by_task()

    methods = {
        "CARDS (plain, cross-image)": (load_csv_scores("cards_celeba_full_scores.csv", "raw_score"), 0.0),
        "TCAV": (load_csv_scores("tcav_celeba_full_scores.csv", "mean_sign_count"), 0.5),
        "PCBM (region-crop, surrogate)": (load_pcbm_scores(), 0.0),
        "Hybrid (SigLIP, best-of-7)": (load_csv_scores("cards_celeba_masking_hybrid_scores_best_of_family_full.csv", "hybrid_raw_score"), 0.0),
    }

    for task_name in TARGET_CLASSES:
        print(f"\n=== {task_name} ===")
        for label, (scores_by_task, threshold) in methods.items():
            evaluate(label, task_name, records_by_task[task_name], scores_by_task[task_name], threshold)


if __name__ == "__main__":
    main()
