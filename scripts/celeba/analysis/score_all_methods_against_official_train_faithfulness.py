"""Full 3-method comparison for the NEW official-train classifier,
against BOTH ground-truth versions, prompted directly ("Do we also have
results of PCBM and TCAV on this version of ground truth?"). Mirrors
`score_all_methods_against_male_faithfulness.py`'s own design (one
shared score_method_agreement/score_sign_agreement call per method) but
loops over both tasks (Attractive, Male) and both GT versions
(non_overlapping, previously_used).

CARDS/masking-hybrid: `cards_celeba_masking_hybrid_official_train_raw_
scores.csv` (already computed, v121). TCAV: `tcav_celeba_official_
train_scores.csv`'s own `mean_sign_count` (method_threshold=0.5).
PCBM: `weight[0, :]` of each task's saved surrogate checkpoint (the
already-known binary-task gotcha -- NOT weight[1, :]).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

RESULTS_DIR = Path("results")
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
TASKS = ["Attractive", "Male"]
GT_VERSIONS = {
    "non_overlapping": "celeba_official_train_faithfulness_non_overlapping.csv",
    "previously_used": "celeba_official_train_faithfulness_previously_used.csv",
}
PCBM_DIR = Path("trained_models_new/celeba_full/celeba_official_train_attractive_male")


def load_records(gt_filename: str, task_name: str) -> list[FaithfulnessResult]:
    records = []
    with open(RESULTS_DIR / gt_filename, newline="") as f:
        for row in csv.DictReader(f):
            if row["target_task"] != task_name:
                continue
            records.append(FaithfulnessResult(
                image=row["image"], concept_number=CONCEPT_TO_IDX[row["concept_name"]], category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return records


def load_cards_scores(task_name: str) -> dict[tuple[int, int], float]:
    scores = {}
    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_official_train_raw_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["task"] != task_name:
                continue
            scores[(CONCEPT_TO_IDX[row["concept_name"]], 1)] = float(row["hybrid_raw_score"])
    return scores


def load_tcav_scores(task_name: str) -> dict[tuple[int, int], float]:
    scores = {}
    with open(RESULTS_DIR / "tcav_celeba_official_train_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["target_task"] != task_name:
                continue
            scores[(CONCEPT_TO_IDX[row["concept_name"]], 1)] = float(row["mean_sign_count"])
    return scores


def load_pcbm_scores(task_name: str) -> dict[tuple[int, int], float]:
    ckpt_path = (PCBM_DIR /
                 f"pcbm_celeba_full__celeba_official_train_attractive_male__{task_name.lower()}__surrogate__seed_42__linear.ckpt")
    pcbm_layer = torch.load(ckpt_path, weights_only=False)
    weight = pcbm_layer.classifier.weight.detach().cpu().numpy()
    return {(CONCEPT_TO_IDX[name], 1): float(weight[0, i]) for i, name in enumerate(pcbm_layer.names)}


def main():
    all_rows = []
    print(f"\n{'task':<12s} {'gt_version':<16s} {'method':<25s} {'n':>4s} {'rho':>9s} {'rho_p':>10s} {'sign':>8s} {'sign_p':>10s}")
    for task_name in TASKS:
        method_scores = {
            "CARDS (masking hybrid)": (load_cards_scores(task_name), 0.0),
            "TCAV": (load_tcav_scores(task_name), 0.5),
            "PCBM": (load_pcbm_scores(task_name), 0.0),
        }
        for gt_name, gt_filename in GT_VERSIONS.items():
            records = load_records(gt_filename, task_name)
            for method_name, (scores, threshold) in method_scores.items():
                rho_r = score_method_agreement(records, scores)
                sign_r = score_sign_agreement(records, scores, method_threshold=threshold)
                if rho_r is None:
                    print(f"{task_name:<12s} {gt_name:<16s} {method_name:<25s} too few pairs")
                    continue
                print(f"{task_name:<12s} {gt_name:<16s} {method_name:<25s} {rho_r.n_pairs:>4d} "
                      f"{rho_r.spearman_rho:>+9.4f} {rho_r.spearman_p:>10.4g} "
                      f"{sign_r.agreement_frac:>7.1%} {sign_r.binom_p:>10.4g}")
                all_rows.append((task_name, gt_name, method_name, rho_r.n_pairs, rho_r.spearman_rho, rho_r.spearman_p,
                                  sign_r.agreement_frac, sign_r.n_agree, sign_r.binom_p))

    out_path = RESULTS_DIR / "score_all_methods_against_official_train_faithfulness.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "gt_version", "method", "n_pairs", "spearman_rho", "spearman_p",
                          "sign_agreement", "n_agree", "binom_p"])
        writer.writerows(all_rows)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
