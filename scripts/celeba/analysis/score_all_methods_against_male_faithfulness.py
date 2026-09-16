"""Full 3-method comparison for the Male task, mirroring `score_all_
methods_against_celeba_faithfulness.py`'s own Phase-7 design exactly
(score_method_agreement + score_sign_agreement, one shared function for
all 3 methods) -- prompted directly ("Can we extend to add Male to our
analysis?" -> "Full 3-method comparison").

CARDS/masking-hybrid: `cards_celeba_masking_hybrid_male_raw_scores.csv`
(v113's winning config). TCAV: `tcav_celeba_male_scores.csv`'s own
`mean_sign_count` (matching the original script's own convention,
method_threshold=0.5 since sign_count is a [0,1] fraction, not a
signed quantity). PCBM: the saved surrogate checkpoint's own learned
weight vector for Male (weight[0, :], NOT weight[1, :] -- the
already-known binary-task gotcha from earlier PCBM work in this track).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")  # needed so torch.load can unpickle the saved PosthocLinearCBM class

from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

RESULTS_DIR = Path("results")
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
PCBM_PATH = Path("trained_models_new/celeba_full/celeba_attractive_young_male/"
                  "pcbm_celeba_full__celeba_attractive_young_male__male__surrogate__seed_42__linear.ckpt")


def load_records() -> list[FaithfulnessResult]:
    records = []
    with open(RESULTS_DIR / "celeba_male_faithfulness_attribute_conditioned.csv", newline="") as f:
        for row in csv.DictReader(f):
            records.append(FaithfulnessResult(
                image=row["image"], concept_number=CONCEPT_TO_IDX[row["concept_name"]], category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return records


def main():
    records = load_records()

    cards_scores = {}
    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_male_raw_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            cards_scores[(CONCEPT_TO_IDX[row["concept_name"]], 1)] = float(row["hybrid_raw_score"])

    tcav_scores = {}
    with open(RESULTS_DIR / "tcav_celeba_male_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            tcav_scores[(CONCEPT_TO_IDX[row["concept_name"]], 1)] = float(row["mean_sign_count"])

    pcbm_layer = torch.load(PCBM_PATH, weights_only=False)
    # weight[0, :] is the task-positive-class row, NOT weight[1, :] -- the
    # already-known binary-task gotcha ("If touching PCBM again on a
    # binary-task backbone, remember this") from earlier PCBM work in
    # this track (project_celeba_track memory, Phase-6/pilot section).
    weight = pcbm_layer.classifier.weight.detach().cpu().numpy()
    print(f"PCBM weight shape: {weight.shape}", flush=True)
    pcbm_scores = {}
    for i, concept_name in enumerate(pcbm_layer.names):
        pcbm_scores[(CONCEPT_TO_IDX[concept_name], 1)] = float(weight[0, i])

    methods = {
        "CARDS (masking hybrid)": (cards_scores, 0.0),
        "TCAV": (tcav_scores, 0.5),
        "PCBM": (pcbm_scores, 0.0),
    }

    all_rows = []
    print(f"\n{'method':<25s} {'n':>4s} {'rho':>9s} {'rho_p':>10s} {'sign':>8s} {'sign_p':>10s}")
    for method_name, (scores, threshold) in methods.items():
        rho_r = score_method_agreement(records, scores)
        sign_r = score_sign_agreement(records, scores, method_threshold=threshold)
        if rho_r is None:
            print(f"{method_name:<25s} too few pairs")
            continue
        print(f"{method_name:<25s} {rho_r.n_pairs:>4d} {rho_r.spearman_rho:>+9.4f} {rho_r.spearman_p:>10.4g} "
              f"{sign_r.agreement_frac:>7.1%} {sign_r.binom_p:>10.4g}")
        all_rows.append((method_name, rho_r.n_pairs, rho_r.spearman_rho, rho_r.spearman_p,
                          sign_r.agreement_frac, sign_r.n_agree, sign_r.binom_p))

    with open(RESULTS_DIR / "score_all_methods_against_male_faithfulness.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        writer.writerows(all_rows)
    print("\nSaved to results/score_all_methods_against_male_faithfulness.csv")


if __name__ == "__main__":
    main()
