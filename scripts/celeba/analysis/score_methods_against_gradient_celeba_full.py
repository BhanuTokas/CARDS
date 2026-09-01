"""A different comparison from the rest of this track: instead of scoring
CARDS/TCAV/PCBM against the masking-based delta_p ground truth (what v72/
v73 already did), this scores them against the gradient-attribution
baseline itself (v74) as a substitute reference -- prompted directly
("Can we compare the 3 methods with Gradient as the ground truth?").

A genuinely different question from the rest of this file: not "does
this method predict how the model's own confidence changes when the
concept region is masked" (delta_p, a behavioral ground truth), but
"does this method's own concept-importance ranking agree with what
pixel-level gradient attribution identifies as important" -- a method-
to-method comparison, not a method-vs-model-behavior one. Since v74
found gradient attribution is the only method that actually predicts
delta_p, agreement with it is a weaker, second-order signal: it would
suggest a method is capturing SOMETHING structured (agreement with a
result that itself correlates with real model behavior) even in cases
where its own direct correlation with delta_p was chance-level -- not
proof of faithfulness on its own, since two methods can agree with each
other while both being wrong about the model.

Reuses no delta_p/FaithfulnessResult machinery at all (unlike the rest
of this file's scoring, which all goes through cards.validation.
broden_faithfulness) -- this is a direct Spearman/sign comparison
between two (concept, task) -> scalar score dicts, computed inline here
rather than forcing an ill-fitting reuse of score_method_agreement/
score_sign_agreement (both of which are hard-coded to aggregate FROM
per-image FaithfulnessResult records, which don't exist for this
comparison).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
from scipy.stats import binomtest, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")  # needed to unpickle the saved PosthocLinearCBM checkpoint

from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS, TARGET_CLASSES

RESULTS_DIR = Path("results")
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}


def load_cards_scores() -> dict[str, dict[int, float]]:
    by_task: dict[str, dict[int, float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "cards_celeba_full_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]][CONCEPT_TO_IDX[row["concept_name"]]] = float(row["raw_score"])
    return by_task


def load_tcav_scores() -> dict[str, dict[int, float]]:
    by_task: dict[str, dict[int, float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "tcav_celeba_full_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]][CONCEPT_TO_IDX[row["concept_name"]]] = float(row["mean_sign_count"])
    return by_task


def load_pcbm_scores() -> dict[str, dict[int, float]]:
    by_task: dict[str, dict[int, float]] = {}
    for task_name in TARGET_CLASSES:
        ckpt_path = (
            Path("trained_models_new/celeba_full/celeba_attractive_young")
            / f"pcbm_celeba_full__celeba_attractive_young__{task_name.lower()}__surrogate__seed_42__linear.ckpt"
        )
        posthoc_layer = torch.load(ckpt_path, weights_only=False)
        weight = posthoc_layer.classifier.weight.detach().cpu().numpy()  # (1, n_concepts)
        names = posthoc_layer.names
        by_task[task_name] = {CONCEPT_TO_IDX[name]: float(weight[0, i]) for i, name in enumerate(names)}
    return by_task


def load_gradient_scores() -> dict[str, dict[int, float]]:
    by_task: dict[str, dict[int, float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "gradient_attribution_celeba_full_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]][CONCEPT_TO_IDX[row["concept_name"]]] = float(row["raw_score"])
    return by_task


def compare(method_scores: dict[int, float], gradient_scores: dict[int, float], method_threshold: float) -> dict:
    concepts = sorted(set(method_scores) & set(gradient_scores))
    x = [method_scores[c] for c in concepts]  # the method being evaluated
    y = [gradient_scores[c] for c in concepts]  # gradient attribution, standing in for ground truth here

    rho, rho_p = spearmanr(x, y)
    n_agree = sum(1 for xi, yi in zip(x, y) if (xi > method_threshold) == (yi > 0))
    n = len(concepts)
    sign_result = binomtest(n_agree, n, p=0.5, alternative="two-sided")

    return {
        "n_pairs": n,
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "n_agree": n_agree,
        "agreement_frac": n_agree / n,
        "sign_p": float(sign_result.pvalue),
    }


def main():
    cards_by_task = load_cards_scores()
    tcav_by_task = load_tcav_scores()
    pcbm_by_task = load_pcbm_scores()
    gradient_by_task = load_gradient_scores()

    methods: list[tuple[str, dict[str, dict[int, float]], float]] = [
        ("CARDS", cards_by_task, 0.0),
        ("TCAV", tcav_by_task, 0.5),
        ("PCBM", pcbm_by_task, 0.0),
    ]

    rows = []
    print(f"{'method':<6s} {'task':<11s} {'n':>4s} {'rho':>8s} {'rho_p':>8s} {'sign':>8s} {'n_agree':>9s} {'sign_p':>8s}")
    for method_name, scores_by_task, threshold in methods:
        for task_name in TARGET_CLASSES:
            result = compare(scores_by_task[task_name], gradient_by_task[task_name], threshold)
            rows.append({"method": method_name, "target_task": task_name, **result})
            print(f"{method_name:<6s} {task_name:<11s} {result['n_pairs']:>4d} "
                  f"{result['spearman_rho']:>+8.3f} {result['spearman_p']:>8.3f} "
                  f"{result['agreement_frac']:>7.1%} {result['n_agree']:>3d}/{result['n_pairs']:<5d} "
                  f"{result['sign_p']:>8.3f}")

    with open(RESULTS_DIR / "methods_vs_gradient_celeba_full.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved results/methods_vs_gradient_celeba_full.csv")


if __name__ == "__main__":
    main()
