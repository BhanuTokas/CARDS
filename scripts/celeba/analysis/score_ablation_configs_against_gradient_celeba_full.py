"""Extends v74b's method-vs-Gradient comparison (score_methods_against_
gradient_celeba_full.py, which only checked CARDS' PRODUCTION config)
across all 6 retrieval-strategy/demean configs v76's own ablation
(ablate_cards_celeba_retrieval_strategy.py) already computed -- prompted
directly ("Can we ablate the results against the Integrated Gradients
baseline?"). Same question v76 asked against the masking-based delta_p
ground truth, asked here against the Integrated Gradients baseline (v74)
instead: does ANY alternative CARDS retrieval config agree with IG
better than production does, even though v76 already found none of them
beat delta_p?

Reuses v76's own raw per-concept scores (cards_celeba_retrieval_strategy_
ablation_raw_scores.csv) rather than re-running retrieval -- that file
was added specifically so a comparison like this one wouldn't need to
repeat the expensive part of the ablation.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from scipy.stats import binomtest, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS, TARGET_CLASSES

RESULTS_DIR = Path("results")
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}


def load_ablation_raw_scores() -> dict[tuple[str, str], dict[int, float]]:
    """(config, target_task) -> {concept_idx: raw_score}."""
    by_config_task: dict[tuple[str, str], dict[int, float]] = {}
    with open(RESULTS_DIR / "cards_celeba_retrieval_strategy_ablation_raw_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["config"], row["target_task"])
            by_config_task.setdefault(key, {})[CONCEPT_TO_IDX[row["concept_name"]]] = float(row["raw_score"])
    return by_config_task


def load_gradient_scores() -> dict[str, dict[int, float]]:
    by_task: dict[str, dict[int, float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "gradient_attribution_celeba_full_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]][CONCEPT_TO_IDX[row["concept_name"]]] = float(row["raw_score"])
    return by_task


def compare(method_scores: dict[int, float], gradient_scores: dict[int, float]) -> dict:
    concepts = sorted(set(method_scores) & set(gradient_scores))
    x = [method_scores[c] for c in concepts]
    y = [gradient_scores[c] for c in concepts]

    rho, rho_p = spearmanr(x, y)
    n_agree = sum(1 for xi, yi in zip(x, y) if (xi > 0) == (yi > 0))
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
    ablation_scores = load_ablation_raw_scores()
    gradient_by_task = load_gradient_scores()

    configs = sorted({config for config, _task in ablation_scores})
    rows = []
    print(f"{'config':<16s} {'task':<11s} {'n':>4s} {'rho':>8s} {'rho_p':>8s} {'sign':>8s} {'n_agree':>9s} {'sign_p':>8s}")
    for config in configs:
        for task_name in TARGET_CLASSES:
            method_scores = ablation_scores[(config, task_name)]
            result = compare(method_scores, gradient_by_task[task_name])
            rows.append({"config": config, "target_task": task_name, **result})
            print(f"{config:<16s} {task_name:<11s} {result['n_pairs']:>4d} "
                  f"{result['spearman_rho']:>+8.3f} {result['spearman_p']:>8.3f} "
                  f"{result['agreement_frac']:>7.1%} {result['n_agree']:>3d}/{result['n_pairs']:<5d} "
                  f"{result['sign_p']:>8.3f}")

    with open(RESULTS_DIR / "ablation_configs_vs_gradient_celeba_full.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved results/ablation_configs_vs_gradient_celeba_full.csv")


if __name__ == "__main__":
    main()
