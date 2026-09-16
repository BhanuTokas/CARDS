"""Normalization ablation for ConceptMask (cfg.scoring_mode ==
"masking_hybrid") on CelebA -- the paired-effect-size normalization the
original pipeline-integration design doc explicitly deferred ("Step 7's
two existing formulas assume a present-vs-absent output-variance framing
that doesn't cleanly transfer to same-image paired deltas -- skip rather
than misapply; a future paired-effect-size normalization over
delta_scores, Cohen's-d-style, is a legitimate later addition, not part
of this change").

Reuses `results/local_attribution_celeba_pairs_corrected_gt.csv`'s own
per-image `hybrid_score` column (built by `local_attribution_comparison_
celeba_corrected_gt.py`, same v96 best config -- alpha=1.0, orth=True,
K=50-ish present set, SigLIP -- as every other "ConceptMask"/masking-
hybrid result in this track) rather than re-running the expensive
localize+mask+score pipeline: 52 (concept, task) groups, 50-90 images
each, plenty for a within-group std.

Two scores per (concept, task), both derived from the SAME per-image
delta_scores:
  - raw: mean(delta_scores) -- what every existing masking-hybrid CSV in
    this track already reports.
  - normalized: mean(delta_scores) / std(delta_scores) -- a PAIRED
    Cohen's d (same-image before/after, not the present-vs-absent pooled
    std cards.attribution.normalization.variance_normalize uses), using
    the same _EPSILON guard against near-zero std.

Both scored against the corrected, attribute-conditioned ground truth
(Spearman + Pearson + sign agreement), printed side by side.
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
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
_EPSILON = 1e-8


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


def load_raw_and_normalized_scores() -> tuple[dict[str, dict], dict[str, dict]]:
    """Groups local_attribution_celeba_pairs_corrected_gt.csv's per-image
    hybrid_score by (concept, task) -> raw (mean) and normalized
    (mean/std, paired Cohen's d) score tables, both in score_method_
    agreement's own (concept_idx, 1) -> scalar shape."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    with open(RESULTS_DIR / "local_attribution_celeba_pairs_corrected_gt.csv", newline="") as f:
        for row in csv.DictReader(f):
            grouped[(row["concept_name"], row["target_task"])].append(float(row["hybrid_score"]))

    raw: dict[str, dict] = {t: {} for t in TARGET_CLASSES}
    normalized: dict[str, dict] = {t: {} for t in TARGET_CLASSES}
    n_near_zero_std = 0
    for (concept_name, task_name), deltas in grouped.items():
        key = (CONCEPT_TO_IDX[concept_name], 1)
        mean = statistics.mean(deltas)
        std = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        raw[task_name][key] = mean
        if std < _EPSILON:
            n_near_zero_std += 1
            normalized[task_name][key] = 0.0
        else:
            normalized[task_name][key] = mean / std
    if n_near_zero_std:
        print(f"({n_near_zero_std} (concept,task) groups had ~zero std -- normalized score set to 0.0 for those)")
    return raw, normalized


def report(label: str, records_by_task, scores_by_task, method_threshold: float = 0.0):
    print(f"\n=== {label} ===", flush=True)
    for task_name in TARGET_CLASSES:
        rho_r = score_method_agreement(records_by_task[task_name], scores_by_task[task_name])
        if rho_r is None:
            print(f"  [{task_name}] too few pairs")
            continue
        sign_r = score_sign_agreement(records_by_task[task_name], scores_by_task[task_name], method_threshold=method_threshold)
        print(f"  [{task_name}] n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} (p={rho_r.spearman_p:.4g})  "
              f"pearson_r={rho_r.pearson_r:+.4f} (p={rho_r.pearson_p:.4g})  "
              f"sign={sign_r.agreement_frac:.1%} ({sign_r.n_agree}/{sign_r.n_pairs}, p={sign_r.binom_p:.4g})", flush=True)


def main():
    new_records = load_records_by_task("celeba_full_faithfulness_attribute_conditioned.csv")
    raw_scores, normalized_scores = load_raw_and_normalized_scores()

    print("############## ConceptMask (masking hybrid) normalization ablation, corrected ground truth ##############")
    report("raw_score = mean(delta_scores)", new_records, raw_scores)
    report("normalized_score = mean(delta_scores) / std(delta_scores)  [paired Cohen's d]", new_records, normalized_scores)


if __name__ == "__main__":
    main()
