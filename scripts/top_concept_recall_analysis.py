"""Recall@N: of the ground truth's own top-3 most important concepts for
a class, what fraction show up in each method's own top-N ranked concept
list (N in {3, 5, 10})? Prompted directly ("should we also test like
presence of top 3 concepts in like top 3, 5 and 10 predicted concepts?"),
extending top_concept_per_class_analysis.py's single-top-1-pick check
(which only asked "does the method's #1 pick match the ground truth's
own #1") to a richer, standard IR-style recall metric that credits a
method for surfacing MULTIPLE genuinely important concepts, not just
nailing the single best one.

Same restriction-to-common-concepts logic as top_concept_per_class_
analysis.py: only concepts BOTH the ground truth and the method have
data for are ranked, so a method is never credited/blamed for concepts
never measured for that class.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "../post_hoc_cbm")

RESULTS_DIR = Path("results")
MIN_SAMPLES = 3
GT_TOP_N = 3  # how many of the ground truth's own top concepts we're checking recall against
METHOD_KS = (3, 5, 10)


def load_ground_truth() -> dict[int, dict[int, float]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    with open(RESULTS_DIR / "cub_attribute_faithfulness.csv", newline="") as f:
        for row in csv.DictReader(f):
            grouped[(int(row["concept_number"]), int(row["predicted_class"]))].append(float(row["delta_p"]))
    by_class: dict[int, dict[int, float]] = defaultdict(dict)
    for (concept, cls), deltas in grouped.items():
        if len(deltas) >= MIN_SAMPLES:
            by_class[cls][concept] = float(np.mean(deltas))
    return by_class


def load_method_csv(fname: str, col: str) -> dict[int, dict[int, float]]:
    by_class: dict[int, dict[int, float]] = defaultdict(dict)
    with open(RESULTS_DIR / fname, newline="") as f:
        for row in csv.DictReader(f):
            by_class[int(row["native_class_idx"])][int(row["attribute_index"])] = float(row[col])
    return by_class


def load_pcbm_official() -> dict[int, dict[int, float]]:
    ckpt = torch.load(
        "../post_hoc_cbm/trained_models_new/cub/resnet18_cub/"
        "pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam_0.0002__alpha_0.99__seed_42__linear.ckpt",
        weights_only=False,
    )
    weight = ckpt.classifier.weight.detach().cpu().numpy()
    by_class: dict[int, dict[int, float]] = defaultdict(dict)
    for cls in range(weight.shape[0]):
        for concept in range(weight.shape[1]):
            by_class[cls][concept] = float(weight[cls, concept])
    return by_class


def load_tcav() -> dict[int, dict[int, float]]:
    by_class: dict[int, dict[int, float]] = defaultdict(dict)
    with open(RESULTS_DIR / "tcav_cub_targeted_v49.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_class[int(row["native_class_idx"])][int(row["attribute_index"])] = float(row["mean_sign_count"])
    return by_class


def evaluate(method_name: str, gt: dict[int, dict[int, float]], method_scores: dict[int, dict[int, float]]) -> None:
    recall_sums = {k: 0.0 for k in METHOD_KS}
    exact_full_recall = {k: 0 for k in METHOD_KS}  # classes where ALL of gt's top-3 were recovered
    n_eval = 0
    chance_recalls = {k: [] for k in METHOD_KS}

    for cls, gt_concepts in gt.items():
        if cls not in method_scores:
            continue
        common = [c for c in gt_concepts if c in method_scores[cls]]
        if len(common) < GT_TOP_N:
            continue

        gt_ranked = sorted(common, key=lambda c: gt_concepts[c], reverse=True)
        gt_top_set = set(gt_ranked[:GT_TOP_N])
        method_ranked = sorted(common, key=lambda c: method_scores[cls][c], reverse=True)

        n_eval += 1
        n_common = len(common)
        for k in METHOD_KS:
            method_top_k = set(method_ranked[:k])
            hit = len(gt_top_set & method_top_k)
            recall = hit / len(gt_top_set)
            recall_sums[k] += recall
            if hit == len(gt_top_set):
                exact_full_recall[k] += 1
            # chance: expected recall if method_top_k were a uniform random k-subset of common
            eff_k = min(k, n_common)
            chance_recalls[k].append(eff_k / n_common)

    if n_eval == 0:
        print(f"{method_name}: no evaluable classes")
        return
    print(f"{method_name:<38s} n_classes={n_eval}")
    for k in METHOD_KS:
        mean_recall = recall_sums[k] / n_eval
        mean_chance = float(np.mean(chance_recalls[k]))
        full_rate = exact_full_recall[k] / n_eval
        print(f"{'':<38s} recall@{k:<3d}= {mean_recall:>6.1%}  (chance={mean_chance:>6.1%})   "
              f"all-3-recovered: {full_rate:.1%}")


def main():
    gt = load_ground_truth()
    n_classes = len([c for c in gt if len(gt[c]) >= GT_TOP_N])
    print(f"{n_classes} classes have >={GT_TOP_N} ground-truth-measured concepts (needed for a top-{GT_TOP_N} target).\n")

    cards = load_method_csv("cards_cub_attribute_scores.csv", "raw_score")
    pcbm_official = load_pcbm_official()
    pcbm_official_siglip = load_method_csv("pcbm_official_siglip_cub_scores.csv", "weight")
    pcbm_siglip_concepts = load_method_csv("pcbm_siglip_concepts_cub_scores.csv", "weight")
    pcbm_clip_concepts = load_method_csv("pcbm_clip_concepts_cub_scores.csv", "weight")
    tcav = load_tcav()

    evaluate("CARDS (SigLIP, K=50, demean=True)", gt, cards)
    evaluate("PCBM (official, resnet18_cub)", gt, pcbm_official)
    evaluate("PCBM (official, SigLIP surrogate)", gt, pcbm_official_siglip)
    evaluate("PCBM (SigLIP-concepts)", gt, pcbm_siglip_concepts)
    evaluate("PCBM (CLIP-concepts)", gt, pcbm_clip_concepts)
    evaluate("TCAV (targeted)", gt, tcav)


if __name__ == "__main__":
    main()
