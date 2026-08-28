"""Top-1 (and top-k) concept IDENTIFICATION accuracy per class, instead
of magnitude-based correlation -- prompted directly ("instead of trying
to compare magnitudes, we try to see if the model is identifying the
correct concepts? Identify the top concepts for each class and see what
percentage of time they are the top reported concept by the attribution
models?").

For each class (species), restricts to the concepts that actually have
ground-truth delta_p measured for that class (only concepts TRUE for
that species get sampled at all, per run_cub_attribute_faithfulness.py's
own design -- so this is never "all 87 concepts," only the subset the
ground truth actually covers for that class). Within that subset, finds
the ground truth's own #1 concept (highest mean delta_p) and checks
whether each method's own #1 concept (highest raw score/weight,
re-ranked over the SAME subset, not the method's full 87/112-concept
matrix -- an unfair advantage otherwise, crediting/blaming a method for
concepts never measured) matches it. Reports exact top-1 match rate,
top-3 recall (is the method's #1 pick within the ground truth's own
top-3?), and a random-guessing baseline for context (chance depends on
how many concepts are available per class, so it's computed per-class
and averaged, not a single global number).
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
MIN_CONCEPTS_PER_CLASS = 2  # need >=2 choices for "top-1" to mean anything


def load_ground_truth() -> dict[int, dict[int, float]]:
    """class -> {concept_idx: mean_delta_p}, only pairs with >=MIN_SAMPLES."""
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
    """class -> {concept_idx: score}."""
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
    weight = ckpt.classifier.weight.detach().cpu().numpy()  # (200, 112)
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
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    chance_rates = []
    chance5_rates = []
    n_concepts_per_class = []
    n_classes_evaluated = 0

    for cls, gt_concepts in gt.items():
        if cls not in method_scores:
            continue
        # restrict to concepts BOTH ground truth AND this method have
        common_concepts = [c for c in gt_concepts if c in method_scores[cls]]
        if len(common_concepts) < MIN_CONCEPTS_PER_CLASS:
            continue

        gt_ranked = sorted(common_concepts, key=lambda c: gt_concepts[c], reverse=True)
        gt_top1 = gt_ranked[0]
        gt_top3 = set(gt_ranked[:3])
        gt_top5 = set(gt_ranked[:5])

        method_ranked = sorted(common_concepts, key=lambda c: method_scores[cls][c], reverse=True)
        method_top1 = method_ranked[0]

        n_classes_evaluated += 1
        n = len(common_concepts)
        n_concepts_per_class.append(n)
        chance_rates.append(1.0 / n)
        chance5_rates.append(min(5, n) / n)  # chance of a random top-1 pick landing in a random top-5 window
        if method_top1 == gt_top1:
            top1_hits += 1
        if method_top1 in gt_top3:
            top3_hits += 1
        if method_top1 in gt_top5:
            top5_hits += 1

    if n_classes_evaluated == 0:
        print(f"{method_name}: no evaluable classes")
        return
    top1_rate = top1_hits / n_classes_evaluated
    top3_rate = top3_hits / n_classes_evaluated
    top5_rate = top5_hits / n_classes_evaluated
    mean_chance1 = float(np.mean(chance_rates))
    mean_chance5 = float(np.mean(chance5_rates))
    mean_n = float(np.mean(n_concepts_per_class))
    print(f"{method_name:<38s} n_classes={n_classes_evaluated:>4d}  avg_concepts/class={mean_n:>5.1f}")
    print(f"{'':<38s} top1={top1_rate:>6.1%} (chance={mean_chance1:>6.1%})   "
          f"top3={top3_rate:>6.1%}   top5={top5_rate:>6.1%} (chance={mean_chance5:>6.1%})")


def main():
    gt = load_ground_truth()
    n_classes_with_gt = len([c for c in gt if len(gt[c]) >= MIN_CONCEPTS_PER_CLASS])
    print(f"{n_classes_with_gt} classes have >={MIN_CONCEPTS_PER_CLASS} ground-truth-measured concepts.\n")

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
