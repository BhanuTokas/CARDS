"""Pools Attractive AND Male into a SINGLE Spearman correlation (52
concept-task pairs) against the official-train classifier's faithfulness
ground truth, prompted directly ("I meant pooling across classes"). Also
adds Concept Mask's LFW/UTKFace cross-dataset pool rows to the same
pooled-across-tasks view, and diagnoses why the pooled rho is so much
higher than either per-task rho (prompted directly, "why is the class
pooled score so much higher"), plus Fisher-z and plain averaging of the
two per-task rhos for comparison.

`score_method_agreement`'s own (concept_number, predicted_class) pair key
can't be reused directly here: `predicted_class` is 1 for BOTH tasks (the
positive class within each task's own binary head), so keying only by
(concept_number, 1) would silently collide Attractive's and Male's rows
for the same concept into one averaged point. This script instead keys
by (concept_name, task_name) directly, so all 52 points stay distinct.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

RESULTS_DIR = Path("results")
TASKS = ["Attractive", "Male"]
GT_FILENAME = "celeba_official_train_faithfulness_previously_used.csv"
PCBM_DIR = Path("trained_models_new/celeba_full/celeba_official_train_attractive_male")

PER_TASK_RHO = {"Attractive": 0.7019, "Male": 0.5275}  # previously_used GT, v123 (already confirmed)


def load_ground_truth() -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    with open(RESULTS_DIR / GT_FILENAME, newline="") as f:
        for row in csv.DictReader(f):
            grouped[(row["concept_name"], row["target_task"])].append(float(row["delta_p"]))
    return {k: float(np.mean(v)) for k, v in grouped.items()}


def load_cards_scores() -> dict[tuple[str, str], float]:
    scores = {}
    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_official_train_raw_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            scores[(row["concept_name"], row["task"])] = float(row["hybrid_raw_score"])
    return scores


def load_tcav_scores() -> dict[tuple[str, str], float]:
    scores = {}
    with open(RESULTS_DIR / "tcav_celeba_official_train_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            scores[(row["concept_name"], row["target_task"])] = float(row["mean_sign_count"])
    return scores


def load_pcbm_resnet18_scores() -> dict[tuple[str, str], float]:
    scores = {}
    for task_name in TASKS:
        ckpt_path = (PCBM_DIR /
                     f"pcbm_celeba_full__celeba_official_train_attractive_male__{task_name.lower()}__surrogate__seed_42__linear.ckpt")
        pcbm_layer = torch.load(ckpt_path, weights_only=False)
        weight = pcbm_layer.classifier.weight.detach().cpu().numpy()
        for i, name in enumerate(pcbm_layer.names):
            scores[(name, task_name)] = float(weight[0, i])
    return scores


def load_clip_concepts_scores(backbone: str) -> dict[tuple[str, str], float]:
    scores = {}
    with open(RESULTS_DIR / f"pcbm_clip_concepts_celeba_official_train_{backbone}_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            scores[(row["concept_name"], row["target_task"])] = float(row["weight"])
    return scores


def load_cross_dataset_scores(pool_name: str) -> dict[tuple[str, str], float]:
    scores = {}
    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_official_train_cross_dataset_pool_raw_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["pool_name"] != pool_name:
                continue
            scores[(row["concept_name"], row["task"])] = float(row["hybrid_raw_score"])
    return scores


def pooled_rho(gt: dict[tuple[str, str], float], scores: dict[tuple[str, str], float]):
    common = sorted(set(gt) & set(scores))
    gt_vals = [gt[k] for k in common]
    score_vals = [scores[k] for k in common]
    rho, p = spearmanr(gt_vals, score_vals)
    return len(common), rho, p


def per_task_rho(gt: dict[tuple[str, str], float], scores: dict[tuple[str, str], float], task: str):
    common = sorted(k for k in (set(gt) & set(scores)) if k[1] == task)
    gt_vals = [gt[k] for k in common]
    score_vals = [scores[k] for k in common]
    rho, p = spearmanr(gt_vals, score_vals)
    return len(common), rho, p


def fisher_average(rhos: list[float], ns: list[int]) -> float:
    zs = [np.arctanh(r) for r in rhos]
    weights = [n - 3 for n in ns]
    z_bar = float(np.average(zs, weights=weights))
    return float(np.tanh(z_bar))


def main():
    gt = load_ground_truth()
    print(f"Ground truth ({GT_FILENAME}): {len(gt)} (concept, task) points (expect 52)", flush=True)

    methods = {
        "CARDS (masking hybrid)": load_cards_scores(),
        "TCAV": load_tcav_scores(),
        "PCBM (ResNet-18)": load_pcbm_resnet18_scores(),
        "PCBM (CLIP-RN50)": load_clip_concepts_scores("clip_rn50"),
        "PCBM (SigLIP)": load_clip_concepts_scores("siglip"),
        "Concept Mask (LFW pool)": load_cross_dataset_scores("lfw"),
        "Concept Mask (UTKFace pool)": load_cross_dataset_scores("utkface"),
    }

    print(f"\n{'method':<30s} {'n':>4s} {'rho':>9s} {'p':>10s}")
    for name, scores in methods.items():
        n, rho, p = pooled_rho(gt, scores)
        print(f"{name:<30s} {n:>4d} {rho:>+9.4f} {p:>10.4g}")

    print("\n=== why is pooled rho >> either per-task rho? group-level diagnostic ===", flush=True)
    cards_scores = methods["CARDS (masking hybrid)"]
    for task in TASKS:
        gt_vals = [v for (c, t), v in gt.items() if t == task]
        score_vals = [cards_scores[(c, t)] for (c, t) in gt if t == task]
        print(f"{task:<12s} n={len(gt_vals)}  GT delta_p: mean={np.mean(gt_vals):+.4f} std={np.std(gt_vals):.4f}  "
              f"CARDS raw_score: mean={np.mean(score_vals):+.4f} std={np.std(score_vals):.4f}", flush=True)

    print("\n=== Fisher-z vs. plain averaging of the two per-task CARDS rhos (previously_used) ===", flush=True)
    r_att, r_male = PER_TASK_RHO["Attractive"], PER_TASK_RHO["Male"]
    plain_avg = (r_att + r_male) / 2
    fisher_avg = fisher_average([r_att, r_male], [26, 26])
    n_pool, rho_pool, p_pool = pooled_rho(gt, cards_scores)
    print(f"per-task: Attractive={r_att:+.4f}  Male={r_male:+.4f}", flush=True)
    print(f"plain average:        {plain_avg:+.4f}", flush=True)
    print(f"Fisher-z average:     {fisher_avg:+.4f}", flush=True)
    print(f"actual pooled (n=52): {rho_pool:+.4f}", flush=True)

    print("\n=== FULL TABLE: per-task rho, plain avg, Fisher-z avg, actual pooled -- every method ===", flush=True)
    header = f"{'method':<30s} {'Attractive':>11s} {'Male':>9s} {'PlainAvg':>9s} {'FisherAvg':>10s} {'Pooled(n=52)':>13s} {'pooled_p':>10s}"
    print(header, flush=True)
    for name, scores in methods.items():
        n_a, rho_a, _ = per_task_rho(gt, scores, "Attractive")
        n_m, rho_m, _ = per_task_rho(gt, scores, "Male")
        plain = (rho_a + rho_m) / 2
        fisher = fisher_average([rho_a, rho_m], [n_a, n_m])
        n_p, rho_p, p_p = pooled_rho(gt, scores)
        print(f"{name:<30s} {rho_a:>+11.4f} {rho_m:>+9.4f} {plain:>+9.4f} {fisher:>+10.4f} {rho_p:>+13.4f} {p_p:>10.4g}", flush=True)


if __name__ == "__main__":
    main()
