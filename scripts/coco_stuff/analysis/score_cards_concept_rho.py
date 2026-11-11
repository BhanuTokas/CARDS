"""Concept-wise rho: for each concept, rank 20 classes by CARDS score vs mean delta_p.
Average rho across 103 concepts. Select best alpha on this metric and report.

Usage:
    cd /scratch/rnair21/CARDS
    /scratch/rnair21/envs/cards_env/bin/python scripts/coco_stuff/analysis/score_cards_concept_rho.py

Needs:
    results/coco_stuff/alpha_ablation_scores.csv
    results/coco_stuff/coco_stuff_faithfulness.csv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
from scipy import stats

RESULTS_DIR = Path("results/coco_stuff")
SCORES_CSV  = RESULTS_DIR / "alpha_ablation_scores.csv"
FAITH_CSV   = RESULTS_DIR / "coco_stuff_faithfulness.csv"
MIN_CLASSES = 2
MIN_IMAGES  = 100   # match balanced 103-concept set used in comparison script
STRATEGIES  = ("black_mask", "black_bbox")


def load_faith(
    path: Path, strategy: str
) -> tuple[dict[tuple[str, str], list[float]], dict[str, int]]:
    """Returns:
      faith:        (concept, class) -> [delta_p values]
      image_counts: concept -> number of unique images
    """
    col   = f"delta_p_{strategy}"
    faith: dict[tuple[str, str], list[float]] = defaultdict(list)
    imgs:  dict[str, set[str]]               = defaultdict(set)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            c, cl, img = row["concept_name"], row["class_name"], row["image_path"]
            faith[(c, cl)].append(float(row[col]))
            imgs[c].add(img)
    image_counts = {c: len(s) for c, s in imgs.items()}
    return dict(faith), image_counts


def load_scores(path: Path) -> dict[float, dict[tuple[str, str], float]]:
    """alpha -> {(concept, class): raw_score}"""
    data: dict[float, dict[tuple[str, str], float]] = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data[float(row["alpha"])][(row["concept_name"], row["class_name"])] = float(row["raw_score"])
    return dict(data)


def concept_rho(
    scores:       dict[tuple[str, str], float],
    faith:        dict[tuple[str, str], list[float]],
    image_counts: dict[str, int],
) -> tuple[float, int]:
    """Mean Spearman rho across concepts (each concept ranked over 20 classes).

    Only concepts with >= MIN_IMAGES unique images are included,
    matching the balanced 103-concept set used in the comparison script.
    """
    mean_dp: dict[tuple[str, str], float] = {
        k: float(np.mean(v)) for k, v in faith.items()
    }

    valid_concepts = {c for c, n in image_counts.items() if n >= MIN_IMAGES}
    concepts = sorted(valid_concepts)
    rhos = []

    for concept in concepts:
        s_vals, g_vals = [], []
        for (c, cl), score in scores.items():
            if c != concept:
                continue
            if (c, cl) not in mean_dp:
                continue
            s_vals.append(score)
            g_vals.append(mean_dp[(c, cl)])

        if len(s_vals) < MIN_CLASSES:
            continue

        rho, _ = stats.spearmanr(np.array(s_vals), np.array(g_vals))
        rhos.append(float(rho))

    if not rhos:
        return float("nan"), 0
    return float(np.mean(rhos)), len(rhos)


def pooled_rho(
    scores:       dict[tuple[str, str], float],
    faith:        dict[tuple[str, str], list[float]],
    image_counts: dict[str, int],
) -> tuple[float, float, int]:
    """Single Spearman rho across all valid (concept, class) pairs (103 x 20).

    Returns (rho, p_value, n_pairs).
    """
    valid_concepts = {c for c, n in image_counts.items() if n >= MIN_IMAGES}
    s_vals, g_vals = [], []
    for (c, cl), dp_list in faith.items():
        if c not in valid_concepts:
            continue
        if (c, cl) not in scores:
            continue
        s_vals.append(scores[(c, cl)])
        g_vals.append(float(np.mean(dp_list)))

    if len(s_vals) < 2:
        return float("nan"), float("nan"), 0
    rho, p = stats.spearmanr(np.array(s_vals), np.array(g_vals))
    return float(rho), float(p), len(s_vals)


def main() -> None:
    for strategy in STRATEGIES:
        print(f"\n{'=' * 60}", flush=True)
        print(f"Strategy: {strategy}", flush=True)

        faith, image_counts = load_faith(FAITH_CSV, strategy)
        by_alpha = load_scores(SCORES_CSV)

        n_valid = sum(1 for n in image_counts.values() if n >= MIN_IMAGES)
        print(f"Concepts with >= {MIN_IMAGES} images: {n_valid}", flush=True)
        print(f"Alphas found: {sorted(by_alpha.keys())}", flush=True)

        # --- Per-concept rho (mean across 103 concepts) ---
        print(f"\n-- Per-concept rho (mean of per-concept rhos, each over 20 classes) --", flush=True)
        results: list[tuple[float, float, int]] = []
        for alpha in sorted(by_alpha.keys()):
            mean_rho, n = concept_rho(by_alpha[alpha], faith, image_counts)
            results.append((alpha, mean_rho, n))
            print(f"  alpha={alpha:.1f}  mean_concept_rho={mean_rho:+.4f}  n_concepts={n}", flush=True)
        best = max(results, key=lambda x: x[1])
        print(f"\n  Best alpha={best[0]}  mean_concept_rho={best[1]:+.4f}  (n_concepts={best[2]})", flush=True)

        # --- Pooled rho (single rho across all 103x20 pairs) ---
        print(f"\n-- Pooled rho (single rho across all concept x class pairs) --", flush=True)
        pooled_results: list[tuple[float, float, float, int]] = []
        for alpha in sorted(by_alpha.keys()):
            rho, p, n_pairs = pooled_rho(by_alpha[alpha], faith, image_counts)
            pooled_results.append((alpha, rho, p, n_pairs))
            print(f"  alpha={alpha:.1f}  pooled_rho={rho:+.4f}  p={p:.4g}  n_pairs={n_pairs}", flush=True)
        best_p = max(pooled_results, key=lambda x: x[1])
        print(f"\n  Best alpha={best_p[0]}  pooled_rho={best_p[1]:+.4f}  p={best_p[2]:.4g}  (n_pairs={best_p[3]})", flush=True)


if __name__ == "__main__":
    main()
