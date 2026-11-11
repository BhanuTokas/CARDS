"""Faithfulness comparison across all methods: mean class rho and pooled rho.

CARDS: full alpha sweep (0.5 – 2.5) from alpha_ablation_scores.csv, best alpha
       selected by pooled rho and shown in the comparison table.
Other methods: single-condition from their respective CSVs.

Metrics (both reported per strategy: black_mask, black_bbox):
  mean class rho  -- for each of 20 classes, spearmanr over ~103 concepts; average
  pooled rho      -- single spearmanr over all valid (concept, class) pairs (~103 x 20)

Usage:
    cd /scratch/rnair21/CARDS
    /scratch/rnair21/envs/cards_env/bin/python \
        scripts/coco_stuff/analysis/compare_all_methods_faithfulness.py

Needs:
    results/coco_stuff/alpha_ablation_scores.csv
    results/coco_stuff/coco_stuff_faithfulness.csv
    results/coco_stuff/tcav_coco_stuff_scores.csv          (optional)
    results/coco_stuff/pcbm_resnet_concept_weights.csv     (optional)
    results/coco_stuff/pcbm_clip_concept_weights.csv       (optional)
    results/coco_stuff/pcbm_siglip_concept_weights.csv     (optional)
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
from scipy import stats

from cards.data.coco_stuff import COCO_STUFF_20_CLASSES

RESULTS_DIR  = Path("results/coco_stuff")
ALPHA_CSV    = RESULTS_DIR / "alpha_ablation_scores.csv"
FAITH_CSV    = RESULTS_DIR / "coco_stuff_faithfulness.csv"
MIN_IMAGES   = 100
MIN_SAMPLES  = 3
MIN_CONCEPTS = 2
STRATEGIES   = ("black_mask", "black_bbox")

OTHER_METHODS: list[tuple[str, Path, str]] = [
    ("TCAV",        RESULTS_DIR / "tcav_coco_stuff_scores.csv",      "mean_sign_count"),
    ("PCBM-resnet", RESULTS_DIR / "pcbm_resnet_concept_weights.csv", "weight"),
    ("PCBM-CLIP",   RESULTS_DIR / "pcbm_clip_concept_weights.csv",   "weight"),
    ("PCBM-SigLIP", RESULTS_DIR / "pcbm_siglip_concept_weights.csv", "weight"),
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_faith(
    path: Path, strategy: str
) -> tuple[dict[tuple[str, str], list[float]], dict[str, int]]:
    col = f"delta_p_{strategy}"
    concept_imgs: dict[str, set[str]] = defaultdict(set)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            concept_imgs[row["concept_name"]].add(row["image_path"])
    valid_imgs: dict[str, set[str]] = {
        c: set(sorted(imgs)[:MIN_IMAGES])
        for c, imgs in concept_imgs.items()
        if len(imgs) >= MIN_IMAGES
    }
    faith: dict[tuple[str, str], list[float]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            c, cl, img = row["concept_name"], row["class_name"], row["image_path"]
            if c in valid_imgs and img in valid_imgs[c]:
                faith[(c, cl)].append(float(row[col]))
    image_counts = {c: MIN_IMAGES for c in valid_imgs}
    return dict(faith), image_counts


def load_alpha_scores(path: Path) -> dict[float, dict[tuple[str, str], float]]:
    """alpha -> {(concept, class): raw_score}"""
    data: dict[float, dict[tuple[str, str], float]] = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data[float(row["alpha"])][(row["concept_name"], row["class_name"])] = float(row["raw_score"])
    return dict(data)


def load_method_scores(path: Path, score_col: str) -> dict[tuple[str, str], float]:
    """(concept, class) -> score from any flat CSV."""
    scores: dict[tuple[str, str], float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            scores[(row["concept_name"], row["class_name"])] = float(row[score_col])
    return scores


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def class_rho(
    scores:       dict[tuple[str, str], float],
    faith:        dict[tuple[str, str], list[float]],
    image_counts: dict[str, int],
) -> tuple[float, int]:
    """Mean Spearman rho across 20 classes (each rho over ~103 balanced concepts)."""
    valid   = {c for c, n in image_counts.items() if n >= MIN_IMAGES}
    rhos    = []
    for cls_name in COCO_STUFF_20_CLASSES:
        s_vals, g_vals = [], []
        for (c, cl), dp_list in faith.items():
            if cl != cls_name or c not in valid or (c, cl) not in scores:
                continue
            if len(dp_list) < MIN_SAMPLES:
                continue
            s_vals.append(scores[(c, cl)])
            g_vals.append(float(np.mean(dp_list)))
        if len(s_vals) < MIN_CONCEPTS:
            continue
        rho, _ = stats.spearmanr(np.array(s_vals), np.array(g_vals))
        rhos.append(float(rho))
    return (float(np.mean(rhos)), len(rhos)) if rhos else (float("nan"), 0)


def pooled_rho(
    scores:       dict[tuple[str, str], float],
    faith:        dict[tuple[str, str], list[float]],
    image_counts: dict[str, int],
) -> tuple[float, float, int]:
    """Single Spearman rho across all valid (concept, class) pairs."""
    valid        = {c for c, n in image_counts.items() if n >= MIN_IMAGES}
    s_vals, g_vals = [], []
    for (c, cl), dp_list in faith.items():
        if c not in valid or (c, cl) not in scores:
            continue
        if len(dp_list) < MIN_SAMPLES:
            continue
        s_vals.append(scores[(c, cl)])
        g_vals.append(float(np.mean(dp_list)))
    if len(s_vals) < 2:
        return float("nan"), float("nan"), 0
    rho, p = stats.spearmanr(np.array(s_vals), np.array(g_vals))
    return float(rho), float(p), len(s_vals)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for strategy in STRATEGIES:
        print(f"\n{'=' * 60}", flush=True)
        print(f"Strategy: {strategy}", flush=True)

        faith, image_counts = load_faith(FAITH_CSV, strategy)
        n_valid = sum(1 for n in image_counts.values() if n >= MIN_IMAGES)
        print(f"Concepts with >= {MIN_IMAGES} images: {n_valid}", flush=True)

        # --- CARDS alpha sweep: pick best alpha by pooled rho ---
        by_alpha = load_alpha_scores(ALPHA_CSV)
        print(f"\nCARDS alpha sweep  (alphas: {sorted(by_alpha.keys())})", flush=True)
        alpha_results = []
        for alpha in sorted(by_alpha.keys()):
            m_rho, n_c          = class_rho(by_alpha[alpha], faith, image_counts)
            p_rho, p_p, n_pairs = pooled_rho(by_alpha[alpha], faith, image_counts)
            alpha_results.append((alpha, m_rho, n_c, p_rho, p_p, n_pairs))
            print(
                f"  alpha={alpha:.1f}  class_rho={m_rho:+.4f}  n={n_c}"
                f"  pooled_rho={p_rho:+.4f}  p={p_p:.4g}  n_pairs={n_pairs}",
                flush=True,
            )
        best_c = max(alpha_results, key=lambda x: x[1])
        best_p = max(alpha_results, key=lambda x: x[3])
        print(f"  Best by class_rho : alpha={best_c[0]}  rho={best_c[1]:+.4f}", flush=True)
        print(f"  Best by pooled_rho  : alpha={best_p[0]}  rho={best_p[3]:+.4f}", flush=True)

        # --- Comparison table: all methods ---
        best_alpha   = best_p[0]
        cards_scores = by_alpha[best_alpha]

        other_available = [(name, load_method_scores(p, col))
                           for name, p, col in OTHER_METHODS if p.exists()]
        other_missing   = [name for name, p, _ in OTHER_METHODS if not p.exists()]

        print(f"\n{'─' * 78}", flush=True)
        print(
            f"{'Method':<16}  {'class_rho':>12}  {'n_classes':>10}"
            f"  {'pooled_rho':>11}  {'pooled_p':>10}  {'n_pairs':>8}",
            flush=True,
        )
        print(f"{'─' * 78}", flush=True)

        for label, scores in [(f"CARDS (a={best_alpha})", cards_scores)] + other_available:
            m_rho, n_c          = class_rho(scores, faith, image_counts)
            p_rho, p_p, n_pairs = pooled_rho(scores, faith, image_counts)
            print(
                f"{label:<16}  {m_rho:>+12.4f}  {n_c:>10d}"
                f"  {p_rho:>+11.4f}  {p_p:>10.4g}  {n_pairs:>8d}",
                flush=True,
            )

        if other_missing:
            print(f"\nSkipped (CSV not found): {', '.join(other_missing)}", flush=True)


if __name__ == "__main__":
    main()
