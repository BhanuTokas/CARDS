"""Evaluate CARDS masking-hybrid scores against ground-truth faithfulness.

Loads:
  results/coco_stuff/cards_coco_stuff_masking_hybrid_scores.csv
    -- CARDS raw_score per (concept_name, class_name)
  results/coco_stuff/coco_stuff_faithfulness.csv
    -- per-(image, concept, class) delta_p and delta_logit from
       Grounding DINO + SAM + ResNet18 (compute_coco_stuff_faithfulness.py)

For each (concept, class) pair present in both:
  ground_truth = mean(delta_p) over all faithfulness records for that pair

Metrics (matching the CelebA masking-hybrid evaluation):
  Spearman rho  -- rank correlation between CARDS raw_score and mean(delta_p)
  Sign agreement -- fraction of pairs where sign(CARDS) == sign(ground truth)
  p-values for both via scipy

Results are printed per class and overall (all 20 classes pooled).

Usage:
  cd /path/to/CARDS
  uv run python scripts/coco_stuff/analysis/score_cards_vs_faithfulness_coco_stuff.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
from scipy import stats

from cards.data.coco_stuff import CO_OCCUR_BRODEN_CONCEPTS, COCO_STUFF_20_CLASSES

RESULTS_DIR    = Path("results/coco_stuff")
CARDS_CSV      = RESULTS_DIR / "cards_coco_stuff_masking_hybrid_scores.csv"
FAITH_CSV      = RESULTS_DIR / "coco_stuff_faithfulness.csv"
COUNTS_CSV     = RESULTS_DIR / "coco_stuff_faithfulness_detection_counts.csv"
MIN_SAMPLES    = 3   # minimum faithfulness records per (concept, class) to include


def load_cards_scores(path: Path) -> dict[tuple[str, str], float]:
    """concept_name, class_name -> raw_score."""
    scores: dict[tuple[str, str], float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            scores[(row["concept_name"], row["class_name"])] = float(row["raw_score"])
    return scores


def load_faithfulness(path: Path) -> dict[tuple[str, str], list[float]]:
    """concept_name, class_name -> list of delta_p values (one per image)."""
    delta_p: dict[tuple[str, str], list[float]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["concept_name"], row["class_name"])
            delta_p[key].append(float(row["delta_p"]))
    return delta_p


def load_detection_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            counts[row["concept_name"]] = int(row["n_detected"])
    return counts


def _sign_agreement_pval(n_agree: int, n_total: int) -> float:
    """Two-sided binomial p-value vs. random chance (p=0.5)."""
    result = stats.binomtest(n_agree, n_total, p=0.5, alternative="two-sided")
    return result.pvalue


def evaluate(
    cards: dict[tuple[str, str], float],
    faith: dict[tuple[str, str], list[float]],
    label: str,
) -> tuple[float, float, float, float, int]:
    """Compute Spearman rho and sign agreement for the supplied subset.

    Returns (rho, rho_pval, sign_agreement, sign_pval, n_pairs).
    """
    cards_vals, gt_vals = [], []
    for key, delta_p_list in faith.items():
        if len(delta_p_list) < MIN_SAMPLES:
            continue
        if key not in cards:
            continue
        cards_vals.append(cards[key])
        gt_vals.append(float(np.mean(delta_p_list)))

    n = len(cards_vals)
    if n < 2:
        print(f"  [{label}] too few pairs ({n}) -- skipping", flush=True)
        return float("nan"), float("nan"), float("nan"), float("nan"), n

    cards_arr = np.array(cards_vals)
    gt_arr    = np.array(gt_vals)

    rho, rho_p = stats.spearmanr(cards_arr, gt_arr)

    n_agree  = int(np.sum(np.sign(cards_arr) == np.sign(gt_arr)))
    sign_agr = n_agree / n
    sign_p   = _sign_agreement_pval(n_agree, n)

    print(
        f"  [{label}]  n={n}  "
        f"rho={rho:+.3f} (p={rho_p:.4g})  "
        f"sign={sign_agr:.1%} (p={sign_p:.4g})",
        flush=True,
    )
    return float(rho), float(rho_p), sign_agr, sign_p, n


def main() -> None:
    for path in (CARDS_CSV, FAITH_CSV):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file missing: {path}\n"
                "Run the corresponding build/run scripts first."
            )

    cards = load_cards_scores(CARDS_CSV)
    faith = load_faithfulness(FAITH_CSV)

    total_faith_pairs = sum(1 for v in faith.values() if len(v) >= MIN_SAMPLES)
    total_cards_pairs = len(cards)
    print(
        f"CARDS pairs: {total_cards_pairs}  "
        f"faithfulness pairs (>={MIN_SAMPLES} samples): {total_faith_pairs}",
        flush=True,
    )

    # --- Optional: detection counts summary -----------------------------
    if COUNTS_CSV.exists():
        counts = load_detection_counts(COUNTS_CSV)
        n_detected = sum(1 for v in counts.values() if v > 0)
        n_zero     = sum(1 for v in counts.values() if v == 0)
        print(f"Detection counts: {n_detected} concepts detected, {n_zero} zero-detection", flush=True)
        print("Zero-detection (skipped by GDINO):", flush=True)
        for c in sorted(c for c, v in counts.items() if v == 0):
            print(f"  {c}", flush=True)
        print(flush=True)

    # --- Per-class evaluation (all 143 concepts) ------------------------
    print("=== Per-class evaluation (all concepts) ===", flush=True)
    class_rhos, class_signs = [], []
    for cls_name in COCO_STUFF_20_CLASSES:
        faith_cls = {k: v for k, v in faith.items() if k[1] == cls_name}
        cards_cls = {k: v for k, v in cards.items() if k[1] == cls_name}
        rho, _, sign_agr, _, _ = evaluate(cards_cls, faith_cls, cls_name)
        if not np.isnan(rho):
            class_rhos.append(rho)
            class_signs.append(sign_agr)

    # --- Overall (all classes pooled, all concepts) ---------------------
    print("\n=== Overall (all 20 classes pooled, all concepts) ===", flush=True)
    evaluate(cards, faith, "ALL-concepts")

    if class_rhos:
        print(
            f"\nMean per-class rho (all concepts)  = {np.mean(class_rhos):.3f}  "
            f"mean sign = {np.mean(class_signs):.1%}",
            flush=True,
        )

    # --- Per-class evaluation (co-occur concepts only) ------------------
    print(
        f"\n=== Per-class evaluation (co-occur concepts only: "
        f"{sorted(CO_OCCUR_BRODEN_CONCEPTS)}) ===",
        flush=True,
    )
    cooccur_rhos, cooccur_signs = [], []
    for cls_name in COCO_STUFF_20_CLASSES:
        faith_co = {k: v for k, v in faith.items()
                    if k[1] == cls_name and k[0] in CO_OCCUR_BRODEN_CONCEPTS}
        cards_co = {k: v for k, v in cards.items()
                    if k[1] == cls_name and k[0] in CO_OCCUR_BRODEN_CONCEPTS}
        rho, _, sign_agr, _, _ = evaluate(cards_co, faith_co, f"{cls_name} [co-occur]")
        if not np.isnan(rho):
            cooccur_rhos.append(rho)
            cooccur_signs.append(sign_agr)

    # --- Overall (all classes pooled, co-occur concepts only) -----------
    print("\n=== Overall (all 20 classes pooled, co-occur concepts only) ===", flush=True)
    faith_co_all = {k: v for k, v in faith.items() if k[0] in CO_OCCUR_BRODEN_CONCEPTS}
    cards_co_all = {k: v for k, v in cards.items() if k[0] in CO_OCCUR_BRODEN_CONCEPTS}
    evaluate(cards_co_all, faith_co_all, "ALL-co-occur")

    if cooccur_rhos:
        print(
            f"\nMean per-class rho (co-occur concepts) = {np.mean(cooccur_rhos):.3f}  "
            f"mean sign = {np.mean(cooccur_signs):.1%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
