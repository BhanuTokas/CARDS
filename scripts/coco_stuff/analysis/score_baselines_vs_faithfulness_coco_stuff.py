"""Unified baseline comparison for COCO-Stuff.

Loads all method scores and evaluates each against the faithfulness CSV,
reporting Spearman rho + sign agreement in a single side-by-side table.

Methods compared:
  CARDS      -- best alpha from alpha_ablation_scores.csv
                (determined by alpha_ablation_summary.csv mean rho)
  TCAV       -- mean_sign_count from tcav_coco_stuff_scores.csv
  PCBM-resnet     -- weights from pcbm_resnet_concept_weights.csv
  PCBM-CLIP  -- weights from pcbm_clip_concept_weights.csv
  PCBM-SigLIP-- weights from pcbm_siglip_concept_weights.csv

Prerequisites (all results/coco_stuff/ CSVs must exist):
  02b  ablate_cards_coco_stuff_alpha.py
  03   compute_coco_stuff_faithfulness.py
  05   run_tcav_coco_stuff.py
  06   train_pcbm_resnet_coco_stuff.py
  07   train_pcbm_clip_zeroshot_coco_stuff.py
  08   train_pcbm_siglip_zeroshot_coco_stuff.py

Outputs:
  results/coco_stuff/baseline_comparison_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
from scipy import stats

from cards.data.coco_stuff import CO_OCCUR_BRODEN_CONCEPTS, COCO_STUFF_20_CLASSES

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
RESULTS_DIR      = Path("results/coco_stuff")
FAITH_CSV        = RESULTS_DIR / "coco_stuff_faithfulness.csv"
ABLATION_SCORES  = RESULTS_DIR / "alpha_ablation_scores.csv"
ABLATION_SUMMARY = RESULTS_DIR / "alpha_ablation_summary.csv"
TCAV_CSV         = RESULTS_DIR / "tcav_coco_stuff_scores.csv"
PCBM_RESNET_CSV  = RESULTS_DIR / "pcbm_resnet_concept_weights.csv"
PCBM_CLIP_CSV    = RESULTS_DIR / "pcbm_clip_concept_weights.csv"
PCBM_SIGLIP_CSV  = RESULTS_DIR / "pcbm_siglip_concept_weights.csv"
OUT_CSV          = RESULTS_DIR / "baseline_comparison_summary.csv"

MIN_SAMPLES      = 3
DELTA_P_NOISE    = 0.01   # exclude |delta_p| <= this from rho (noisy/negligible effect)
SEED             = 42
STRATEGIES       = ("black_mask", "black_bbox")


# ---------------------------------------------------------------------------
# Faithfulness loading + balancing
# ---------------------------------------------------------------------------

def _load_faithfulness_raw(
    path: Path, strategy: str
) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """concept -> class -> [(image_path, delta_p)]"""
    col  = f"delta_p_{strategy}"
    data: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data[row["concept_name"]][row["class_name"]].append(
                (row["image_path"], float(row[col]))
            )
    return data


def _balance(
    faith_raw: dict[str, dict[str, list[tuple[str, float]]]],
    min_count: int,
    rng: np.random.Generator,
) -> dict[tuple[str, str], list[float]]:
    """Subsample each concept to min_count images; return (concept, class) -> [delta_p]."""
    result: dict[tuple[str, str], list[float]] = defaultdict(list)
    for concept, class_dict in faith_raw.items():
        all_images: set[str] = set()
        for img_dp_list in class_dict.values():
            for img, _ in img_dp_list:
                all_images.add(img)
        if len(all_images) < min_count:
            continue
        sampled = set(rng.choice(sorted(all_images), size=min_count, replace=False))
        for cls_name, img_dp_list in class_dict.items():
            for img, dp in img_dp_list:
                if img in sampled:
                    result[(concept, cls_name)].append(dp)
    return dict(result)


def _load_generic_scores(path: Path, score_col: str) -> dict[tuple[str, str], float]:
    """Load any (concept_name, class_name, <score_col>) CSV."""
    scores: dict[tuple[str, str], float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            scores[(row["concept_name"], row["class_name"])] = float(row[score_col])
    return scores


def _load_cards_best_alpha(
    scores_path: Path, summary_path: Path, strategy: str
) -> tuple[dict[tuple[str, str], float], float]:
    """Pick best alpha for the given strategy by mean rho, then load those scores."""
    alpha_rhos: dict[float, list[float]] = defaultdict(list)
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("strategy", "black_mask") != strategy:
                continue
            rho = row["spearman_rho"]
            if rho and rho != "nan":
                alpha_rhos[float(row["alpha"])].append(float(rho))

    best_alpha: float | None = None
    best_rho = -float("inf")
    for alpha, rhos in alpha_rhos.items():
        mean_rho = float(np.mean(rhos))
        if mean_rho > best_rho:
            best_rho, best_alpha = mean_rho, alpha

    scores: dict[tuple[str, str], float] = {}
    with open(scores_path, newline="") as f:
        for row in csv.DictReader(f):
            if float(row["alpha"]) == best_alpha:
                scores[(row["concept_name"], row["class_name"])] = float(row["raw_score"])
    return scores, best_alpha  # type: ignore[return-value]


def _load_tcav_scores(path: Path) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            scores[(row["concept_name"], row["class_name"])] = float(row["mean_sign_count"])
    return scores


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_method(
    scores: dict[tuple[str, str], float],
    faith: dict[tuple[str, str], list[float]],
) -> dict:
    """Compute per-class Spearman rho, Pearson r, sign agreement (all + co-occur)."""
    class_rhos_all,    class_pearson_all, class_signs_all = [], [], []
    class_rhos_co,     class_pearson_co,  class_signs_co  = [], [], []

    for cls_name in COCO_STUFF_20_CLASSES:
        for rho_list, pearson_list, sign_list, co_only in [
            (class_rhos_all, class_pearson_all, class_signs_all, False),
            (class_rhos_co,  class_pearson_co,  class_signs_co,  True),
        ]:
            s_vals, g_vals = [], []
            for (c, cl), dp_list in faith.items():
                if cl != cls_name:
                    continue
                if co_only and c not in CO_OCCUR_BRODEN_CONCEPTS:
                    continue
                if (c, cl) not in scores:
                    continue
                if len(dp_list) < MIN_SAMPLES:
                    continue
                s_vals.append(scores[(c, cl)])
                g_vals.append(float(np.mean(dp_list)))
            if len(s_vals) < 2:
                continue
            s_arr = np.array(s_vals)
            g_arr = np.array(g_vals)
            rho, _     = stats.spearmanr(s_arr, g_arr)
            pearson, _ = stats.pearsonr(s_arr, g_arr)
            sign_agr   = float(np.mean(np.sign(s_arr) == np.sign(g_arr)))
            rho_list.append(float(rho))
            pearson_list.append(float(pearson))
            sign_list.append(sign_agr)

    return {
        "mean_rho_all":     float(np.mean(class_rhos_all))    if class_rhos_all    else float("nan"),
        "mean_pearson_all": float(np.mean(class_pearson_all)) if class_pearson_all else float("nan"),
        "mean_sign_all":    float(np.mean(class_signs_all))   if class_signs_all   else float("nan"),
        "mean_rho_co":      float(np.mean(class_rhos_co))     if class_rhos_co     else float("nan"),
        "mean_pearson_co":  float(np.mean(class_pearson_co))  if class_pearson_co  else float("nan"),
        "mean_sign_co":     float(np.mean(class_signs_co))    if class_signs_co    else float("nan"),
        "n_classes_all":    len(class_rhos_all),
        "n_classes_co":     len(class_rhos_co),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_count", type=int, required=True,
                        help="Images per concept to subsample to before computing rho.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not FAITH_CSV.exists():
        raise FileNotFoundError(f"Faithfulness CSV not found: {FAITH_CSV}")

    # --- Load strategy-agnostic method scores ---
    non_cards_methods: list[tuple[str, dict[tuple[str, str], float]]] = []

    for label, csv_path, score_col in [
        ("TCAV",        TCAV_CSV,        None),
        ("PCBM-resnet", PCBM_RESNET_CSV, "weight"),
        ("PCBM-CLIP",   PCBM_CLIP_CSV,   "weight"),
        ("PCBM-SigLIP", PCBM_SIGLIP_CSV, "weight"),
    ]:
        if not csv_path.exists():
            print(f"{label}: CSV not found ({csv_path}) -- skipping", flush=True)
            continue
        scores = _load_tcav_scores(csv_path) if label == "TCAV" else _load_generic_scores(csv_path, score_col)  # type: ignore[arg-type]
        non_cards_methods.append((label, scores))
        print(f"{label}: {len(scores)} scores", flush=True)

    # --- Evaluate each method × each masking strategy ---
    out_rows = []
    for strategy in STRATEGIES:
        # Load CARDS best alpha for this specific strategy
        methods: list[tuple[str, dict[tuple[str, str], float]]] = []
        if ABLATION_SCORES.exists() and ABLATION_SUMMARY.exists():
            cards_scores, best_alpha = _load_cards_best_alpha(ABLATION_SCORES, ABLATION_SUMMARY, strategy)
            if best_alpha is not None:
                methods.append((f"CARDS (alpha={best_alpha})", cards_scores))
                print(f"CARDS [{strategy}]: {len(cards_scores)} scores  (best alpha={best_alpha})", flush=True)
        else:
            print("CARDS ablation CSVs not found -- skipping CARDS", flush=True)
        methods.extend(non_cards_methods)

        faith_raw = _load_faithfulness_raw(FAITH_CSV, strategy=strategy)
        faith     = _balance(faith_raw, args.min_count, np.random.default_rng(SEED))
        n_concepts = len({c for c, _ in faith})
        print(f"\n{'=' * 80}", flush=True)
        print(f"Strategy: {strategy}  |  min_count={args.min_count}  |  concepts={n_concepts}  |  pairs={len(faith)}", flush=True)
        print(
            f"{'Method':<22}  {'rho_all':>8}  {'r_all':>8}  {'sign_all':>9}  "
            f"{'rho_co':>8}  {'r_co':>8}  {'sign_co':>9}  {'n_cls_all':>9}  {'n_cls_co':>9}",
            flush=True,
        )
        print("-" * 100, flush=True)

        for method_name, scores in methods:
            res = _evaluate_method(scores, faith)
            print(
                f"{method_name:<22}  "
                f"{res['mean_rho_all']:>+8.3f}  {res['mean_pearson_all']:>+8.3f}  {res['mean_sign_all']:>9.1%}  "
                f"{res['mean_rho_co']:>+8.3f}  {res['mean_pearson_co']:>+8.3f}  {res['mean_sign_co']:>9.1%}  "
                f"{res['n_classes_all']:>9d}  {res['n_classes_co']:>9d}",
                flush=True,
            )
            out_rows.append({"strategy": strategy, "method": method_name, **res})

    # --- Save summary CSV ---
    if out_rows:
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\nSaved → {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
