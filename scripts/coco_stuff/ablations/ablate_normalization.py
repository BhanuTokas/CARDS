"""Normalization ablation for COCO-Stuff CARDS.

Compares two aggregation methods for CARDS delta_logit scores:
  raw        -- mean(delta_logit) per (concept, class)
  normalized -- mean(delta_logit) / std(delta_logit)  [Cohen's d-style]

Both computed from alpha_ablation_per_image_scores.csv (per-image CARDS delta
logits from pseudo-mask scoring at alpha=1.0). Ground truth is mean(delta_p)
from coco_stuff_faithfulness.csv (ground-truth SAM masks). No GPU required.

Outputs (results/coco_stuff/):
  normalization_ablation_summary.csv -- (method, strategy, class_name, n_pairs,
                                         spearman_rho, spearman_p, sign_agreement,
                                         sign_p, pooled_rho, pooled_p, pooled_n)

Usage:
    cd /scratch/rnair21/CARDS
    /scratch/rnair21/envs/cards_env/bin/python \\
        scripts/coco_stuff/ablations/ablate_normalization.py
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR    = Path("results/coco_stuff")
FAITH_CSV      = RESULTS_DIR / "coco_stuff_faithfulness.csv"
PER_IMAGE_CSV  = RESULTS_DIR / "alpha_ablation_per_image_scores.csv"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
ALPHA          = 1.5   # which alpha condition to use from per-image CSV
MIN_SAMPLES    = 3
MIN_FAITH_IMGS = 100
STRATEGIES     = ("black_mask", "black_bbox")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_cards_per_image(
    path: Path, alpha: float
) -> dict[tuple[str, str], list[float]]:
    """(concept, class) -> [delta_logit from CARDS pseudo-mask scoring] at given alpha."""
    data: dict[tuple[str, str], list[float]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if abs(float(row["alpha"]) - alpha) < 1e-6:
                data[(row["concept_name"], row["class_name"])].append(
                    float(row["delta_logit"])
                )
    return dict(data)


def load_faithfulness(
    path: Path, strategy: str
) -> tuple[dict[tuple[str, str], list[float]], dict[str, int]]:
    col = f"delta_p_{strategy}"
    concept_imgs: dict[str, set[str]] = defaultdict(set)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            concept_imgs[row["concept_name"]].add(row["image_path"])
    valid_imgs: dict[str, set[str]] = {
        c: set(sorted(imgs)[:MIN_FAITH_IMGS])
        for c, imgs in concept_imgs.items()
        if len(imgs) >= MIN_FAITH_IMGS
    }
    faith: dict[tuple[str, str], list[float]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            c, cl, img = row["concept_name"], row["class_name"], row["image_path"]
            if c in valid_imgs and img in valid_imgs[c]:
                faith[(c, cl)].append(float(row[col]))
    image_counts = {c: MIN_FAITH_IMGS for c in valid_imgs}
    return dict(faith), image_counts


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(
    per_image: dict[tuple[str, str], list[float]],
    method: str,
) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for key, vals in per_image.items():
        if len(vals) < MIN_SAMPLES:
            continue
        arr = np.array(vals)
        if method == "raw":
            scores[key] = float(arr.mean())
        else:  # normalized (Cohen's d)
            std = float(arr.std(ddof=1))
            scores[key] = float(arr.mean() / std) if std > 1e-8 else 0.0
    return scores


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _sign_pval(n_agree: int, n: int) -> float:
    return stats.binomtest(n_agree, n, p=0.5, alternative="two-sided").pvalue


def evaluate(
    scores:       dict[tuple[str, str], float],
    faith:        dict[tuple[str, str], list[float]],
    image_counts: dict[str, int],
) -> tuple[list[dict], float, float, float, float, int]:
    valid = {c for c, n in image_counts.items() if n >= MIN_FAITH_IMGS}
    rows: list[dict] = []
    class_rhos, class_signs = [], []
    pooled_s, pooled_g = [], []

    for cls_name in COCO_STUFF_20_CLASSES:
        s_vals, g_vals = [], []
        for key, dp_list in faith.items():
            if key[1] != cls_name or key[0] not in valid or len(dp_list) < MIN_SAMPLES:
                continue
            if key not in scores:
                continue
            s_vals.append(scores[key])
            g_vals.append(float(np.mean(dp_list)))
        pooled_s.extend(s_vals)
        pooled_g.extend(g_vals)

        n = len(s_vals)
        if n < 2:
            rows.append(dict(class_name=cls_name, n_pairs=n,
                             spearman_rho=float("nan"), spearman_p=float("nan"),
                             sign_agreement=float("nan"), sign_p=float("nan")))
            continue

        rho, rho_p = stats.spearmanr(s_vals, g_vals)
        n_agree  = int(np.sum(np.sign(np.array(s_vals)) == np.sign(np.array(g_vals))))
        sign_agr = n_agree / n
        rows.append(dict(class_name=cls_name, n_pairs=n,
                         spearman_rho=float(rho), spearman_p=float(rho_p),
                         sign_agreement=sign_agr, sign_p=_sign_pval(n_agree, n)))
        class_rhos.append(float(rho))
        class_signs.append(sign_agr)

    mean_rho  = float(np.mean(class_rhos))  if class_rhos  else float("nan")
    mean_sign = float(np.mean(class_signs)) if class_signs else float("nan")
    if len(pooled_s) >= 2:
        pr, pp = stats.spearmanr(pooled_s, pooled_g)
    else:
        pr, pp = float("nan"), float("nan")
    return rows, mean_rho, mean_sign, float(pr), float(pp), len(pooled_s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cards_per_image = load_cards_per_image(PER_IMAGE_CSV, alpha=ALPHA)
    print(f"CARDS per-image pairs loaded: {len(cards_per_image)}  (alpha={ALPHA})", flush=True)

    summary_rows: list[dict] = []

    for strategy in STRATEGIES:
        faith, image_counts = load_faithfulness(FAITH_CSV, strategy)
        print(f"\n{'=' * 60}  strategy={strategy}", flush=True)

        for method in ("raw", "normalized"):
            scores = aggregate(cards_per_image, method)
            rows, mean_rho, mean_sign, pr, pp, pn = evaluate(scores, faith, image_counts)
            print(
                f"  method={method:<12s}  mean_rho={mean_rho:+.4f}  "
                f"mean_sign={mean_sign:.4f}  "
                f"pooled_rho={pr:+.4f}  pooled_p={pp:.4g}  n_pairs={pn}",
                flush=True,
            )
            for r in rows:
                if not np.isnan(r["spearman_rho"]):
                    print(
                        f"    [{r['class_name']:<15s}] n={r['n_pairs']:>4d}  "
                        f"rho={r['spearman_rho']:+.3f}  sign={r['sign_agreement']:.1%}",
                        flush=True,
                    )
                summary_rows.append({
                    "method": method, "strategy": strategy, **r,
                    "pooled_rho": pr, "pooled_p": pp, "pooled_n": pn,
                })

    summary_csv = RESULTS_DIR / "normalization_ablation_summary.csv"
    fields = ["method", "strategy", "class_name", "n_pairs",
              "spearman_rho", "spearman_p", "sign_agreement", "sign_p",
              "pooled_rho", "pooled_p", "pooled_n"]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved -> {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
