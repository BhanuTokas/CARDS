"""Predicted (method score) vs actual (masking-based faithfulness delta_p)
scatter plots for every method scored in this investigation's Part 2
(87-attribute) comparison -- prompted directly ("can you visualize the
predicted vs. actual attribution in plots for different techniques?").

One panel per method: x = the method's own (concept, class) importance
score, y = mean delta_p (the masking ground truth) for that same pair,
aggregated via the SAME `_aggregate_faithfulness_pairs` helper
`score_method_agreement`/`score_sign_agreement` use internally, so what's
plotted is exactly what's being correlated -- no separate/inconsistent
aggregation logic. Quadrant lines at x=0 (or x=0.5 for TCAV's own
sign_count, its natural chance line) and y=0 make sign agreement visually
legible: points in the upper-right or lower-left quadrants (relative to
those lines) are sign-agreeing.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    _aggregate_faithfulness_pairs,
    score_method_agreement,
    score_sign_agreement,
)

RESULTS_DIR = Path("results")
OUT_PATH = RESULTS_DIR / "predicted_vs_actual_cub.png"


def load_records() -> list[FaithfulnessResult]:
    records = []
    with open(RESULTS_DIR / "cub_attribute_faithfulness.csv", newline="") as f:
        for row in csv.DictReader(f):
            records.append(FaithfulnessResult(
                image=row["image"], concept_number=int(row["concept_number"]), category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return records


def load_csv_scores(fname: str, col: str) -> dict[tuple[int, int], float]:
    scores = {}
    with open(RESULTS_DIR / fname, newline="") as f:
        for row in csv.DictReader(f):
            scores[(int(row["attribute_index"]), int(row["native_class_idx"]))] = float(row[col])
    return scores


def load_pcbm_official_resnet18cub_scores() -> dict[tuple[int, int], float]:
    ckpt = torch.load(
        "../post_hoc_cbm/trained_models_new/cub/resnet18_cub/"
        "pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam_0.0002__alpha_0.99__seed_42__linear.ckpt",
        weights_only=False,
    )
    weight = ckpt.classifier.weight.detach().cpu().numpy()
    return {(a, c): float(weight[c, a]) for a in range(weight.shape[1]) for c in range(weight.shape[0])}


def plot_panel(ax, name: str, records: list[FaithfulnessResult], scores: dict[tuple[int, int], float],
               threshold: float = 0.0) -> None:
    if not scores:
        ax.set_title(f"{name}\n(not available)", fontsize=10, color="#999")
        ax.axis("off")
        return

    aggregated = _aggregate_faithfulness_pairs(records, scores, min_samples_per_pair=3)
    rho_r = score_method_agreement(records, scores, min_samples_per_pair=3)
    sign_r = score_sign_agreement(records, scores, min_samples_per_pair=3, method_threshold=threshold)

    pairs = list(aggregated.keys())
    y = [aggregated[p] for p in pairs]  # actual: mean delta_p
    x = [scores[p] for p in pairs]  # predicted: method's own score

    agree_color, disagree_color = "#2a7f62", "#c1443c"
    colors = []
    for xi, yi in zip(x, y):
        agree = (xi > threshold) == (yi > 0)
        colors.append(agree_color if agree else disagree_color)

    ax.scatter(x, y, c=colors, s=16, alpha=0.65, edgecolors="none")
    ax.axhline(0, color="#888", linewidth=0.8, zorder=0)
    ax.axvline(threshold, color="#888", linewidth=0.8, zorder=0)
    ax.set_xlabel("predicted (method score)", fontsize=8)
    ax.set_ylabel("actual (mean $\\Delta p$, masking)", fontsize=8)
    ax.tick_params(labelsize=7)

    if rho_r is not None:
        subtitle = (f"n={rho_r.n_pairs}  rho={rho_r.spearman_rho:+.3f} (p={rho_r.spearman_p:.2g})\n"
                    f"sign={sign_r.agreement_frac:.1%} ({sign_r.n_agree}/{sign_r.n_pairs}, p={sign_r.binom_p:.2g})")
    else:
        subtitle = "too few pairs"
    ax.set_title(f"{name}\n{subtitle}", fontsize=9)


def main():
    records = load_records()
    print(f"{len(records)} faithfulness records loaded.", flush=True)

    methods: list[tuple[str, dict[tuple[int, int], float], float]] = [
        ("CARDS\n(K=50, demean=True, aligned, SigLIP)", load_csv_scores("cards_cub_attribute_scores.csv", "raw_score"), 0.0),
        ("PCBM (official 112-bank,\nresnet18_cub backbone, ground-truth)", load_pcbm_official_resnet18cub_scores(), 0.0),
        ("PCBM (official 112-bank,\nSigLIP backbone, surrogate)",
         load_csv_scores("pcbm_official_siglip_cub_scores.csv", "weight")
         if (RESULTS_DIR / "pcbm_official_siglip_cub_scores.csv").exists() else {}, 0.0),
        ("PCBM (official 112-bank,\nSigLIP backbone, ground-truth)",
         load_csv_scores("pcbm_official_siglip_groundtruth_cub_scores.csv", "weight")
         if (RESULTS_DIR / "pcbm_official_siglip_groundtruth_cub_scores.csv").exists() else {}, 0.0),
        ("PCBM (CLIP-RN50-concepts,\nno image dataset)", load_csv_scores("pcbm_clip_concepts_cub_scores.csv", "weight"), 0.0),
        ("PCBM (SigLIP-concepts,\nno image dataset)", load_csv_scores("pcbm_siglip_concepts_cub_scores.csv", "weight"), 0.0),
        ("TCAV (targeted, full coverage)",
         load_csv_scores("tcav_cub_targeted_v49.csv", "mean_sign_count")
         if (RESULTS_DIR / "tcav_cub_targeted_v49.csv").exists() else {}, 0.5),
    ]

    n_cols = 4
    n_rows = -(-len(methods) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows))
    axes = axes.flatten()

    for ax, (name, scores, threshold) in zip(axes, methods):
        plot_panel(ax, name, records, scores, threshold=threshold)
    for ax in axes[len(methods):]:
        ax.axis("off")

    fig.suptitle(
        f"Predicted vs. actual concept importance -- CUB Part 2 (87-attribute bank), n_ground_truth_records={len(records)}\n"
        "green = sign-agreeing pair, red = sign-disagreeing pair",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
