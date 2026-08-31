"""zero_fill variant of plot_predicted_vs_actual_celeba_full.py -- reads
run_celeba_full_faithfulness_zerofill.py's own ground truth instead of
the blur-based one, so the two masking strategies' results can sit side
by side rather than one overwriting the other. CARDS/TCAV/PCBM scores
are the SAME underlying CSVs/checkpoints (unaffected by masking
strategy) -- only the ground truth changes.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")  # needed to unpickle the saved PosthocLinearCBM checkpoint

from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS, TARGET_CLASSES
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    _aggregate_faithfulness_pairs,
    score_method_agreement,
    score_sign_agreement,
)

RESULTS_DIR = Path("results")
OUT_PATH = RESULTS_DIR / "predicted_vs_actual_celeba_full_zerofill.png"

CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}


def load_records_by_task() -> dict[str, list[FaithfulnessResult]]:
    by_task: dict[str, list[FaithfulnessResult]] = {t: [] for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "celeba_full_faithfulness_zerofill.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]].append(FaithfulnessResult(
                image=row["image"], concept_number=CONCEPT_TO_IDX[row["concept_name"]], category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return by_task


def load_cards_scores() -> dict[str, dict[tuple[int, int], float]]:
    by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "cards_celeba_full_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]][(CONCEPT_TO_IDX[row["concept_name"]], 1)] = float(row["raw_score"])
    return by_task


def load_tcav_scores() -> dict[str, dict[tuple[int, int], float]]:
    by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "tcav_celeba_full_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]][(CONCEPT_TO_IDX[row["concept_name"]], 1)] = float(row["mean_sign_count"])
    return by_task


def load_pcbm_scores() -> dict[str, dict[tuple[int, int], float]]:
    by_task: dict[str, dict[tuple[int, int], float]] = {}
    for task_name in TARGET_CLASSES:
        ckpt_path = (
            Path("trained_models_new/celeba_full/celeba_attractive_young")
            / f"pcbm_celeba_full__celeba_attractive_young__{task_name.lower()}__surrogate__seed_42__linear.ckpt"
        )
        if not ckpt_path.exists():
            by_task[task_name] = {}
            continue
        posthoc_layer = torch.load(ckpt_path, weights_only=False)
        # binary-task sklearn coef_ is a single (1, n_concepts) decision-
        # function row, not one row per class -- see
        # train_pcbm_surrogate_celeba_pilot.py's own writeup.
        weight = posthoc_layer.classifier.weight.detach().cpu().numpy()  # (1, n_concepts)
        names = posthoc_layer.names
        by_task[task_name] = {(CONCEPT_TO_IDX[name], 1): float(weight[0, i]) for i, name in enumerate(names)}
    return by_task


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

    ax.scatter(x, y, c=colors, s=20, alpha=0.7, edgecolors="none")
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
    records_by_task = load_records_by_task()
    cards_by_task = load_cards_scores()
    tcav_by_task = load_tcav_scores()
    pcbm_by_task = load_pcbm_scores()
    for task in TARGET_CLASSES:
        print(f"{task}: {len(records_by_task[task])} faithfulness records loaded.", flush=True)

    method_cols: list[tuple[str, dict[str, dict[tuple[int, int], float]], float]] = [
        ("CARDS\n(K=50, demean=True, aligned, SigLIP)", cards_by_task, 0.0),
        ("TCAV\n(sign_count, N_random=6)", tcav_by_task, 0.5),
        ("PCBM\n(region-crop bank, surrogate)", pcbm_by_task, 0.0),
    ]

    n_rows, n_cols = len(TARGET_CLASSES), len(method_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.8 * n_rows))

    for row, task in enumerate(TARGET_CLASSES):
        for col, (name, scores_by_task, threshold) in enumerate(method_cols):
            ax = axes[row, col]
            plot_panel(ax, f"{name}\ntarget: {task}", records_by_task[task], scores_by_task[task], threshold=threshold)

    n_total_records = sum(len(r) for r in records_by_task.values())
    fig.suptitle(
        f"Predicted vs. actual concept importance -- CelebA full 26-concept bank (zero_fill masking), "
        f"n_ground_truth_records={n_total_records}\n"
        "green = sign-agreeing pair, red = sign-disagreeing pair",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
