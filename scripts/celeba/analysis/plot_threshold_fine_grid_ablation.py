"""Line chart of the fine-grid alpha ablation's own class-pooled results
-- prompted directly ("Can you give me a plot for the alpha values?"),
the natural follow-up to `generate_latex_threshold_fine_grid_ablation.
py`'s own table (same source data: `cards_celeba_masking_hybrid_
threshold_official_train_fine_grid_ablation.csv`, previously_used
ground truth, Attractive+Male pooled via naive mean).

Single series (one line, no categorical palette needed) -- the chosen
alpha=1.0 point is marked distinctly (larger marker, callout) since
it's the setting actually used everywhere else in this track, and the
whole point of this chart is showing it sits atop a broad plateau
(0.75-1.25 all within ~0.014 of each other) rather than a narrow spike.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
IN_CSV = RESULTS_DIR / "cards_celeba_masking_hybrid_threshold_official_train_fine_grid_ablation.csv"
OUT_PNG = RESULTS_DIR / "cards_celeba_masking_hybrid_threshold_official_train_fine_grid_ablation.png"

GT_VERSION = "previously_used"
ALPHA_VALUES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
CHOSEN_ALPHA = 1.0

LINE_COLOR = "#2a78d6"  # dataviz skill categorical slot 1, blue
CHOSEN_COLOR = "#eb6834"  # dataviz skill categorical slot 2, orange
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_HAIRLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def main():
    by_key: dict[tuple[float, str], float] = {}
    with open(IN_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["gt_version"] != GT_VERSION:
                continue
            by_key[(round(float(row["alpha"]), 2), row["target_task"])] = float(row["spearman_rho"])

    pooled = []
    for alpha in ALPHA_VALUES:
        rho_a = by_key[(alpha, "Attractive")]
        rho_m = by_key[(alpha, "Male")]
        pooled.append((rho_a + rho_m) / 2)

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(ALPHA_VALUES, pooled, color=LINE_COLOR, linewidth=2.2, marker="o",
             markersize=6, markeredgewidth=0, zorder=3)

    chosen_idx = ALPHA_VALUES.index(CHOSEN_ALPHA)
    ax.plot(CHOSEN_ALPHA, pooled[chosen_idx], color=CHOSEN_COLOR, marker="o",
             markersize=11, markeredgewidth=2, markeredgecolor=SURFACE, zorder=4)
    ax.annotate(f"chosen: $\\alpha$={CHOSEN_ALPHA:g}\n$\\rho$={pooled[chosen_idx]:.3f}",
                xy=(CHOSEN_ALPHA, pooled[chosen_idx]), xytext=(CHOSEN_ALPHA + 0.18, pooled[chosen_idx] + 0.035),
                color=CHOSEN_COLOR, fontsize=10, fontweight="bold")

    ax.set_xlabel("z-score threshold (alpha)", color=INK_SECONDARY, fontsize=11)
    ax.set_ylabel("Pooled Spearman rho\n(Attractive + Male, naive mean)", color=INK_SECONDARY, fontsize=11)
    ax.set_xticks(ALPHA_VALUES)
    ax.tick_params(colors=INK_MUTED, labelsize=9.5)
    ax.set_ylim(0.40, 0.68)
    ax.grid(axis="y", color=GRID_HAIRLINE, linewidth=1, zorder=0)
    for spine_name, spine in ax.spines.items():
        spine.set_visible(spine_name == "bottom")
        if spine_name == "bottom":
            spine.set_color(GRID_HAIRLINE)

    ax.set_title("Fine-grid $\\alpha$ ablation: official-train CelebA masking hybrid (ResNet18)",
                 color=INK_PRIMARY, fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, facecolor=SURFACE)
    print(f"Saved {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
