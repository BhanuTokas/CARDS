"""Small-multiples comparison of all 4 attribution methods' mean |score|
across the 26-concept bank, per task, per shortcut-injection rate --
visualizes the table already computed ad hoc for the official-train
shortcut experiment (prompted directly, "Can you give me a graph for
this?"). One panel per method (5: masking hybrid, TCAV magnitude, PCBM
conventional, PCBM-SigLIP, PCBM-CLIP-RN50) rather than one shared axis --
their native scales span 4 orders of magnitude (TCAV ~0.02, PCBM-CLIP
~20-280), so a single y-axis would flatten the small ones to invisible.
Color encodes TASK (Attractive/Male) consistently across every panel,
using the dataviz skill's own validated categorical slots 1/2 (blue/
orange, CVD-safe pair).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
RATES = list(range(0, 101, 10))
TASKS = ["Attractive", "Male"]
TASK_COLORS = {"Attractive": "#2a78d6", "Male": "#eb6834"}  # dataviz skill categorical slots 1/2

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_HAIRLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

METHODS = [
    ("Masking hybrid", RESULTS_DIR / "cards_celeba_official_train_shortcut_experiment_raw_scores.csv", "hybrid_raw_score"),
    ("TCAV (magnitude)", RESULTS_DIR / "tcav_celeba_official_train_shortcut_experiment.csv", "mean_magnitude"),
    ("PCBM (conventional)", RESULTS_DIR / "pcbm_celeba_official_train_shortcut_experiment.csv", "weight"),
    ("PCBM (SigLIP)", RESULTS_DIR / "pcbm_clip_concepts_shortcut_official_train_experiment_siglip.csv", "weight"),
    ("PCBM (CLIP-RN50)", RESULTS_DIR / "pcbm_clip_concepts_shortcut_official_train_experiment_clip_rn50.csv", "weight"),
]


def load_mean_abs(path: Path, score_col: str) -> dict[str, dict[int, float]]:
    by_task_rate: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_task_rate[row["task"]][int(row["rate_pct"])].append(abs(float(row[score_col])))
    return {
        task: {rate: sum(vals) / len(vals) for rate, vals in rates.items()}
        for task, rates in by_task_rate.items()
    }


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), facecolor=SURFACE)
    axes_flat = axes.flatten()

    for ax, (method_name, path, score_col) in zip(axes_flat, METHODS):
        ax.set_facecolor(SURFACE)
        data = load_mean_abs(path, score_col)

        for task in TASKS:
            y = [data[task][r] for r in RATES]
            ax.plot(RATES, y, color=TASK_COLORS[task], linewidth=2.2, marker="o",
                     markersize=6, markeredgewidth=0, label=task, zorder=3)

        ax.set_title(method_name, color=INK_PRIMARY, fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(RATES[::2])
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.set_ylabel("mean |score|", color=INK_SECONDARY, fontsize=9.5)
        ax.yaxis.set_tick_params(labelcolor=INK_MUTED)
        ax.grid(axis="y", color=GRID_HAIRLINE, linewidth=1, zorder=0)
        for spine_name, spine in ax.spines.items():
            spine.set_visible(spine_name == "bottom")
            if spine_name == "bottom":
                spine.set_color(GRID_HAIRLINE)

    # hide the 6th (unused) panel
    axes_flat[5].axis("off")

    for ax in axes_flat[3:5]:
        ax.set_xlabel("injection rate (%)", color=INK_SECONDARY, fontsize=9.5)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2,
               frameon=False, fontsize=11, labelcolor=INK_PRIMARY)
    fig.suptitle("Official-train shortcut experiment: mean |score| across 26 concepts",
                 color=INK_PRIMARY, fontsize=13.5, y=1.08)

    fig.tight_layout(rect=[0, 0, 1, 1.0])
    out_path = RESULTS_DIR / "celeba_official_train_shortcut_methods_comparison.png"
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
