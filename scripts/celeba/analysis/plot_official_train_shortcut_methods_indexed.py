"""Single-axis comparison of all 5 attribution methods PLUS clean val
accuracy, task-averaged (Attractive+Male mean) and INDEXED to each
series' own rate=0% value (=100), so all 6 -- despite spanning 4 orders
of native magnitude (TCAV ~0.02, PCBM-CLIP ~20-280, accuracy ~0-1) --
share one meaningful y-axis: "% of that series' own rate=0% baseline
remaining at this injection rate." Prompted directly ("Can we average
across classes and report all the methods in a single graph with clean
test accuracy as the baseline also plotted there").

Indexing-to-a-common-base (not raw values on one axis) is the dataviz
skill's own prescribed fix for exactly this situation ("Two measures of
different scale -> two charts, small multiples, or indexed to a common
base") -- picking each series' own starting point as the base, rather
than min-max or z-score, keeps the answer directly interpretable ("how
much of the original signal is left") and keeps a real spike (PCBM
conventional shooting past 100% instead of declining) visually obvious
rather than compressed away.

Clean accuracy is styled distinctly (dashed, dark ink, thicker) rather
than as a 6th same-style categorical color -- it's the actual outcome
being explained, not a peer attribution method, and that asymmetry is
worth keeping visible rather than encoding it the same way as the 5
methods.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
RATES = list(range(0, 101, 10))
TASKS = ["Attractive", "Male"]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_HAIRLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

# dataviz skill's validated categorical slots 1-5 (blue/orange/aqua/yellow/magenta)
METHOD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

METHODS = [
    ("Masking hybrid", RESULTS_DIR / "cards_celeba_official_train_shortcut_experiment_raw_scores.csv", "hybrid_raw_score"),
    ("TCAV (magnitude)", RESULTS_DIR / "tcav_celeba_official_train_shortcut_experiment.csv", "mean_magnitude"),
    ("PCBM (conventional)", RESULTS_DIR / "pcbm_celeba_official_train_shortcut_experiment.csv", "weight"),
    ("PCBM (SigLIP)", RESULTS_DIR / "pcbm_clip_concepts_shortcut_official_train_experiment_siglip.csv", "weight"),
    ("PCBM (CLIP-RN50)", RESULTS_DIR / "pcbm_clip_concepts_shortcut_official_train_experiment_clip_rn50.csv", "weight"),
]
ACCURACY_CSV = RESULTS_DIR / "celeba_official_train_shortcut_accuracy_gap_summary.csv"


def task_averaged_mean_abs(path: Path, score_col: str) -> dict[int, float]:
    """rate -> mean(|score_col|) averaged over BOTH tasks (each task's own
    26-concept mean computed first, then the 2 task-means averaged --
    matches this track's established Attractive/Male combination
    convention of averaging TASK-LEVEL summaries, not pooling raw rows)."""
    by_task_rate: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_task_rate[row["task"]][int(row["rate_pct"])].append(abs(float(row[score_col])))
    task_means = {
        task: {rate: sum(vals) / len(vals) for rate, vals in rates.items()}
        for task, rates in by_task_rate.items()
    }
    return {rate: sum(task_means[t][rate] for t in TASKS) / len(TASKS) for rate in RATES}


def task_averaged_clean_acc() -> dict[int, float]:
    by_task_rate: dict[str, dict[int, float]] = defaultdict(dict)
    with open(ACCURACY_CSV, newline="") as f:
        for row in csv.DictReader(f):
            by_task_rate[row["task"]][int(row["rate_pct"])] = float(row["clean_val_acc"])
    return {rate: sum(by_task_rate[t][rate] for t in TASKS) / len(TASKS) for rate in RATES}


def index_to_baseline(series: dict[int, float]) -> list[float]:
    base = series[0]
    return [100.0 * series[r] / base for r in RATES]


def main():
    fig, ax = plt.subplots(figsize=(9.5, 6.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for (method_name, path, score_col), color in zip(METHODS, METHOD_COLORS):
        series = task_averaged_mean_abs(path, score_col)
        y = index_to_baseline(series)
        ax.plot(RATES, y, color=color, linewidth=2.2, marker="o", markersize=6,
                 markeredgewidth=0, label=method_name, zorder=3)

    acc_series = task_averaged_clean_acc()
    acc_y = index_to_baseline(acc_series)
    ax.plot(RATES, acc_y, color=INK_PRIMARY, linewidth=2.6, linestyle="--",
             marker="s", markersize=6, markeredgewidth=0, label="Clean val accuracy", zorder=4)

    ax.axhline(100, color=GRID_HAIRLINE, linewidth=1, zorder=0)
    ax.set_xticks(RATES)
    ax.set_xlabel("injection rate (%)", color=INK_SECONDARY, fontsize=11)
    ax.set_ylabel("% of rate=0% baseline (task-averaged)", color=INK_SECONDARY, fontsize=11)
    ax.tick_params(colors=INK_MUTED, labelsize=9.5)
    ax.grid(axis="y", color=GRID_HAIRLINE, linewidth=1, zorder=0)
    for spine_name, spine in ax.spines.items():
        spine.set_visible(spine_name == "bottom")
        if spine_name == "bottom":
            spine.set_color(GRID_HAIRLINE)

    ax.set_title("Official-train shortcut experiment: task-averaged, indexed to rate=0%",
                 color=INK_PRIMARY, fontsize=13.5, pad=14)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=INK_PRIMARY)
    legend.get_frame().set_alpha(0)

    fig.tight_layout()
    out_path = RESULTS_DIR / "celeba_official_train_shortcut_methods_indexed.png"
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
