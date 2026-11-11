"""Plot the COCO-Stuff shortcut experiment comparison.

Reads all method score CSVs + model accuracy CSV and produces a line plot:
  x-axis: shortcut injection rate (0-100%)
  left y-axis: mean |score| per method (normalised to [0,1] for comparability)
  right y-axis: accuracy gap (mean_shortcut_acc - mean_clean_acc)

Also saves a raw (un-normalised) version.

Inputs (results/coco_stuff/shortcut/):
  shortcut_model_accuracy.csv
  cards_shortcut_scores.csv
  tcav_shortcut_scores.csv
  pcbm_resnet_shortcut_weights.csv
  pcbm_clip_shortcut_weights.csv
  pcbm_siglip_shortcut_weights.csv

Output:
  results/coco_stuff/shortcut/shortcut_experiment_comparison.png
  results/coco_stuff/shortcut/shortcut_experiment_comparison_raw.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR  = Path("results/coco_stuff/shortcut")
RATES_PCT = list(range(0, 101, 10))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_accuracy(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            r = int(row["rate_pct"])
            rows[r] = {
                "mean_clean_acc":    float(row["mean_clean_acc"]),
                "mean_shortcut_acc": float(row["mean_shortcut_acc"]),
                "mean_gap":          float(row["mean_gap"]),
            }
    return rows


def _mean_abs_scores(path: Path, score_col: str) -> dict[int, float]:
    """Compute mean |score| per rate_pct from a CSV with (rate_pct, ..., score_col)."""
    acc: dict[int, list[float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            r = int(row["rate_pct"])
            acc.setdefault(r, []).append(abs(float(row[score_col])))
    return {r: float(np.mean(vals)) for r, vals in acc.items()}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(
    methods: list[tuple[str, dict[int, float]]],
    gap_by_rate: dict[int, float],
    clean_by_rate: dict[int, float],
    shortcut_by_rate: dict[int, float],
    normalise: bool,
    out_path: Path,
) -> None:
    rates = RATES_PCT
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for (label, scores_by_rate), color in zip(methods, colors):
        vals = [scores_by_rate.get(r, float("nan")) for r in rates]
        if normalise:
            max_v = max((v for v in vals if not np.isnan(v)), default=1.0)
            vals  = [v / max_v if not np.isnan(v) else float("nan") for v in vals]
        ax1.plot(rates, vals, marker="o", label=label, color=color, linewidth=2)

    # Accuracy curves on right axis
    gap_vals      = [gap_by_rate.get(r, float("nan")) for r in rates]
    clean_vals    = [clean_by_rate.get(r, float("nan")) for r in rates]
    shortcut_vals = [shortcut_by_rate.get(r, float("nan")) for r in rates]

    ax2.plot(rates, gap_vals,      marker="s", linestyle="--", color="black",
             label="Acc gap (shortcut - clean)", linewidth=2)
    ax2.plot(rates, clean_vals,    marker="^", linestyle=":",  color="gray",
             label="Clean test acc", linewidth=1.5, alpha=0.7)
    ax2.plot(rates, shortcut_vals, marker="v", linestyle=":",  color="dimgray",
             label="Shortcut test acc", linewidth=1.5, alpha=0.7)

    ax1.set_xlabel("Shortcut injection rate (%)", fontsize=12)
    y_label = "Mean |score| (normalised)" if normalise else "Mean |score|"
    ax1.set_ylabel(y_label, fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax1.set_xticks(rates)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    suffix = "(normalised)" if normalise else "(raw)"
    plt.title(f"COCO-Stuff Shortcut Experiment — Attribution Score Decline {suffix}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    acc_csv = OUT_DIR / "shortcut_model_accuracy.csv"
    if not acc_csv.exists():
        raise FileNotFoundError(f"Missing: {acc_csv}")
    acc = _load_accuracy(acc_csv)
    gap_by_rate      = {r: v["mean_gap"]          for r, v in acc.items()}
    clean_by_rate    = {r: v["mean_clean_acc"]    for r, v in acc.items()}
    shortcut_by_rate = {r: v["mean_shortcut_acc"] for r, v in acc.items()}

    method_files = [
        ("CARDS",        OUT_DIR / "cards_shortcut_scores.csv",        "raw_score"),
        ("TCAV",         OUT_DIR / "tcav_shortcut_scores.csv",         "mean_sign_count"),
        ("PCBM-resnet",  OUT_DIR / "pcbm_resnet_shortcut_weights.csv", "weight"),
        ("PCBM-CLIP",    OUT_DIR / "pcbm_clip_shortcut_weights.csv",   "weight"),
        ("PCBM-SigLIP",  OUT_DIR / "pcbm_siglip_shortcut_weights.csv", "weight"),
    ]

    methods: list[tuple[str, dict[int, float]]] = []
    for label, path, col in method_files:
        if not path.exists():
            print(f"{label}: CSV not found ({path}) -- skipping", flush=True)
            continue
        scores = _mean_abs_scores(path, col)
        methods.append((label, scores))
        print(f"{label}: loaded scores for {len(scores)} rates", flush=True)

    if not methods:
        print("No method CSVs found. Run shortcut experiment scripts first.", flush=True)
        return

    # Print table
    print(f"\n{'Rate':>6}  {'Gap':>8}  {'Clean':>8}  {'Shortcut':>9}", end="")
    for label, _ in methods:
        print(f"  {label:>14}", end="")
    print()
    for r in RATES_PCT:
        print(
            f"{r:>5}%  {gap_by_rate.get(r, float('nan')):>+8.4f}"
            f"  {clean_by_rate.get(r, float('nan')):>8.4f}"
            f"  {shortcut_by_rate.get(r, float('nan')):>9.4f}",
            end="",
        )
        for _, scores in methods:
            print(f"  {scores.get(r, float('nan')):>14.4f}", end="")
        print()

    _plot(methods, gap_by_rate, clean_by_rate, shortcut_by_rate,
          normalise=True,  out_path=OUT_DIR / "shortcut_experiment_comparison.png")
    _plot(methods, gap_by_rate, clean_by_rate, shortcut_by_rate,
          normalise=False, out_path=OUT_DIR / "shortcut_experiment_comparison_raw.png")


if __name__ == "__main__":
    main()
