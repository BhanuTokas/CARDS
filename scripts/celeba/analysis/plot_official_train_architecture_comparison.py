"""Visualizes `compare_official_train_across_architectures.py`'s own
combined table -- prompted directly as the natural follow-up ("Can we
have a graph for this?" -> "Yes please") to the 3-architecture
comparison (ResNet18 / ViT-B/16 / ConvNeXt-Tiny) on the official-train
CelebA main pipeline.

2x2 small multiples: rows = task (Attractive, Male), columns =
gt_version (non_overlapping, previously_used) -- both dimensions the
comparison script already reports per-cell. Within each panel, x-axis =
architecture (ordinal: resnet18 -> vit -> convnext, roughly ResNet-CNN
-> Transformer -> modern-CNN), y-axis = Spearman rho, one line per
method. Color encodes METHOD (not task, unlike the shortcut-experiment
chart) using the dataviz skill's own validated categorical slots 1-5
in order (CARDS, TCAV, PCBM-conventional, PCBM-SigLIP, PCBM-CLIP-RN50)
-- ResNet18 never got the PCBM-CLIP-concepts backfill (an explicit
earlier scope decision, "ViT + ConvNeXt only"), so those two methods'
lines simply start at the vit point, not a plotting bug.

A horizontal zero-line marks the rho=0 boundary (below it, a method is
anti-correlated with ground-truth faithfulness) since several
non-CARDS cells flip negative on ViT/ConvNeXt (see the comparison
script's own printed table).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
COMPARISON_CSV = RESULTS_DIR / "celeba_official_train_architecture_comparison.csv"
OUT_PNG = RESULTS_DIR / "celeba_official_train_architecture_comparison.png"

ARCHITECTURES = ["resnet18", "vit", "convnext"]
ARCH_LABELS = {"resnet18": "ResNet18", "vit": "ViT-B/16", "convnext": "ConvNeXt-Tiny"}
TASKS = ["Attractive", "Male"]
GT_VERSIONS = ["non_overlapping", "previously_used"]
GT_LABELS = {"non_overlapping": "non-overlapping ground truth", "previously_used": "previously-used ground truth"}

METHODS = ["CARDS (masking hybrid)", "TCAV", "PCBM (conventional)", "PCBM (CLIP-SigLIP)", "PCBM (CLIP-RN50)"]
METHOD_COLORS = {
    "CARDS (masking hybrid)": "#2a78d6",  # slot 1, blue
    "TCAV": "#eb6834",                     # slot 2, orange
    "PCBM (conventional)": "#1baf7a",      # slot 3, aqua
    "PCBM (CLIP-SigLIP)": "#eda100",       # slot 4, yellow
    "PCBM (CLIP-RN50)": "#e87ba4",         # slot 5, magenta
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_HAIRLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_table() -> dict[tuple[str, str, str, str], float]:
    rho_by_key: dict[tuple[str, str, str, str], float] = {}
    with open(COMPARISON_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["architecture"], row["task"], row["gt_version"], row["method"])
            rho_by_key[key] = float(row["spearman_rho"])
    return rho_by_key


def main():
    rho_by_key = load_table()

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), facecolor=SURFACE)

    for row_i, task in enumerate(TASKS):
        for col_i, gt_version in enumerate(GT_VERSIONS):
            ax = axes[row_i, col_i]
            ax.set_facecolor(SURFACE)

            for method in METHODS:
                xs, ys = [], []
                for arch_i, architecture in enumerate(ARCHITECTURES):
                    rho = rho_by_key.get((architecture, task, gt_version, method))
                    if rho is not None:
                        xs.append(arch_i)
                        ys.append(rho)
                if len(xs) < 2:
                    # ResNet18-only points (none here) would be invisible as a line;
                    # still draw whatever segment exists (vit->convnext for CLIP variants).
                    pass
                ax.plot(xs, ys, color=METHOD_COLORS[method], linewidth=2.2, marker="o",
                         markersize=6, markeredgewidth=0, label=method, zorder=3)

            ax.axhline(0, color=INK_MUTED, linewidth=1, linestyle=(0, (2, 2)), zorder=1)
            ax.set_xticks(range(len(ARCHITECTURES)))
            ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHITECTURES], color=INK_MUTED, fontsize=9.5)
            ax.set_xlim(-0.15, len(ARCHITECTURES) - 1 + 0.15)
            ax.set_ylim(-0.25, 0.85)
            ax.tick_params(colors=INK_MUTED, labelsize=9)
            ax.yaxis.set_tick_params(labelcolor=INK_MUTED)
            if col_i == 0:
                ax.set_ylabel("Spearman rho\nvs. ground-truth faithfulness", color=INK_SECONDARY, fontsize=9.5)
            ax.grid(axis="y", color=GRID_HAIRLINE, linewidth=1, zorder=0)
            for spine_name, spine in ax.spines.items():
                spine.set_visible(spine_name == "bottom")
                if spine_name == "bottom":
                    spine.set_color(GRID_HAIRLINE)

            title = f"{task} — {GT_LABELS[gt_version]}"
            ax.set_title(title, color=INK_PRIMARY, fontsize=11.5, fontweight="bold", pad=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               labelcolor=INK_SECONDARY, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("CARDS vs. TCAV vs. PCBM: architecture generalization (official-train CelebA)",
                 color=INK_PRIMARY, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(OUT_PNG, dpi=150, facecolor=SURFACE)
    print(f"Saved {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
