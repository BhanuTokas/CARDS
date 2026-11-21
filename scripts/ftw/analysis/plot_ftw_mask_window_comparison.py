"""Mask-window comparison bar chart for FTW's masking-hybrid method: three
masking conditions (Window A, Window B, Both) per land-cover concept, from
pasted results (no source CSV -- values supplied directly).

Same grouped-vertical-bar-chart convention as `plot_ftw_encoder_ablation.py`
(zero baseline via axhline, dashed y-grid, rotated x labels, wide figsize
for a print-legible legend/labels). Concepts are kept in the order supplied
(already sorted ascending by "Mask Both", from most-suppressed -- urban/
agricultural land cover -- to most-supported -- open water), not re-sorted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/ftw_conceptmask_results")
OUT_PATH = RESULTS_DIR / "mask_window_comparison_bars.png"

# (full feature name, short concept name, Mask Window A, Mask Window B, Mask Both)
# Corrected mapping -- the concept-name labels in the first version of this
# table were misaligned with their numeric rows (prompted directly: "the
# labels were misplaced in the previous one"). The numeric values below are
# unchanged from before; only which name is attached to which row changed.
DATA: list[tuple[str, str, float, float, float]] = [
    ("Inland waters", "Inland waters", -0.1775133333, -0.114352549, -0.1852337255),
    ("Broad-leaved forest", "Broad-leaved forest", -0.2231564706, -0.08929019608, -0.1781780392),
    ("Urban fabric", "Urban", -0.1822443137, -0.04864, -0.1707807843),
    ("Coniferous forest", "Coniferous forest", -0.16122, -0.09578352941, -0.1367396078),
    ("Industrial or commercial units", "Industrial", -0.07869647059, -0.03455490196, -0.11432),
    ("Coastal wetlands", "Coastal wetlands", -0.05512745098, -0.03375882353, -0.08640117647),
    ("Moors, heathland and sclerophyllous vegetation", "Moors/heath/sclerophyll", -0.04624980392, -0.03134196078, -0.06233411765),
    ("Mixed forest", "Mixed forest", -0.09020117647, -0.02906117647, -0.06159647059),
    ("Transitional woodland-shrub", "Woodland-shrub", -0.02514313725, -0.04486, -0.05033019608),
    ("Agro-forestry areas", "Agro-forestry", 0.0008129411765, -0.03321372549, -0.03686392157),
    ("Marine waters", "Marine waters", -0.0151, -0.01029215686, -0.02936980392),
    ("Beaches, dunes, sands", "Beaches/dunes", -0.04115490196, 0.01468509804, -0.01360901961),
    ("Inland wetlands", "Inland wetlands", 0.006773333333, -0.02316627451, -0.001207058824),
    ("Natural grassland and sparsely vegetated areas", "Grass/sparse veg.", 0.008019215686, 0.0175254902, 0.01919529412),
    ("Pastures", "Pastures", 0.03802156863, 0.07525176471, 0.08907882353),
    ("Land principally occupied by agriculture, with significant areas of natural vegetation",
     "Agri. + natural veg.", 0.05088784314, 0.08284823529, 0.1033890196),
    ("Permanent crops", "Permanent crops", 0.03623607843, 0.05519372549, 0.1132498039),
    ("Complex cultivation patterns", "Complex cultivation", -0.00567254902, 0.01011490196, 0.1549203922),
    ("Arable land", "Arable", 0.1231839216, 0.2867294118, 0.3739835294),
]

WINDOW_COLORS = {"Mask Window A": "#3B82F6", "Mask Window B": "#F59E0B", "Mask Both": "#10B981"}


def main():
    short_names = [short for _full, short, *_rest in DATA]
    window_a = np.array([a for _f, _s, a, _b, _both in DATA])
    window_b = np.array([b for _f, _s, _a, b, _both in DATA])
    mask_both = np.array([both for *_rest, both in DATA])

    n_concepts = len(DATA)
    n_windows = 3
    x = np.arange(n_concepts)
    bar_w = 0.8 / n_windows

    fig, ax = plt.subplots(figsize=(17, 8))
    for i, (label, values) in enumerate(
        [("Mask Window A", window_a), ("Mask Window B", window_b), ("Mask Both", mask_both)]
    ):
        ax.bar(x + i * bar_w - 0.4 + bar_w / 2, values, width=bar_w,
               color=WINDOW_COLORS[label], edgecolor="black", linewidth=0.6, label=label)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylabel("Attribution", fontsize=22)
    ax.legend(fontsize=18, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_PATH}  ({n_concepts} concepts x {n_windows} mask windows)", flush=True)


if __name__ == "__main__":
    main()
