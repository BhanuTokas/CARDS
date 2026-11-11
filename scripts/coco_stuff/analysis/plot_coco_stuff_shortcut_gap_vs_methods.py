"""COCO-Stuff counterpart of `plot_shortcut_pooled_inverted_gap_vs_
methods_indexed.py` -- prompted directly ("Can you create a similar
graph for COCO-Stuff based on this data"), same dual-axis design
(inverted gap on its own real-scale left axis, 5 methods min-max
scaled to [0,1] on the right axis, no title -- a caption goes outside
this image), applied to a co-author's already-computed COCO-Stuff
shortcut-experiment numbers pasted directly into this script rather
than read from a CSV (no COCO-Stuff results file exists in this repo
yet -- this is a one-off chart from that pasted table, not part of a
larger COCO-Stuff pipeline here).

**Sign flip from the pasted table**: the pasted `Gap` column is
`shortcut_acc - clean_acc` (positive, growing with rate -- CelebA's
OWN original, non-inverted convention). This script's own `INV_GAP`
below negates it (`clean_acc - shortcut_acc`) to match this chart
family's established convention (`inverted gap`, negative and growing
in magnitude as reliance grows) -- so the left axis reads the same
direction as the CelebA version, not the pasted table's own sign.

**Method-name mapping** (pasted column -> this chart's label, matching
the CelebA chart's own 5-method set exactly): CARDS -> Masking hybrid,
TCAV -> TCAV (magnitude), PCBM-resnet -> PCBM (conventional),
PCBM-CLIP -> PCBM (CLIP-RN50), PCBM-SigLIP -> PCBM (SigLIP).

Accuracy values (Clean/Shortcut) are already plain 0-1 fractions in the
pasted table (e.g. 0.9615), confirmed directly against the `Gap`
column (0.9610 - 0.9615 = -0.0005, matching the pasted rate=0% Gap
exactly) -- "scaled by 100" describes how these fractions were derived
upstream, not a further rescaling needed here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
OUT_PNG = RESULTS_DIR / "coco_stuff_shortcut_inverted_gap_vs_methods.png"

RATES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
CLEAN_ACC = [0.9615, 0.9613, 0.9617, 0.9605, 0.9607, 0.9599, 0.9588, 0.9576, 0.9549, 0.9523, 0.9320]
SHORTCUT_ACC = [0.9610, 0.9905, 0.9933, 0.9949, 0.9956, 0.9970, 0.9977, 0.9986, 0.9993, 0.9997, 1.0000]

METHOD_SCORES = {
    "Masking hybrid": [0.3573, 0.3149, 0.3412, 0.2326, 0.3051, 0.3491, 0.2777, 0.2646, 0.2391, 0.2199, 0.0407],
    "TCAV (magnitude)": [0.5689, 0.5354, 0.4952, 0.5470, 0.5120, 0.5683, 0.5518, 0.6131, 0.6332, 0.6574, 0.6044],
    "PCBM (conventional)": [1.8793, 1.9379, 2.6160, 2.0039, 2.1386, 2.8883, 2.0478, 1.9285, 2.1525, 1.9900, 2.3981],
    "PCBM (CLIP-RN50)": [7.1247, 6.5669, 9.1923, 6.6037, 6.4928, 8.8211, 6.4320, 6.6272, 6.6297, 6.5382, 48.0692],
    "PCBM (SigLIP)": [5.4262, 5.3634, 5.4993, 5.3839, 5.3128, 7.1805, 5.2072, 5.3873, 5.3424, 5.2950, 32.2988],
}

METHOD_COLORS = {
    "Masking hybrid": "#2a78d6",       # slot 1, blue
    "TCAV (magnitude)": "#eb6834",     # slot 2, orange
    "PCBM (conventional)": "#1baf7a",  # slot 3, aqua
    "PCBM (SigLIP)": "#eda100",        # slot 4, yellow
    "PCBM (CLIP-RN50)": "#e87ba4",     # slot 5, magenta
}
GAP_COLOR = "#0b0b0b"

INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_HAIRLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

# inverted gap = clean - shortcut (negated from the pasted table's own shortcut-minus-clean convention)
INV_GAP = [c - s for c, s in zip(CLEAN_ACC, SHORTCUT_ACC)]


def min_max_scale(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return [(v - lo) / span for v in values]


def main():
    fig, ax_gap = plt.subplots(figsize=(11, 8.5), facecolor=SURFACE)
    ax_gap.set_facecolor(SURFACE)
    ax_attr = ax_gap.twinx()
    ax_attr.set_facecolor("none")

    gap_line, = ax_gap.plot(RATES, INV_GAP, color=GAP_COLOR, linewidth=3.2, linestyle=(0, (5, 2)),
                             marker="s", markersize=9, markeredgewidth=0, label="Inverted gap (clean − same-rate)", zorder=4)

    method_lines = [gap_line]
    for method_name, raw_scores in METHOD_SCORES.items():
        scaled = min_max_scale(raw_scores)
        line, = ax_attr.plot(RATES, scaled, color=METHOD_COLORS[method_name], linewidth=2.8, marker="o",
                              markersize=8, markeredgewidth=0, label=method_name, zorder=3)
        method_lines.append(line)

    # font sizes bumped significantly -- these charts get shrunk to ~half textwidth in the
    # paper (side-by-side figure), so on-screen-legible sizes here read as illegible in print.
    ax_gap.set_xlabel("Shortcut injection rate (%)", color=INK_SECONDARY, fontsize=22)
    ax_gap.set_ylabel("Accuracy gap (clean − same-rate)", color=GAP_COLOR, fontsize=21)
    ax_attr.set_ylabel("Min-max scaled mean |attribution|", color=INK_SECONDARY, fontsize=21)
    ax_gap.set_xticks(RATES)
    ax_attr.set_ylim(-0.05, 1.05)
    ax_gap.tick_params(colors=GAP_COLOR, labelsize=19)
    ax_gap.tick_params(axis="x", colors=INK_MUTED, labelsize=19)
    ax_attr.tick_params(colors=INK_MUTED, labelsize=19)
    ax_gap.grid(axis="y", color=GRID_HAIRLINE, linewidth=1, zorder=0)
    for spine_name, spine in ax_gap.spines.items():
        spine.set_visible(spine_name == "bottom")
        if spine_name == "bottom":
            spine.set_color(GRID_HAIRLINE)
    for spine_name, spine in ax_attr.spines.items():
        spine.set_visible(False)

    labels = [line.get_label() for line in method_lines]
    fig.legend(method_lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               frameon=False, labelcolor=INK_SECONDARY, fontsize=17, ncol=2, handlelength=2.5)

    fig.tight_layout(rect=(0, 0.22, 1, 1))
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, facecolor=SURFACE)
    print(f"Saved {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
