"""Visualizes the v61 masking-strategy-bias finding (+ its three
follow-ups: zero_fill_noise for "solid", white_fill for "black") --
prompted directly ("Can you create the plots for the new analysis?").

Panel 1: mean delta_p by attribute TYPE (color/pattern) under blur,
zero_fill, and hue_shift -- the core triangulation result showing each
strategy's bias flips depending on what information it destroys.
Panel 2: the two targeted fixes -- zero_fill_noise vs plain zero_fill
for "solid" (does noise reduce the OOD-hole inflation?), and white_fill
vs plain zero_fill for "black" (does a maximally-contrasting fill color
give a stronger signal?).

All numbers are hardcoded from the already-computed, already-logged
results in notes/cub_correlation_investigation.md (v61 and its
follow-ups) -- this script is presentation-only, it does not recompute
anything.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
OUT_PATH = RESULTS_DIR / "masking_strategy_bias.png"

# --- Panel 1: color vs pattern, three strategies (v61 core finding) ---
STRATEGIES = ["blur", "zero_fill", "hue_shift"]
STRATEGY_LABELS = ["Blur\n(preserves color)", "Zero-fill\n(destroys both)", "Hue-shift\n(preserves structure)"]
COLOR_DELTAS = [0.0351, 0.0441, 0.0278]
PATTERN_DELTAS = [0.0408, 0.0457, 0.0207]

# --- Panel 2: targeted fixes for "solid" and "black" ---
SOLID_LABELS = ["Blur", "Zero-fill", "Zero-fill\n+ noise", "Hue-shift"]
SOLID_DELTAS = [0.0079, 0.0151, 0.0095, 0.0016]
BLACK_LABELS = ["Blur", "Zero-fill\n(black)", "White-fill", "Hue-shift"]
BLACK_DELTAS = [0.0117, 0.0099, 0.0164, 0.0019]

COLOR_HUE = "#c76b3f"    # warm terracotta -- for "color" attributes
PATTERN_HUE = "#3f6fc7"  # cool blue -- for "pattern" attributes
NEUTRAL = "#6b6b6b"


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#333333")
    ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def bar_with_labels(ax, x, heights, color, width, offset=0.0, label=None):
    bars = ax.bar(x + offset, heights, width=width, color=color, zorder=3, label=label, edgecolor="white", linewidth=0.5)
    for rect, h in zip(bars, heights):
        ax.text(rect.get_x() + rect.get_width() / 2, h + max(heights) * 0.02, f"{h:.4f}",
                 ha="center", va="bottom", fontsize=8, color="#333333")
    return bars


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1
    ax = axes[0]
    x = np.arange(len(STRATEGIES))
    width = 0.35
    bar_with_labels(ax, x, COLOR_DELTAS, COLOR_HUE, width, offset=-width / 2, label="color attributes")
    bar_with_labels(ax, x, PATTERN_DELTAS, PATTERN_HUE, width, offset=width / 2, label="pattern attributes")
    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGY_LABELS, fontsize=9)
    ax.set_ylabel("mean $\\Delta p$ (masking effect)")
    ax.set_title("Each fill strategy's bias tracks what it destroys", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_ylim(0, 0.052)
    style_axis(ax)

    # Panel 2a: solid
    ax = axes[1]
    x = np.arange(len(SOLID_LABELS))
    colors = [NEUTRAL, PATTERN_HUE, "#2a7f62", NEUTRAL]
    bars = ax.bar(x, SOLID_DELTAS, color=colors, zorder=3, edgecolor="white", linewidth=0.5)
    for rect, h in zip(bars, SOLID_DELTAS):
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.0005, f"{h:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(SOLID_LABELS, fontsize=9)
    ax.set_ylabel("mean $\\Delta p$")
    ax.set_title('"solid" (pattern) -- noise pulls\nzero-fill back toward blur', fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.017)
    style_axis(ax)

    # Panel 2b: black
    ax = axes[2]
    x = np.arange(len(BLACK_LABELS))
    colors = [NEUTRAL, COLOR_HUE, "#8b3fc7", NEUTRAL]
    bars = ax.bar(x, BLACK_DELTAS, color=colors, zorder=3, edgecolor="white", linewidth=0.5)
    for rect, h in zip(bars, BLACK_DELTAS):
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.0005, f"{h:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(BLACK_LABELS, fontsize=9)
    ax.set_ylabel("mean $\\Delta p$")
    ax.set_title('"black" (color) -- contrasting fill\nbeats same-color fill', fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.019)
    style_axis(ax)

    fig.suptitle("Masking-strategy bias in the faithfulness metric (v61 + follow-ups, n=60/group diagnostic)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
