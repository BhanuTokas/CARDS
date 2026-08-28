"""Visualizes the v60 winner's-curse finding -- prompted directly ("Can
you create the plots for the new analysis?"): sign agreement for CARDS
(the grid-search winner), PCBM (a fixed, never-tuned-against-this-ground-
truth checkpoint), and TCAV (a fixed algorithm) on the 282 "overlap"
pairs (already reachable under the smaller v53 ground truth) vs. the 970
"newly added" pairs (only reachable once the species-per-attribute
target was raised) -- CARDS collapses to non-significance on the fresh
pairs, PCBM/TCAV barely move.

Numbers hardcoded from the already-computed, already-logged v60 result
in notes/cub_correlation_investigation.md -- presentation only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
OUT_PATH = RESULTS_DIR / "winners_curse_overlap_vs_new.png"

METHODS = ["CARDS\n(grid-search winner)", "PCBM (official)\n(fixed checkpoint)", "TCAV\n(fixed algorithm)"]
OVERLAP_SIGN = [67.0, 63.1, 76.2]
OVERLAP_P = ["p=1.1e-08", "p=1.2e-05", "p=2.9e-19"]
NEW_SIGN = [52.6, 58.6, 66.9]
NEW_P = ["p=0.12 (n.s.)", "p=1.1e-07", "p=2.9e-26"]
DROPS = [-14.4, -4.5, -9.3]


def main():
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(METHODS))
    width = 0.35

    c_overlap = "#2a7f62"
    c_new = "#c1443c"

    bars1 = ax.bar(x - width / 2, OVERLAP_SIGN, width, label="Overlap pairs (n=282)\nalready reachable under the smaller ground truth",
                    color=c_overlap, zorder=3, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, NEW_SIGN, width, label="Newly-added pairs (n=970)\nonly reached once sampling scaled up",
                    color=c_new, zorder=3, edgecolor="white", linewidth=0.5)

    for rect, val, p in zip(bars1, OVERLAP_SIGN, OVERLAP_P):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 1.5, f"{val:.1f}%\n{p}", ha="center", va="bottom", fontsize=8.5)
    for rect, val, p in zip(bars2, NEW_SIGN, NEW_P):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 1.5, f"{val:.1f}%\n{p}", ha="center", va="bottom", fontsize=8.5,
                 fontweight="bold" if "n.s." in p else "normal", color="#c1443c" if "n.s." in p else "#333333")

    ax.axhline(50, color="#999999", linewidth=1, linestyle="--", zorder=2)
    ax.text(len(METHODS) - 0.4, 51, "chance (50%)", fontsize=8, color="#999999", va="bottom")

    for i, drop in enumerate(DROPS):
        ax.annotate("", xy=(x[i] + width / 2, NEW_SIGN[i] + 8), xytext=(x[i] - width / 2, OVERLAP_SIGN[i] + 8),
                     arrowprops=dict(arrowstyle="->", color="#666666", lw=1.2))
        ax.text(x[i], max(OVERLAP_SIGN[i], NEW_SIGN[i]) + 11, f"{drop:+.1f} pts", ha="center", fontsize=8.5,
                 color="#666666", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=10)
    ax.set_ylabel("Sign agreement with faithfulness ground truth")
    ax.set_ylim(0, 95)
    ax.set_title("Winner's-curse check: does performance hold up on genuinely fresh pairs?\n"
                  "(v60 -- only CARDS, the repeatedly grid-searched method, loses significance)",
                  fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
