"""The paper's shortcut metric and figure (appendix "Shortcut Experiment") with the new baselines added.

Metric, as in score_shortcut_pooled_inverted_gap_vs_methods.py (branch Shortcut_v2), which produced the paper's
CelebA column: at each rate, the plain mean over Attractive and Male of (a) the accuracy gap, clean - same-rate val
accuracy, and (b) a method's mean |score| over the 26 concepts; then Pearson r (and Spearman rho) of the two pooled
series over the 11 rates, and over 0-90%. Positive r = attribution falls as shortcut reliance grows.

Figure, as plot_shortcut_pooled_inverted_gap_vs_methods_indexed.py (same branch), the paper's imgs/shortcut/celebA.png:
pooled gap on the left axis at its real scale, each method's pooled mean |score| min-max scaled to [0, 1] on the right.
Two versions: every method, and Hide and Seek with the four new baselines only.

Inputs: $SHORTCUT_OUT/shortcut_all_methods_mean_abs.csv (from summarize.py) and
$CARDS_RESULTS_DIR/celeba_official_train_shortcut_accuracy_gap_summary.csv.
Outputs in $SHORTCUT_OUT: shortcut_pooled_vs_gap.csv, shortcut_pooled_all_methods.{png,pdf},
shortcut_pooled_new_baselines.{png,pdf}."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr

R = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
S = Path(os.environ.get("SHORTCUT_OUT", "results/shortcut_baselines"))
RATES = list(range(0, 101, 10))

# method in summarize.py -> (legend label, color, marker). Nine well-separated hues, each with its own marker so the
# lines stay distinguishable in grayscale and for colorblind readers (the gap keeps the black squares).
STYLE = {
    "Masking hybrid": ("Hide and Seek", "#1f77b4", "o"),
    "TCAV (magnitude)": ("TCAV (magnitude)", "#ff7f0e", "D"),
    "PCBM (conventional)": ("PCBM (conventional)", "#2ca02c", "^"),
    "PCBM (SigLIP)": ("PCBM (SigLIP)", "#bcbd22", "v"),
    "PCBM (CLIP-RN50)": ("PCBM (CLIP-RN50)", "#17becf", "<"),
    "ConceptSAE": ("ConceptSAE", "#9467bd", "P"),
    "EAC": ("EAC", "#8c564b", "X"),
    "ConceptSHAP (signed)": ("ConceptSHAP (signed)", "#d62728", ">"),
    "ConceptSHAP (original)": ("ConceptSHAP (original)", "#e377c2", "*"),
}
FIGURES = {"shortcut_pooled_all_methods": list(STYLE),
           "shortcut_pooled_new_baselines": ["Masking hybrid", "ConceptSAE", "EAC", "ConceptSHAP (signed)",
                                             "ConceptSHAP (original)"]}
ALPHA = 0.6  # method lines semi-transparent so crossings stay readable; the gap line stays opaque
GAP_COLOR, INK_SECONDARY, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def pooled() -> tuple[pd.Series, pd.DataFrame]:
    g = pd.read_csv(R / "celeba_official_train_shortcut_accuracy_gap_summary.csv")
    gap = (g.clean_val_acc - g.same_rate_val_acc).groupby(g.rate_pct).mean().reindex(RATES)
    m = pd.read_csv(S / "shortcut_all_methods_mean_abs.csv")
    n_tasks = m.groupby(["method", "rate_pct"]).task.nunique()
    assert (n_tasks == 2).all(), f"a method/rate is missing a task: {n_tasks[n_tasks != 2]}"
    attr = m.groupby(["rate_pct", "method"]).mean_abs.mean().unstack().reindex(RATES)
    return gap, attr


def scores(gap: pd.Series, attr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for meth in attr:
        row = {"method": meth}
        for lab, rs in (("all", RATES), ("0_90", RATES[:-1])):
            p, s = pearsonr(attr.loc[rs, meth], gap[rs]), spearmanr(attr.loc[rs, meth], gap[rs])
            row |= {f"pearson_{lab}": p[0], f"pearson_p_{lab}": p[1], f"spearman_{lab}": s[0], f"spearman_p_{lab}": s[1]}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pearson_all", ascending=False)


def plot(gap: pd.Series, attr: pd.DataFrame, methods: list[str], stem: str):
    rows = -(-(len(methods) + 1) // 2)  # legend rows, 2 columns
    h = 6.6 + 0.45 * rows
    fig, ax_gap = plt.subplots(figsize=(11, h), facecolor=SURFACE)
    ax_gap.set_facecolor(SURFACE)
    ax_attr = ax_gap.twinx(); ax_attr.set_facecolor("none")
    lines = ax_gap.plot(RATES, gap.values, color=GAP_COLOR, linewidth=3.2, linestyle=(0, (5, 2)), marker="s",
                        markersize=9, markeredgewidth=0, label="Inverted gap (clean − same-rate)", zorder=4)
    for meth in methods:
        y = attr[meth]; y = (y - y.min()) / ((y.max() - y.min()) or 1.0)
        lab, col, mk = STYLE[meth]
        lines += ax_attr.plot(RATES, y.values, color=col, linewidth=2.8, marker=mk, markersize=11 if mk == "*" else 9,
                              markeredgewidth=0, label=lab, zorder=3, alpha=ALPHA)
    ax_gap.set_xlabel("Shortcut injection rate (%)", color=INK_SECONDARY, fontsize=22)
    ax_gap.set_ylabel("Accuracy gap (clean − same-rate)", color=GAP_COLOR, fontsize=21)
    ax_attr.set_ylabel("Min-max scaled mean |attribution|", color=INK_SECONDARY, fontsize=21)
    ax_gap.set_xticks(RATES); ax_attr.set_ylim(-0.05, 1.05)
    ax_gap.tick_params(colors=GAP_COLOR, labelsize=19); ax_gap.tick_params(axis="x", colors=INK_MUTED, labelsize=19)
    ax_attr.tick_params(colors=INK_MUTED, labelsize=19)
    ax_gap.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    for name, sp in ax_gap.spines.items():
        sp.set_visible(name == "bottom"); sp.set_color(GRID)
    for sp in ax_attr.spines.values():
        sp.set_visible(False)
    fig.legend(lines, [ln.get_label() for ln in lines], loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=False,
               labelcolor=INK_SECONDARY, fontsize=17, ncol=2, handlelength=2.5)
    fig.tight_layout(rect=(0, (0.45 * rows + 0.1) / h, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(S / f"{stem}.{ext}", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    gap, attr = pooled()
    sc = scores(gap, attr)
    sc.to_csv(S / "shortcut_pooled_vs_gap.csv", index=False)
    print(sc[["method", "pearson_all", "pearson_p_all", "spearman_all", "pearson_0_90", "spearman_0_90"]]
          .round(3).to_string(index=False))
    for stem, methods in FIGURES.items():
        plot(gap, attr, methods, stem)
    print(f"wrote {S / 'shortcut_pooled_vs_gap.csv'} and {', '.join(FIGURES)} (.png, .pdf) in {S}")


if __name__ == "__main__":
    main()
