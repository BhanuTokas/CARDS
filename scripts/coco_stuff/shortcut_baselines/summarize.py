"""The paper's COCO-Stuff shortcut metric and figure with the new baselines (ConceptSAE, ConceptSHAP original and signed,
EAC) added.

Metric, as in the paper's COCO-Stuff column: per method and rate, mean |score| over every (concept, class) row; then
Pearson r (and Spearman rho) of that series with the inverted accuracy gap, clean - shortcut test accuracy, over the 11
rates, and over 0-90%. Positive r = attribution falls as shortcut reliance grows. COCO-Stuff has one 20-class model per
rate, so there is no pooling over tasks (unlike CelebA's Attractive and Male).

The paper's five methods and the accuracies come from the values in
scripts/coco_stuff/analysis/plot_coco_stuff_shortcut_gap_vs_methods.py (the co-author's runs; no CSV of them is in the
repo). If $COCO_SHORTCUT_ACC (default results/coco_stuff/shortcut/shortcut_model_accuracy.csv) exists, its accuracies are
checked against those values.

Inputs: $SHORTCUT_OUT/rate_{rate}/{sae,shap,eac}.csv (default results/coco_stuff/shortcut_baselines).
Outputs in $SHORTCUT_OUT: shortcut_all_methods_mean_abs.csv, shortcut_vs_gap.csv,
shortcut_all_methods.{png,pdf}, shortcut_new_baselines.{png,pdf} (same style as the CelebA figures)."""
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, "scripts/coco_stuff/analysis")
import plot_coco_stuff_shortcut_gap_vs_methods as paper

S = Path(os.environ.get("SHORTCUT_OUT", "results/coco_stuff/shortcut_baselines"))
ACC = Path(os.environ.get("COCO_SHORTCUT_ACC", "results/coco_stuff/shortcut/shortcut_model_accuracy.csv"))
RATES = paper.RATES
NEW = [("ConceptSAE", "sae.csv", "score"), ("ConceptSHAP (signed)", "shap.csv", "shap_logit"),
       ("ConceptSHAP (original)", "shap.csv", "shap_acc"), ("EAC", "eac.csv", "score")]
# method -> (legend label, color, marker); the same as the CelebA figures (scripts/celeba/shortcut_baselines/plot_pooled.py)
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
FIGURES = {"shortcut_all_methods": list(STYLE),
           "shortcut_new_baselines": ["Masking hybrid", "ConceptSAE", "EAC", "ConceptSHAP (signed)", "ConceptSHAP (original)"]}
GAP_COLOR, INK_SECONDARY, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def series() -> tuple[pd.Series, pd.DataFrame]:
    gap = pd.Series(paper.INV_GAP, index=RATES)
    if ACC.exists():
        a = pd.read_csv(ACC).set_index("rate_pct").reindex(RATES)
        assert np.allclose(a.mean_clean_acc - a.mean_shortcut_acc, gap, atol=1e-3), f"{ACC} disagrees with the paper's values"
    attr = pd.DataFrame(paper.METHOD_SCORES, index=RATES)
    missing = []
    for m, f, col in NEW:
        vals = {}
        for r in RATES:
            p = S / f"rate_{r}" / f
            if not p.exists():
                missing.append(str(p)); continue
            v = pd.to_numeric(pd.read_csv(p)[col], errors="coerce").abs().dropna()
            if len(v):
                vals[r] = v.mean()
        if len(vals) >= 4:
            attr[m] = pd.Series(vals)
        else:
            print(f"{m}: only {len(vals)} rates with values, left out")
    if missing:
        print(f"{len(missing)} baseline files missing, e.g. {sorted(set(missing))[:3]}")
    return gap, attr


def scores(gap: pd.Series, attr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in attr:
        row = {"method": m}
        for lab, rs in (("all", RATES), ("0_90", RATES[:-1])):
            rs = [r for r in rs if not np.isnan(attr.at[r, m])]
            p, s = pearsonr(attr.loc[rs, m], gap[rs]), spearmanr(attr.loc[rs, m], gap[rs])
            row |= {f"n_{lab}": len(rs), f"pearson_{lab}": p[0], f"pearson_p_{lab}": p[1], f"spearman_{lab}": s[0]}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pearson_all", ascending=False)


def plot(gap: pd.Series, attr: pd.DataFrame, methods: list[str], stem: str):
    methods = [m for m in methods if m in attr]
    rows = -(-(len(methods) + 1) // 2)  # legend rows, 2 columns
    h = 6.6 + 0.45 * rows
    fig, ax_gap = plt.subplots(figsize=(11, h), facecolor=SURFACE)
    ax_gap.set_facecolor(SURFACE)
    ax_attr = ax_gap.twinx(); ax_attr.set_facecolor("none")
    lines = ax_gap.plot(RATES, gap.values, color=GAP_COLOR, linewidth=3.2, linestyle=(0, (5, 2)), marker="s",
                        markersize=9, markeredgewidth=0, label="Inverted gap (clean − shortcut)", zorder=4)
    for m in methods:
        y = attr[m]; y = (y - y.min()) / ((y.max() - y.min()) or 1.0)
        lab, col, mk = STYLE[m]
        lines += ax_attr.plot(RATES, y.values, color=col, linewidth=2.8, marker=mk, markersize=11 if mk == "*" else 9,
                              markeredgewidth=0, label=lab, zorder=3)
    ax_gap.set_xlabel("Shortcut injection rate (%)", color=INK_SECONDARY, fontsize=22)
    ax_gap.set_ylabel("Accuracy gap (clean − shortcut)", color=GAP_COLOR, fontsize=21)
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
    S.mkdir(parents=True, exist_ok=True)
    gap, attr = series()
    with open(S / "shortcut_all_methods_mean_abs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "rate_pct", "mean_abs", "inv_gap"])
        for m in attr:
            for r in RATES:
                w.writerow([m, r, attr.at[r, m], gap[r]])
    sc = scores(gap, attr)
    sc.to_csv(S / "shortcut_vs_gap.csv", index=False)
    print(sc[["method", "n_all", "pearson_all", "pearson_p_all", "spearman_all", "pearson_0_90", "spearman_0_90"]]
          .round(3).to_string(index=False))
    for stem, methods in FIGURES.items():
        plot(gap, attr, methods, stem)
    print(f"wrote shortcut_all_methods_mean_abs.csv, shortcut_vs_gap.csv and {', '.join(FIGURES)} (.png, .pdf) in {S}")


if __name__ == "__main__":
    main()
