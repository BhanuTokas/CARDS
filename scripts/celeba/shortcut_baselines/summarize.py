"""All methods of the official-train shortcut experiment side by side: the existing runs (masking hybrid, TCAV, PCBM
x3) plus the baselines of run_baseline.py (ConceptSAE, ConceptSHAP original and signed, EAC).

The question of the experiment: as the shortcut rate rises and the model leans on the injected cue, does a method's
attribution to the 26 real concepts fall? Per method, task and rate: mean |score| over the 26 concepts. Then, as in
celeba_official_train_shortcut_inverted_gap_vs_all_methods_pearson.csv, Pearson and Spearman correlation of that
series with the inverted accuracy gap (-(same-rate val acc - clean val acc)) over the 11 rates; positive = the
attribution declines as shortcut reliance grows. Reported over all 11 rates and over 0-90% (the 100% model is the
extreme case, where clean accuracy collapses).

Inputs: $CARDS_RESULTS_DIR (default results) for the existing CSVs and the accuracy-gap summary;
$SHORTCUT_OUT (default results/shortcut_baselines) for {task}_{rate}/{sae,shap,eac}.csv.
Outputs in $SHORTCUT_OUT: shortcut_all_methods_mean_abs.csv (method, task, rate_pct, mean_abs, n_concepts) and
shortcut_all_methods_vs_gap.csv."""
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

R = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
S = Path(os.environ.get("SHORTCUT_OUT", "results/shortcut_baselines"))
TASKS, RATES = ["Attractive", "Male"], list(range(0, 101, 10))
EXISTING = [("Masking hybrid", "cards_celeba_official_train_shortcut_experiment_raw_scores.csv", "hybrid_raw_score"),
            ("TCAV (magnitude)", "tcav_celeba_official_train_shortcut_experiment.csv", "mean_magnitude"),
            ("PCBM (conventional)", "pcbm_celeba_official_train_shortcut_experiment.csv", "weight"),
            ("PCBM (SigLIP)", "pcbm_clip_concepts_shortcut_official_train_experiment_siglip.csv", "weight"),
            ("PCBM (CLIP-RN50)", "pcbm_clip_concepts_shortcut_official_train_experiment_clip_rn50.csv", "weight")]
NEW = [("ConceptSAE", "sae.csv", "score"), ("ConceptSHAP (signed)", "shap.csv", "shap_logit"),
       ("ConceptSHAP (original)", "shap.csv", "shap_acc"), ("EAC", "eac.csv", "score")]


def read(p):
    return list(csv.DictReader(open(p, newline="")))


def main():
    vals = defaultdict(list)  # (method, task, rate) -> |scores|
    for m, f, col in EXISTING:
        if not (R / f).exists():
            print(f"missing {R / f}"); continue
        for r in read(R / f):
            vals[m, r["task"], int(r["rate_pct"])].append(abs(float(r[col])))
    missing = []
    for m, f, col in NEW:
        for t in TASKS:
            for rate in RATES:
                p = S / f"{t.lower()}_{rate}" / f
                if not p.exists():
                    missing.append(str(p)); continue
                vals[m, t, rate] += [abs(float(r[col])) for r in read(p) if r["task"] == t and r[col] not in ("", "nan")]
    if missing:
        print(f"{len(missing)} baseline files missing, e.g. {sorted(set(missing))[:3]}")
    gap = {(r["task"], int(r["rate_pct"])): float(r["gap"]) for r in read(R / "celeba_official_train_shortcut_accuracy_gap_summary.csv")}
    S.mkdir(parents=True, exist_ok=True)
    with open(S / "shortcut_all_methods_mean_abs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "task", "rate_pct", "mean_abs", "n_concepts"])
        for (m, t, rate), v in sorted(vals.items()):
            w.writerow([m, t, rate, np.mean(v), len(v)])
    rows = []
    print(f"{'method':24s} {'task':10s} | r all 11 (rho) | r rates 0-90 (rho)")
    for m in [x[0] for x in EXISTING + NEW]:
        for t in TASKS:
            out = {"method": m, "task": t}
            for lab, rs in (("all", RATES), ("0_90", RATES[:-1])):
                rs = [x for x in rs if (m, t, x) in vals]
                if len(rs) < 4:
                    continue
                y = [np.mean(vals[m, t, x]) for x in rs]; g = [-gap[t, x] for x in rs]
                out |= {f"n_{lab}": len(rs), f"pearson_{lab}": pearsonr(y, g)[0], f"spearman_{lab}": spearmanr(y, g)[0]}
            if "n_all" in out:
                rows.append(out)
                print(f"{m:24s} {t:10s} | {out['pearson_all']:+.2f} ({out['spearman_all']:+.2f}) | "
                      f"{out.get('pearson_0_90', np.nan):+.2f} ({out.get('spearman_0_90', np.nan):+.2f})")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("method", "task"), k))
    with open(S / "shortcut_all_methods_vs_gap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {S / 'shortcut_all_methods_mean_abs.csv'} and {S / 'shortcut_all_methods_vs_gap.csv'}")


if __name__ == "__main__":
    main()
