# ruff: noqa: SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""Summarise removal_check output. argv[1]=removal CSV. Positives = images labelled with the concept (CelebAMask-HQ labels).
'still' = still detected after the fill, among positives the detector caught on the original.
'drop'  = (s_orig - s_fill) / gap, gap = median original score of positives minus the detector threshold;
          1.0 means a typical positive is pushed down to the 'concept absent' boundary."""
import collections
import csv
import sys

import numpy as np

rows = list(csv.DictReader(open(sys.argv[1])))
by = collections.defaultdict(list)
for r in rows: by[r["concept_name"]].append(r)
V = ["lama", "blur", "rand_lama"]
print(f"{'concept':20s} {'detAUC':>6s} {'npos':>4s} {'det0':>5s} | still detected after: {'LaMa':>5s} {'blur':>5s} {'randL':>5s} | "
      f"norm. drop: {'LaMa':>5s} {'blur':>5s} {'randL':>5s}")
agg = collections.defaultdict(list); summ = {}
for c in sorted(by):
    v = by[c]; t = float(v[0]["threshold"]); auc = float(v[0]["detector_auroc"])
    pos = [r for r in v if r["label"] == "1"]; neg = [r for r in v if r["label"] == "0"]
    if len(pos) < 10:
        print(f"{c:20s} {auc:6.3f} {len(pos):4d}  too few labelled positives"); continue
    so = np.array([float(r["s_orig"]) for r in pos])
    gap = np.median(so) - t  # all GT images are labelled positives, so normalise by distance to the decision threshold
    det = so > t
    out = {}
    for k in V:
        p = [r for r in pos if r["s_" + k] != ""]
        s = np.array([float(r["s_" + k]) for r in p]); o = np.array([float(r["s_orig"]) for r in p]); d = o > t
        out[k] = ((s[d] > t).mean() if d.any() else np.nan, np.median((o - s) / gap))
        agg[k].append(out[k])
    summ[c] = dict(auc=auc, npos=len(pos), det=det.mean(), **{f"still_{k}": out[k][0] for k in V}, **{f"drop_{k}": out[k][1] for k in V})
    flag = " (weak detector)" if auc < 0.8 else ""
    print(f"{c:20s} {auc:6.3f} {len(pos):4d} {det.mean():5.2f} |                       {out['lama'][0]:5.2f} {out['blur'][0]:5.2f} "
          f"{out['rand_lama'][0]:5.2f} |             {out['lama'][1]:+5.2f} {out['blur'][1]:+5.2f} {out['rand_lama'][1]:+5.2f}{flag}")
good = [c for c in summ if summ[c]["auc"] >= 0.8]
for name, cs in (("all concepts", list(summ)), ("detector AUROC>=0.8", good)):
    print(f"\nmedian over {name} (n={len(cs)}): still detected LaMa {np.nanmedian([summ[c]['still_lama'] for c in cs]):.2f}, "
          f"blur {np.nanmedian([summ[c]['still_blur'] for c in cs]):.2f}, random-shape LaMa {np.nanmedian([summ[c]['still_rand_lama'] for c in cs]):.2f}; "
          f"norm. drop LaMa {np.median([summ[c]['drop_lama'] for c in cs]):+.2f}, blur {np.median([summ[c]['drop_blur'] for c in cs]):+.2f}, "
          f"random-shape LaMa {np.median([summ[c]['drop_rand_lama'] for c in cs]):+.2f}")
print("\nLaMa leaves the concept detected in >50% of detected positives:",
      sorted(c for c in good if summ[c]["still_lama"] > 0.5))
print("LaMa removes less than blur (higher 'still'):", sorted(c for c in good if summ[c]["still_lama"] > summ[c]["still_blur"] + 0.05))
with open(sys.argv[1].replace(".csv", "_summary.csv"), "w", newline="") as f:
    w = csv.writer(f); keys = list(next(iter(summ.values())))
    w.writerow(["concept_name"] + keys)
    for c, d in summ.items(): w.writerow([c] + [d[k] for k in keys])
