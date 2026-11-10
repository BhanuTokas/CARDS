# ruff: noqa: C408
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""Summarise removal_runs.csv (diagnostic only). Positives = images labelled with the concept.
drop = median over positives of (s_orig - s_fill) / gap, gap = median positive s_orig - threshold (1.0 = pushed to the boundary).
still = share of detected positives still detected after the fill. m1 = first random-shape fill (artifact baseline)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from paths import WORK

d = pd.read_csv(WORK / "removal_runs.csv"); d = d[d.label == 1]
RUNS = ["lama_s0", "sd2_quality_s0", "sdxl_quality_s0", "flux_none_s0", "sd2_concept_s0", "sdxl_concept_s0"]
SHORT = ["LaMa", "SD2", "SDXL", "FLUX", "SD2neg", "SDXLneg"]
rows = []
for c, g in d.groupby("concept_name"):
    t = g.threshold.iloc[0]; gap = g[g.run == RUNS[0]].s_orig.median() - t
    r = dict(concept=c, auc=g.detector_auroc.iloc[0])
    for run, s in zip(RUNS, SHORT):
        x = g[g.run == run]; det = x.s_orig > t
        r["drop_" + s] = np.median((x.s_orig - x.s_m0) / gap); r["still_" + s] = (x.s_m0[det] > t).mean()
        m1 = x.dropna(subset=["s_m1"]); r["ctl_" + s] = np.median((m1.s_orig - m1.s_m1) / gap)
    rows.append(r)
R = pd.DataFrame(rows).set_index("concept")
pd.set_option("display.width", 200); pd.set_option("display.precision", 2)
print("normalised drop of the concept fill (m0); detector AUROC in first column")
print(R[["auc"] + ["drop_" + s for s in SHORT]].to_string())
print("\nshare of detected positives STILL detected after the concept fill")
print(R[["still_" + s for s in SHORT]].to_string())
good = R[R.auc >= 0.8]
print(f"\nmedians over {len(good)} concepts with detector AUROC>=0.8:")
for s in SHORT:
    print(f"  {s:8s} drop {good['drop_' + s].median():+.2f}  still {good['still_' + s].median():.2f}  random-shape drop {good['ctl_' + s].median():+.2f}")
R.to_csv(WORK / "removal_runs_summary.csv")
