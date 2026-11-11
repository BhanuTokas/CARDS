
"""Build GT CSVs for all_methods_an_keep.py from the fills_*.csv scores.
Adds area / fill_frac / delta_p_stored (identical masks -> taken from lama_full.csv). Writes gt_{name}.csv for:
 each run; each inpainter averaged over its seeds; DIFF = mean of the three neutral/quality diffusion inpainters
 (each seed-averaged); ALL4 = DIFF's three + LaMa; NEG = SD-2 + SDXL concept-negative. Averages are per row (same
 image/concept/task) of delta_* and blob_*. Also prints row-level and concept-level agreement between inpainters."""
import itertools
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from paths import WORK as S

K = ["image", "concept_name", "target_task"]
V = ["delta_p", "delta_logit", "delta_logodds", "blob_p", "blob_logit", "blob_logodds"]
base = pd.read_csv(S / "lama_full.csv")[K + ["area", "fill_frac", "delta_p_stored", "n_valid"]]
runs = {p.stem[6:]: pd.read_csv(p).set_index(K) for p in sorted(S.glob("fills_*.csv"))}
print("runs:", list(runs))


def write(name, df):
    out = base.merge(df[V].reset_index(), on=K, how="inner")
    assert len(out) == len(base), (name, len(out))
    out.to_csv(S / f"gt_{name}.csv", index=False)


def avg(dfs): return sum(d[V] for d in dfs) / len(dfs)


for r, d in runs.items(): write(r, d)
inp = {}
for m in ["sd2_quality", "sdxl_quality", "flux_none"]:
    ss = [runs[k] for k in runs if k.startswith(m + "_s")]; inp[m] = avg(ss); write(m, inp[m]); print(m, "seeds", len(ss))
inp["lama"] = runs["lama_s0"][V]
write("DIFF", avg([inp[m] for m in ["sd2_quality", "sdxl_quality", "flux_none"]]))
write("ALL4", avg(list(inp.values())))
write("NEG", avg([runs["sd2_concept_s0"], runs["sdxl_concept_s0"]]))

print("\nconcept-level adjusted GT (logit) agreement, Spearman over 52 task-concepts:")
adj = {}
for m, d in {**inp, "sd2_concept": runs["sd2_concept_s0"][V], "sdxl_concept": runs["sdxl_concept_s0"][V]}.items():
    g = (d["delta_logit"] - d["blob_logit"]).groupby(level=["target_task", "concept_name"]).mean(); adj[m] = g
    raw = d["delta_logit"].groupby(level=["target_task", "concept_name"]).mean()
    print(f"  {m:14s} rho(raw, adjusted) {spearmanr(raw, g)[0]:+.2f}")
for a, b in itertools.combinations(adj, 2):
    print(f"  {a:14s} vs {b:14s} {spearmanr(adj[a], adj[b], nan_policy='omit')[0]:+.2f}")
for m in ["sd2_quality", "sdxl_quality", "flux_none"]:
    ss = sorted(k for k in runs if k.startswith(m + "_s"))
    g = [(runs[k]["delta_logit"] - runs[k]["blob_logit"]).groupby(level=["target_task", "concept_name"]).mean() for k in ss]
    print(f"  seed agreement {m}: " + ", ".join(f"{spearmanr(x, y)[0]:+.2f}" for x, y in itertools.combinations(g, 2)))
