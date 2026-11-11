# ruff: noqa: SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""Per-concept HnS scores on CelebA from celeba_soft.py output: raw, minus the fill-matched null, and the soft score
D - rho[task, fill] * A. rho = sum(second-patch drops) / sum(first-patch drops) over all images (no labels).
Images without a null keep their raw drop in 'raw' but are left out of the corrected scores; coverage is reported.
argv[1] = full.csv, argv[2] = output scores CSV (task, concept_name, raw, full_null, soft, n, n_corrected)."""
import csv
import sys
from collections import defaultdict

import numpy as np

rows = list(csv.DictReader(open(sys.argv[1])))
FILLS = sorted({c.rsplit("_first_", 1)[0] for c in rows[0] if "_first_" in c})
rho = {}
print("two-patch test (logit of each task; mean first-patch drop, mean second-patch drop, rho):")
for t in ("A", "M"):
    for f in FILLS:
        v = [(float(r[f"{f}_first_{t}"]), float(r[f"{f}_second_{t}"])) for r in rows if r[f"{f}_first_{t}"] != ""]
        a = np.array(v)
        rho[t, f] = a[:, 1].sum() / a[:, 0].sum()
        print(f"  {'Attractive' if t == 'A' else 'Male':10s} {f:16s} n={len(a)} first {a[:, 0].mean():+.3f} second {a[:, 1].mean():+.3f} rho {rho[t, f]:+.2f}")
fills_used = defaultdict(int)
for r in rows:
    fills_used[r["fill"]] += 1
print("fills chosen by best-of-7:", dict(fills_used))
out = []
per = defaultdict(lambda: defaultdict(list))
for r in rows:
    for t, task in (("A", "Attractive"), ("M", "Male")):
        D = float(r[f"real_{t}"])
        per[task, r["concept"]]["raw"].append(D)
        if r[f"null_{t}"] != "":
            A = float(r[f"null_{t}"])
            per[task, r["concept"]]["full_null"].append(D - A)
            per[task, r["concept"]]["soft"].append(D - rho[t, r["fill"]] * A)
            per[task, r["concept"]]["A"].append(A)
with open(sys.argv[2], "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["task", "concept_name", "raw", "full_null", "soft", "null_mean", "n", "n_corrected"])
    for (task, c), d in sorted(per.items()):
        nc = len(d["soft"])
        w.writerow([task, c, np.mean(d["raw"]), np.mean(d["full_null"]) if nc else "", np.mean(d["soft"]) if nc else "",
                    np.mean(d["A"]) if nc else "", len(d["raw"]), nc])
        if nc < 0.5 * len(d["raw"]):
            print(f"  low coverage: {task} {c} corrected on {nc}/{len(d['raw'])} images")
print("wrote", sys.argv[2])
