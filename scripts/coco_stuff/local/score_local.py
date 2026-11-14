"""Merge and score the COCO-Stuff local (per-image) attributions (ResNet18).

Inputs in $COCO_LOCAL_OUT (or argv[1]): records.csv, cards_*.csv, tcav.csv, eac_*.csv (present records only; absent
records get EAC 0, as EAC has no region for an undetected concept), sae.csv.
Methods are scored on the records every method has (present records without a saved EAC region are dropped and counted).

Outputs:
  local_merged.csv      one row per record with every method's score
  local_scores.csv      per method:
    rho_pooled   Spearman(score, ground-truth delta_p) over all present records, 95% CI from image-cluster bootstraps
    rho_within   mean over (concept, class) pairs of the Spearman correlation inside the pair (present records): the
                 per-image ranking alone, without the between-pair differences the pooled value also rewards;
                 95% CI from bootstrapping pairs
    hit@N        per pair, rank present + absent records by |score|: share of the top N that are present and share of
                 the bottom N that are absent; mean over pairs; N = 3 (as in the figures), 5, 10. Ties are broken at
                 random (expectation over 200 orders), so file order cannot favour a method.
    auroc / ap   of |score| for present vs absent, mean over pairs (AP chance = share of present records)
  local_pair_scores.csv the same per pair"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

METHODS = ["cards", "tcav_sign", "tcav_dd", "tcav4_dd", "eac", "sae"]
LABELS = {"cards": "Hide and Seek (CARDS)", "tcav_sign": "TCAV layer3 (sign)", "tcav_dd": "TCAV layer3 (derivative)",
          "tcav4_dd": "TCAV layer4 (derivative)", "eac": "EAC", "sae": "ConceptSAE"}
NS, N_BOOT, N_TIE = (3, 5, 10), 1000, 200


def read(p):
    return list(csv.DictReader(open(p, newline="")))


def key(r):
    return r["set"], r["image"], r["concept_name"], r["class_name"]


def load(d):
    rec = {key(r): {"gt": float(r["gt_delta_p"])} for r in read(d / "records.csv")}
    for col, files in (("cards", sorted(d.glob("cards_*.csv"))), ("eac", sorted(d.glob("eac_*.csv"))),
                       ("tcav_sign", [d / "tcav.csv"]), ("tcav_dd", [d / "tcav.csv"]),
                       ("tcav4_dd", [d / "tcav.csv"]), ("sae", [d / "sae.csv"])):
        files = [f for f in files if f.exists()]
        if not files:
            print(f"missing {col} files in {d}"); continue
        for f in files:
            for r in read(f):
                if key(r) in rec and r[col] != "":
                    rec[key(r)][col] = float(r[col])
    if any(d.glob("eac_*.csv")):  # EAC not run (e.g. the full class pools of the paper figures): leave it out
        for k, v in rec.items():
            if k[0] == "absent":
                v["eac"] = 0.0
    return rec


def hit_rates(present, score, n, rng):
    a = -np.abs(score); top = bot = 0.0
    for _ in range(N_TIE):
        o = np.lexsort((rng.random(len(a)), a))
        top += present[o[:n]].mean(); bot += (~present[o[-n:]]).mean()
    return top / N_TIE, bot / N_TIE


def srho(x, y):
    return spearmanr(x, y)[0] if len(x) > 2 and np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("COCO_LOCAL_OUT", "results/coco_stuff_local/resnet18"))
    rec = load(d)
    global METHODS
    METHODS = [m for m in METHODS if any(m in v for v in rec.values())]  # methods that were run
    print(f"methods: {', '.join(METHODS)}")
    full = {k: v for k, v in rec.items() if all(m in v for m in METHODS)}
    no_eac = sum(1 for k, v in rec.items() if k[0] == "present" and "eac" not in v)
    print(f"{len(rec)} records, {len(full)} with every method ({sum(1 for k in rec if k[0] == 'present')} present"
          + (f", of which {no_eac} have no EAC region)" if "eac" in METHODS else ")"))
    with open(d / "local_merged.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["set", "image", "concept_name", "class_name", "gt", *METHODS])
        for k, v in sorted(full.items()):
            w.writerow([*k, v["gt"], *[v[m] for m in METHODS]])
    pairs = defaultdict(list)
    for k, v in full.items():
        pairs[k[2], k[3]].append((k, v))
    pairs = {p: g for p, g in pairs.items() if 0 < sum(k[0] == "present" for k, _ in g) < len(g)}
    pres = [(k, v) for k, v in full.items() if k[0] == "present"]
    img = np.array([k[1] for k, _ in pres]); y = np.array([v["gt"] for _, v in pres])
    chance = np.mean([np.mean([k[0] == "present" for k, _ in g]) for g in pairs.values()])
    print(f"{len(pres)} present records, {len(pairs)} (concept, class) pairs; chance: top {chance:.2f} present, "
          f"bottom {1 - chance:.2f} absent, AP {chance:.2f}\n")
    rng = np.random.default_rng(0); out, per_pair = [], []
    ims = np.unique(img); at = {i: np.where(img == i)[0] for i in ims}
    for m in METHODS:
        x = np.array([v[m] for _, v in pres])
        rp = srho(x, y)
        bp = [srho(x[s], y[s]) for s in (np.concatenate([at[i] for i in rng.choice(ims, len(ims))]) for _ in range(N_BOOT))]
        rows = []
        for (c, k), g in sorted(pairs.items()):
            p = np.array([kk[0] == "present" for kk, _ in g]); s = np.array([v[m] for _, v in g])
            gy = np.array([v["gt"] for _, v in g])
            row = {"method": LABELS[m], "concept_name": c, "class_name": k, "n_present": int(p.sum()),
                   "rho_within": srho(s[p], gy[p]), "auroc": roc_auc_score(p, np.abs(s)),
                   "ap": average_precision_score(p, np.abs(s))}
            for n in NS:
                row[f"hit_top{n}"], row[f"hit_bottom{n}"] = hit_rates(p, s, n, rng)
            rows.append(row)
        per_pair += rows
        rw = np.array([r["rho_within"] for r in rows])
        bw = [np.nanmean(rng.choice(rw, len(rw))) for _ in range(N_BOOT)]
        row = {"method": LABELS[m], "n_present": len(pres), "n_pairs": len(rows),
               "rho_pooled": rp, "rho_pooled_lo": np.nanpercentile(bp, 2.5), "rho_pooled_hi": np.nanpercentile(bp, 97.5),
               "rho_within": np.nanmean(rw), "rho_within_lo": np.nanpercentile(bw, 2.5),
               "rho_within_hi": np.nanpercentile(bw, 97.5), "n_pairs_rho_within": int(np.isfinite(rw).sum())}
        for q in ["auroc", "ap"] + [f"hit_{t}{n}" for n in NS for t in ("top", "bottom")]:
            row[q] = float(np.mean([r[q] for r in rows]))
        row["chance_top"] = chance; out.append(row)
        print(f" {LABELS[m]:26s} rho pooled {rp:+.3f} [{row['rho_pooled_lo']:+.2f},{row['rho_pooled_hi']:+.2f}]  "
              f"within {row['rho_within']:+.3f} [{row['rho_within_lo']:+.2f},{row['rho_within_hi']:+.2f}]  "
              f"AUROC {row['auroc']:.3f}  AP {row['ap']:.3f}  "
              + "  ".join(f"hit@{n} {row[f'hit_top{n}']:.2f}/{row[f'hit_bottom{n}']:.2f}" for n in NS))
    for name, rows in (("local_scores.csv", out), ("local_pair_scores.csv", per_pair)):
        with open(d / name, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote local_merged.csv, local_scores.csv, local_pair_scores.csv in {d}")


if __name__ == "__main__":
    main()
