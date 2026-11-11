"""Merge and score the local (per-image) attributions of the official-train ResNet18 on CelebA.

Inputs (CARDS and TCAV come from the existing local run; the rest from this folder's local_*.py, both record sets):
  results/local_attribution_celeba_official_train_pairs.csv          present records: gt_delta_p, hybrid_score, tcav_score
  results/local_attribution_celeba_official_train_absent_pairs.csv   absent records (ground truth 0)
  argv[1]/{soft,eac,sae}_{present,absent}_*.csv                      local_soft.py, local_eac.py, local_sae.py
Soft CARDS = D - rho[task, fill] * A per record, rho = sum(second-patch drop) / sum(first-patch drop) over all records
(no labels), as in soft_aggregate.py; records without a null (mask too large to place one) are dropped for every
method so all methods are scored on the same records.

Outputs in argv[1]:
  local_merged.csv  one row per (image, concept, task, set) with every method's score
  local_scores.csv  per method and task:
    rho      Spearman(score, ground-truth delta_p) over present records, 95% CI from 2000 image-cluster bootstraps
    hit@N    per (concept, task), rank present + absent records by |score|: share of the top N that are present and
             share of the bottom N that are absent; mean over (concept, task); N = 3 (as in the figures), 5, 10
    auroc    AUROC of |score| for present vs absent, mean over (concept, task)
  Chance levels (share of present records) are printed with the table."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from local_common import RESULTS, image_name

METHODS = ["cards", "soft", "tcav", "eac", "sae"]
LABELS = {"cards": "CARDS", "soft": "CARDS soft (D-rho*A)", "tcav": "TCAV", "eac": "EAC", "sae": "ConceptSAE"}
FILLS = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
NS, N_BOOT = (3, 5, 10), 2000


def read(path):
    return list(csv.DictReader(open(path, newline="")))


def num(x):
    return float(x) if x not in ("", None) else np.nan


def load(d):
    rec = {}
    for s, f in (("present", "local_attribution_celeba_official_train_pairs.csv"),
                 ("absent", "local_attribution_celeba_official_train_absent_pairs.csv")):
        for r in read(RESULTS / f):
            rec[(image_name(r["image"]), r["concept_name"], r["target_task"], s)] = {
                "gt": float(r["gt_delta_p"]), "cards": float(r["hybrid_score"]), "tcav": float(r["tcav_score"])}
    soft_rows = []
    for m, col in (("eac", "eac"), ("sae", "sae"), ("soft", None)):
        for s in ("present", "absent"):
            files = sorted(d.glob(f"{m}_{s}_*.csv"))
            if not files:
                print(f"missing {m}_{s}_*.csv in {d}"); continue
            for f in files:
                for r in read(f):
                    k = (r["image"], r["concept_name"], r["target_task"], s)
                    if k not in rec:
                        continue
                    if col:
                        rec[k][m] = float(r[col])
                    else:
                        soft_rows.append((k, r)); rec[k]["soft_D"] = float(r["cards"])
    rho = {}
    for t in ("Attractive", "Male"):
        for f in FILLS:
            a = np.array([(num(r[f"{f}_first"]), num(r[f"{f}_second"])) for k, r in soft_rows if k[2] == t])
            a = a[~np.isnan(a).any(1)] if len(a) else a
            rho[t, f] = a[:, 1].sum() / a[:, 0].sum() if len(a) else np.nan
    for k, r in soft_rows:
        if r["null"] != "":
            rec[k]["soft"] = float(r["cards"]) - rho[k[2], r["fill"]] * float(r["null"])
    print("two-patch rho per (task, fill):", {f"{t[0]}/{f}": round(v, 2) for (t, f), v in rho.items()})
    return rec


def boot_rho(x, y, img, rng):
    ims = np.unique(img); idx = {i: np.where(img == i)[0] for i in ims}; out = []
    for _ in range(N_BOOT):
        sel = np.concatenate([idx[i] for i in rng.choice(ims, len(ims))])
        out.append(spearmanr(x[sel], y[sel])[0])
    return np.nanpercentile(out, [2.5, 97.5])


def main():
    d = Path(sys.argv[1]); rec = load(d)
    full = {k: v for k, v in rec.items() if all(m in v for m in METHODS)}
    print(f"{len(rec)} records, {len(full)} with every method "
          f"({sum(1 for v in rec.values() if 'soft_D' in v and 'soft' not in v)} dropped: no null for soft CARDS)")
    with open(d / "local_merged.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "concept_name", "target_task", "set", "gt", *METHODS])
        for k, v in sorted(full.items()):
            w.writerow([*k, v["gt"], *[v[m] for m in METHODS]])
    rng = np.random.default_rng(0); out = []
    groups = defaultdict(list)
    for k, v in full.items():
        groups[k[1], k[2]].append((k[3] == "present", v))
    for t in ("Attractive", "Male"):
        pres = [(k, v) for k, v in full.items() if k[2] == t and k[3] == "present"]
        img = np.array([k[0] for k, _ in pres]); y = np.array([v["gt"] for _, v in pres])
        gs = [g for (c, tt), g in groups.items() if tt == t and 0 < sum(p for p, _ in g) < len(g)]
        if len(pres) < 3 or not gs:
            print(f"\n=== {t}: too few records to score"); continue
        chance = np.mean([np.mean([p for p, _ in g]) for g in gs])
        print(f"\n=== {t}: rho on {len(pres)} present records; hit rates over {len(gs)} concepts "
              f"(chance: top {chance:.2f} present, bottom {1 - chance:.2f} absent)")
        for m in METHODS:
            x = np.array([v[m] for _, v in pres]); r = spearmanr(x, y)[0]; lo, hi = boot_rho(x, y, img, rng)
            row = {"method": LABELS[m], "task": t, "n_present": len(pres), "rho": r, "rho_lo": lo, "rho_hi": hi}
            for n in NS:
                top, bot = [], []
                for g in gs:
                    order = sorted(g, key=lambda pv: -abs(pv[1][m]))
                    top.append(np.mean([p for p, _ in order[:n]])); bot.append(np.mean([not p for p, _ in order[-n:]]))
                row[f"hit_top{n}"], row[f"hit_bottom{n}"] = np.mean(top), np.mean(bot)
            row["auroc"] = np.mean([roc_auc_score([p for p, _ in g], [abs(v[m]) for _, v in g]) for g in gs])
            row["chance_top"] = chance; out.append(row)
            print(f" {LABELS[m]:22s} rho {r:+.3f} [{lo:+.2f},{hi:+.2f}]  "
                  + "  ".join(f"hit@{n} {row[f'hit_top{n}']:.2f}/{row[f'hit_bottom{n}']:.2f}" for n in NS)
                  + f"  AUROC {row['auroc']:.3f}")
    if not out:
        return
    with open(d / "local_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(out[0])); w.writeheader(); w.writerows(out)
    print(f"\nwrote {d / 'local_merged.csv'} and {d / 'local_scores.csv'}")


if __name__ == "__main__":
    main()
