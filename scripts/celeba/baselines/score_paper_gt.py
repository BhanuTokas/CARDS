# ruff: noqa: SIM115
# One-shot research script: bare open() on small CSVs is intentional.

"""Score every method against the paper's CelebA ground truth (raw blur: mean delta_p per task and concept, all 26
concepts), for the black box chosen by env CARDS_ARCH (resnet18 default, vit, convnext). Both ground-truth image sets
are scored: previously_used (the paper's main table) and non_overlapping (images outside the classifier's training set).
Pooled rho = mean of the two per-task Spearman rhos; 95% CI from 2000 concept bootstrap resamples.

Methods:
  resnet18: CARDS, CARDS (Cohen's d), TCAV, PCBM conventional and both CLIP-concept PCBMs, via
            load_methods() of scripts/celeba/analysis/score_methods_against_strict_control_official_train.py
            (needs post_hoc_cbm on PYTHONPATH for the PCBM checkpoints).
  vit / convnext: CARDS and the two CLIP-concept PCBMs from main's per-model result files, where present.
  env EXTRA = "label|csv|column;..." adds methods from CSVs with (task or target_task), concept_name columns, e.g. the
  outputs of this folder's baselines.
Inputs are read from $CARDS_RESULTS_DIR (default results). argv[1] = output CSV. Run from the repo root."""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from black_boxes import ARCH, SUFFIX

R = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
TASKS = ("Attractive", "Male")
N_BOOT = 2000


def read_scores(path, column):
    out = {}
    for r in csv.DictReader(open(path)):
        out[(r.get("task") or r["target_task"], r["concept_name"])] = float(r[column])
    return out


def methods():
    if ARCH == "resnet18":
        sys.path.insert(0, "scripts/celeba/analysis")
        from score_methods_against_strict_control_official_train import load_methods
        m = dict(load_methods())
    else:
        m = {"CARDS": read_scores(R / f"cards_celeba_masking_hybrid_official_train{SUFFIX}_raw_scores.csv", "hybrid_raw_score")}
        for label, stem in (("PCBM CLIP-SigLIP", "siglip"), ("PCBM CLIP-RN50", "clip_rn50")):
            f = R / f"pcbm_clip_concepts_celeba_official_train{SUFFIX}_{stem}_weights.csv"
            if f.exists():
                m[label] = read_scores(f, "weight")
    for spec in filter(None, os.environ.get("EXTRA", "").split(";")):
        label, path, column = spec.split("|")
        m[label] = read_scores(path, column)
    return m


def main():
    out_path = sys.argv[1]
    M = methods(); rng = np.random.default_rng(0); out = []
    print("black box:", ARCH)
    for gv in ("previously_used", "non_overlapping"):
        g = defaultdict(list)
        for r in csv.DictReader(open(R / f"celeba_official_train{SUFFIX}_faithfulness_{gv}.csv")):
            g[(r["target_task"], r["concept_name"])].append(float(r["delta_p"]))
        gt = {k: np.mean(v) for k, v in g.items()}
        print(f"\n=== {gv}")
        for m, s in M.items():
            arr, rho = {}, []
            for t in TASKS:
                cs = sorted(c for (tt, c) in gt if tt == t and (t, c) in s)
                arr[t] = (np.array([s[(t, c)] for c in cs]), np.array([gt[(t, c)] for c in cs]))
                rho.append((spearmanr(*arr[t])[0], len(cs)))
            boot = []
            for _ in range(N_BOOT):
                v = []
                for t in TASKS:
                    x, y = arr[t]; i = rng.integers(0, len(x), len(x)); v.append(spearmanr(x[i], y[i])[0])
                boot.append(np.mean(v))
            pool = np.mean([r[0] for r in rho]); lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            print(f" {m:32s} A {rho[0][0]:+.3f} (n={rho[0][1]})  M {rho[1][0]:+.3f} (n={rho[1][1]})  "
                  f"pool {pool:+.3f} [{lo:+.2f},{hi:+.2f}]")
            out.append((ARCH, gv, m, rho[0][0], rho[1][0], pool, lo, hi))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "gt_version", "method", "rho_A", "rho_M", "pooled", "ci_lo", "ci_hi"]); w.writerows(out)


if __name__ == "__main__":
    main()
