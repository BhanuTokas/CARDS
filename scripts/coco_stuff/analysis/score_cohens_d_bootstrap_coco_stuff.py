"""COCO-Stuff: CARDS with and without Cohen's d, plus bootstrap CIs for every method.

Mirrors the CelebA scorers (`score_masking_hybrid_cohens_d*.py`, `score_paper_gt.py`).

Ground-truth images: a seeded random draw of 100 per concept (gt_sampling.py, env GT_SEED / GT_N_IMAGES), not the
first 100 by path that compare_all_methods_faithfulness.py uses, so numbers differ slightly from that script.

Cohen's d here is the one-sample d_z used on CelebA: for each (concept, class) pair,
    d_z = mean(delta_logit) / std(delta_logit, ddof=1)
over the K retrieved images, instead of the raw mean delta_logit that the headline
CARDS score uses. It is computed from `alpha_ablation_per_image_scores.csv`, so no
GPU or re-run is needed.

For each ground-truth strategy and each alpha it reports, for raw and d_z:
  pooled_rho   single Spearman over all (concept, class) pairs (the paper-table metric)
  class_rho    mean over classes of the per-class Spearman
  95% CI       concept-cluster bootstrap (resample concepts with replacement, keep all
               20 class pairs of a drawn concept together), N_BOOT resamples.
TCAV and the PCBMs (and any `EXTRA` methods) get the same bootstrap, so every number in
the comparison table carries a CI.

Usage (from the repo root):
    python scripts/coco_stuff/analysis/score_cohens_d_bootstrap_coco_stuff.py
    EXTRA="ConceptSAE|results/coco_stuff/baselines/sae_scores.csv|score;..." python ...
Outputs:
    results/coco_stuff/cohens_d_bootstrap_summary.csv
    results/coco_stuff/cohens_d_scores.csv   (alpha, concept, class, raw_score, d_z, n)
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from gt_sampling import load_gt_random  # noqa: E402
from compare_all_methods_faithfulness import (  # noqa: E402
    FAITH_CSV,
    MIN_IMAGES,
    MIN_SAMPLES,
    OTHER_METHODS,
    RESULTS_DIR,
    STRATEGIES,
    class_rho,
    load_method_scores,
    pooled_rho,
)

PER_IMAGE_CSV = RESULTS_DIR / "alpha_ablation_per_image_scores.csv"
OUT_SUMMARY = RESULTS_DIR / "cohens_d_bootstrap_summary.csv"
OUT_SCORES = RESULTS_DIR / "cohens_d_scores.csv"
N_BOOT = int(os.environ.get("N_BOOT", "1000"))
SEED = 0


def per_pair_scores() -> pd.DataFrame:
    """(alpha, concept, class) -> raw mean delta_logit, d_z, n from the per-image CSV."""
    df = pd.read_csv(PER_IMAGE_CSV, usecols=["alpha", "concept_name", "class_name", "delta_logit"])
    g = df.groupby(["alpha", "concept_name", "class_name"])["delta_logit"]
    out = g.agg(raw_score="mean", sd="std", n="count").reset_index()
    ok = out["sd"].notna() & (out["sd"] > 0) & (out["n"] >= MIN_SAMPLES)
    out["d_z"] = np.where(ok, out["raw_score"] / out["sd"].where(ok, 1.0), np.nan)
    return out.drop(columns="sd")


def pair_arrays(scores, faith, image_counts):
    """Aligned (score, ground truth, concept index) arrays over valid pairs."""
    valid = {c for c, n in image_counts.items() if n >= MIN_IMAGES}
    s, g, cid, cls = [], [], [], []
    concepts = sorted(valid)
    cidx = {c: i for i, c in enumerate(concepts)}
    for (c, cl), dp in faith.items():
        if c not in valid or (c, cl) not in scores or len(dp) < MIN_SAMPLES:
            continue
        v = scores[(c, cl)]
        if not np.isfinite(v):
            continue
        s.append(v); g.append(float(np.mean(dp))); cid.append(cidx[c]); cls.append(cl)
    return np.array(s), np.array(g), np.array(cid), np.array(cls), len(concepts)


def bootstrap_pooled(s, g, cid, n_concepts, rng) -> tuple[float, float]:
    by_c = [np.where(cid == i)[0] for i in range(n_concepts)]
    rhos = []
    for _ in range(N_BOOT):
        draw = rng.integers(0, n_concepts, n_concepts)
        idx = np.concatenate([by_c[i] for i in draw if len(by_c[i])])
        if len(idx) > 2:
            rhos.append(spearmanr(s[idx], g[idx])[0])
    lo, hi = np.nanpercentile(rhos, [2.5, 97.5])
    return float(lo), float(hi)


def extra_methods() -> list[tuple[str, dict]]:
    out = []
    for spec in filter(None, os.environ.get("EXTRA", "").split(";")):
        label, path, col = spec.split("|")
        out.append((label, load_method_scores(Path(path), col)))
    return out


def main() -> None:
    print("Computing raw / Cohen's d per pair from per-image deltas ...", flush=True)
    pp = per_pair_scores()
    pp.to_csv(OUT_SCORES, index=False)
    by_alpha = {}
    for a, sub in pp.groupby("alpha"):
        by_alpha[float(a)] = {
            "raw": {(r.concept_name, r.class_name): r.raw_score for r in sub.itertuples()},
            "d_z": {(r.concept_name, r.class_name): r.d_z for r in sub.itertuples()},
        }
    others = [(n, load_method_scores(p, c)) for n, p, c in OTHER_METHODS if p.exists()] + extra_methods()

    gts, info = load_gt_random({"coco_stuff": FAITH_CSV}, tuple(STRATEGIES))
    print(f"ground truth: random {info['n_images']} images/concept, seed {info['seed']}, {info['n_concepts']} concepts", flush=True)
    rows = []
    for strategy in STRATEGIES:
        faith, image_counts = gts[strategy]["coco_stuff"]
        print(f"\n=== {strategy} ===", flush=True)
        entries = []
        for alpha in sorted(by_alpha):
            for kind in ("raw", "d_z"):
                entries.append((f"CARDS {kind}", alpha, by_alpha[alpha][kind]))
        entries += [(name, None, sc) for name, sc in others]
        for label, alpha, sc in entries:
            s, g, cid, _, n_c = pair_arrays(sc, faith, image_counts)
            if len(s) < 3:
                continue
            rng = np.random.default_rng(SEED)
            p_rho = float(spearmanr(s, g)[0])
            lo, hi = bootstrap_pooled(s, g, cid, n_c, rng)
            c_rho, n_cls = class_rho(sc, faith, image_counts)
            rows.append({
                "strategy": strategy, "method": label, "alpha": "" if alpha is None else alpha,
                "pooled_rho": p_rho, "ci_lo": lo, "ci_hi": hi,
                "class_rho": c_rho, "n_classes": n_cls, "n_pairs": len(s),
            })
            tag = f" a={alpha}" if alpha is not None else ""
            print(f"  {label + tag:<22s} pooled {p_rho:+.4f} [{lo:+.3f},{hi:+.3f}]  "
                  f"class_rho {c_rho:+.4f}  n={len(s)}", flush=True)

    with open(OUT_SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {OUT_SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
