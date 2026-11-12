"""Rescore the patch-count (L) ablation on the seeded random-100 ground truth (gt_sampling.py) instead of the first-100.

Reads results/coco_stuff/patch_count_ablation_scores.csv; for each strategy and seed (SEEDS, default 0-4) reports pooled
rho per L, plus a concept-cluster bootstrap CI for seed 0. Output: results/coco_stuff/patch_count_random100_summary.csv
Run from the repo root."""

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

from compare_all_methods_faithfulness import FAITH_CSV, RESULTS_DIR, STRATEGIES  # noqa: E402
from gt_sampling import load_gt_random  # noqa: E402
from score_cohens_d_bootstrap_coco_stuff import bootstrap_pooled, pair_arrays  # noqa: E402

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
OUT = RESULTS_DIR / "patch_count_random100_summary.csv"


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "patch_count_ablation_scores.csv")
    scores = {L: {(r.concept_name, r.class_name): r.raw_score for r in g.itertuples()} for L, g in df.groupby("L")}
    rows = []
    for seed in SEEDS:
        gts, info = load_gt_random({"coco_stuff": FAITH_CSV}, tuple(STRATEGIES), seed=seed)
        for strategy in STRATEGIES:
            faith, counts = gts[strategy]["coco_stuff"]
            for L, sc in sorted(scores.items()):
                s, g, cid, _, n_c = pair_arrays(sc, faith, counts)
                rho = float(spearmanr(s, g)[0])
                lo = hi = float("nan")
                if seed == SEEDS[0]:
                    lo, hi = bootstrap_pooled(s, g, cid, n_c, np.random.default_rng(0))
                rows.append({"seed": seed, "strategy": strategy, "L": L, "pooled_rho": rho, "ci_lo": lo, "ci_hi": hi,
                             "n_pairs": len(s), "n_concepts": info["n_concepts"]})
                print(f"seed {seed} {strategy:<10s} L={L:<5d} pooled {rho:+.4f}  n={len(s)}", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    t = pd.DataFrame(rows)
    print("\nmean (sd) over seeds:")
    print(t.groupby(["strategy", "L"]).pooled_rho.agg(["mean", "std"]).round(3).to_string())
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
