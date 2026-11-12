"""Raw vs Cohen's d (d_z) for (a) the patch-count ablation and (b) the main CARDS rows of all three black boxes.

Inputs (per-image deltas written by the *_dz reruns):
  results/coco_stuff/patch_count_per_image_dz.csv            L, concept_name, class_name, image_idx, delta_logit
  results/coco_stuff/model_variants/{backbone}_cards_per_image_dz.csv   concept_name, image_idx, class_name, delta_logit
raw = mean(delta_logit) per (concept, class); d_z = mean / std(ddof=1) (pairs with std 0 or n < MIN_SAMPLES dropped).
Both are scored from the SAME per-image file, so raw vs d_z is a like-for-like comparison. Ground truth: seeded random
100 images per concept (gt_sampling.py), seeds SEEDS (default 0-4); concept-cluster bootstrap CI for the first seed.
Output: results/coco_stuff/dz_patch_and_main_summary.csv   Run from the repo root."""

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

from compare_all_methods_faithfulness import FAITH_CSV, MIN_SAMPLES, RESULTS_DIR, STRATEGIES  # noqa: E402
from gt_sampling import load_gt_random  # noqa: E402
from score_cohens_d_bootstrap_coco_stuff import bootstrap_pooled, pair_arrays  # noqa: E402

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
OUT = RESULTS_DIR / "dz_patch_and_main_summary.csv"
BACKBONES = {"coco_stuff": "ResNet18", "coco_stuff_vit_b_16": "ViT-B/16", "coco_stuff_convnext_tiny": "ConvNeXt-T"}


def raw_and_dz(df: pd.DataFrame, group_cols: list[str]) -> dict:
    """{group key: {"raw": {(concept, class): v}, "d_z": {...}}}"""
    out = {}
    for key, g in df.groupby(group_cols, sort=True) if group_cols else [((), df)]:
        a = g.groupby(["concept_name", "class_name"])["delta_logit"].agg(raw="mean", sd="std", n="count")
        ok = a.sd.notna() & (a.sd > 0) & (a.n >= MIN_SAMPLES)
        a["d_z"] = np.where(ok, a.raw / a.sd.where(ok, 1.0), np.nan)
        out[key if isinstance(key, tuple) else (key,)] = {k: a[k].to_dict() for k in ("raw", "d_z")}
    return out


def main() -> None:
    entries = []  # (label, kind, score dict)
    pc = RESULTS_DIR / "patch_count_per_image_dz.csv"
    if pc.exists():
        for (L,), d in raw_and_dz(pd.read_csv(pc), ["L"]).items():
            for kind in ("raw", "d_z"):
                entries.append((f"patch L={L}", kind, d[kind]))
    for bb, name in BACKBONES.items():
        f = RESULTS_DIR / "model_variants" / f"{bb}_cards_per_image_dz.csv"
        if f.exists():
            (d,) = raw_and_dz(pd.read_csv(f), []).values()
            for kind in ("raw", "d_z"):
                entries.append((f"CARDS {name}", kind, d[kind]))
    rows = []
    for seed in SEEDS:
        gts, _ = load_gt_random({"coco_stuff": FAITH_CSV}, tuple(STRATEGIES), seed=seed)
        for strategy in STRATEGIES:
            faith, counts = gts[strategy]["coco_stuff"]
            for label, kind, sc in entries:
                s, g, cid, _, n_c = pair_arrays(sc, faith, counts)
                rho = float(spearmanr(s, g)[0]); lo = hi = float("nan")
                if seed == SEEDS[0]:
                    lo, hi = bootstrap_pooled(s, g, cid, n_c, np.random.default_rng(0))
                rows.append({"seed": seed, "strategy": strategy, "row": label, "score": kind, "pooled_rho": rho,
                             "ci_lo": lo, "ci_hi": hi, "n_pairs": len(s)})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    t = pd.DataFrame(rows)
    m = t.groupby(["strategy", "row", "score"]).pooled_rho.agg(["mean", "std"]).round(3).unstack("score")
    print(m.to_string()); print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
