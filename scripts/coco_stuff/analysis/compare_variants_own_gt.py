"""All-methods comparison across COCO-Stuff black boxes (ResNet18, ViT-B/16, ConvNeXt-Tiny), each scored against ITS OWN
masking ground truth, next to the ResNet18 ground truth the variants were scored against so far.

compare_all_methods_faithfulness.py only covers ResNet18. The variant run scripts evaluate ViT / ConvNeXt attributions
against coco_stuff_faithfulness.csv, which is ResNet18's masking effect. Build the matching ground truths with
scripts/coco_stuff/build/compute_coco_stuff_faithfulness_variants.py (files coco_stuff_faithfulness_{backbone}.csv, or
_shard{i}of{n}.csv pieces, which are merged here), then run this.

Methods per backbone, from results/coco_stuff/model_variants/{backbone}_*: CARDS (cards_scores.csv, raw_score),
TCAV (tcav_scores.csv, mean_sign_count), PCBM (pcbm_concept_weights.csv, weight). env EXTRA adds methods:
    EXTRA="coco_stuff_vit_b_16:PCBM-SigLIP|path.csv|weight;..."   (backbone:label|csv|column)
Ground truth: seeded random 100 images per concept, concepts intersected over the available black boxes, the
SAME images for every model (gt_sampling.py).
Metrics: pooled rho (single Spearman over all (concept, class) pairs, the paper-table metric) with a concept-cluster
bootstrap 95% CI, and mean class rho. Output: results/coco_stuff/variants_own_gt_summary.csv
Run from the repo root (results/ relative)."""

from __future__ import annotations

import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from compare_all_methods_faithfulness import FAITH_CSV, RESULTS_DIR, STRATEGIES, class_rho, load_method_scores  # noqa: E402
from gt_sampling import load_gt_random  # noqa: E402
from score_cohens_d_bootstrap_coco_stuff import SEED, bootstrap_pooled, pair_arrays  # noqa: E402

BACKBONES = ["coco_stuff", "coco_stuff_vit_b_16", "coco_stuff_convnext_tiny"]
VARIANT_DIR = RESULTS_DIR / "model_variants"
OUT = RESULTS_DIR / "variants_own_gt_summary.csv"
METHODS = [("CARDS", "cards_scores.csv", "raw_score"), ("TCAV", "tcav_scores.csv", "mean_sign_count"),
           ("PCBM", "pcbm_concept_weights.csv", "weight")]


def own_gt_path(bb: str) -> Path | None:
    if bb == "coco_stuff":
        return FAITH_CSV
    merged = RESULTS_DIR / f"coco_stuff_faithfulness_{bb}.csv"
    shards = sorted(glob.glob(str(RESULTS_DIR / f"coco_stuff_faithfulness_{bb}_shard*of*.csv")))
    if shards:  # (re)merge, header once
        with open(merged, "w", newline="") as out:
            for i, s in enumerate(shards):
                with open(s) as f:
                    lines = f.readlines()
                out.writelines(lines if i == 0 else lines[1:])
    return merged if merged.exists() else None


def extra() -> dict[str, list[tuple[str, dict]]]:
    out: dict[str, list] = {}
    for spec in filter(None, os.environ.get("EXTRA", "").split(";")):
        bb, rest = spec.split(":", 1); label, path, col = rest.split("|")
        out.setdefault(bb, []).append((label, load_method_scores(Path(path), col)))
    return out


def main() -> None:
    ex = extra(); rows = []
    paths = {bb: p for bb in BACKBONES if (p := own_gt_path(bb)) is not None}
    for bb in BACKBONES:
        if bb not in paths:
            print(f"[{bb}] no own ground truth yet; skipped", flush=True)
    gts, info = load_gt_random(paths, tuple(STRATEGIES))
    print(f"ground truth: random {info['n_images']} images/concept, seed {info['seed']}, {info['n_concepts']} concepts "
          f"(intersection over {list(paths)}); same images for every model", flush=True)
    for bb, dropped in info["dropped_concepts"].items():
        print(f"  dropped for {bb}: {len(dropped)} concepts", flush=True)
    for strategy in STRATEGIES:
        for bb in paths:
            gt_cols = [("own_gt", gts[strategy][bb])]
            if bb != "coco_stuff":  # the cross-model setup used so far: this model's attributions vs ResNet18 masking
                gt_cols.append(("resnet18_gt", gts[strategy]["coco_stuff"]))
            methods = [(n, load_method_scores(VARIANT_DIR / f"{bb}_{f}", c)) for n, f, c in METHODS
                       if (VARIANT_DIR / f"{bb}_{f}").exists()] + ex.get(bb, [])
            print(f"\n=== {bb} / {strategy} ===", flush=True)
            for gt_name, (faith, counts) in gt_cols:
                for label, sc in methods:
                    s, g, cid, _, n_c = pair_arrays(sc, faith, counts)
                    if len(s) < 3:
                        continue
                    rho = float(spearmanr(s, g)[0]); lo, hi = bootstrap_pooled(s, g, cid, n_c, np.random.default_rng(SEED))
                    c_rho, _ = class_rho(sc, faith, counts)
                    rows.append({"backbone": bb, "strategy": strategy, "gt": gt_name, "method": label,
                                 "pooled_rho": rho, "ci_lo": lo, "ci_hi": hi, "class_rho": c_rho, "n_pairs": len(s)})
                    print(f"  {gt_name:<12s} {label:<14s} pooled {rho:+.4f} [{lo:+.3f},{hi:+.3f}]  class_rho {c_rho:+.4f}  n={len(s)}",
                          flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
