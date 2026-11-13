"""Step 1b: patch count (L) x z-score alpha sweep for HnS's pseudo-masks on FunnyBirds, TRAINING images only.

Step 1 (localization_check.py) found that SigLIP finds the parts (AUROC .75-.93, above the mismatched-query
control) but the alpha=1 cutoff marks ~15% of the image while parts cover 0.2-1.5%. L is changed as in the CelebA
patch-count ablation (scripts/celeba/ablate/..._patch_count_official_train.py): the published B/16 SigLIP
checkpoints at 224 px (L=196) and 512 px (L=1024). Queries fixed at the step-1 default (demean + orthogonalize).
Cutoffs from the same N_CAL calibration images as step 1; metrics on the same evaluation images.

Selecting (L, alpha) here uses the training images' part masks, which HnS itself never sees; report that.
Run from the repo root:  python scripts/funnybirds/localization_sweep.py [n_eval]
Writes $FUNNYBIRDS_OUT/localization_sweep.csv.
"""
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, "src")
from cards.attribution.localization import concept_zscore_cutoff, threshold_mask
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.pipeline import instantiate_encoder, orthogonalize_queries

sys.path.insert(0, str(Path(__file__).parent))
from localization_check import DEVICE, N_CAL, SEED, auroc, load_rgb, part_masks, sim_maps
from paths import OUT, PARTS, QUERY_TEXT, ROOT

ENCODERS = {196: "ViT-B-16-SigLIP", 1024: "ViT-B-16-SigLIP-512"}
ALPHAS = [1.0, 1.5, 2.0, 2.5, 3.0]


def main():
    n_eval = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    params = json.loads((ROOT / "dataset_train.json").read_text())
    idx = list(range(len(params)))
    random.Random(SEED).shuffle(idx)
    cal, ev = idx[:N_CAL], idx[N_CAL:N_CAL + n_eval]

    def load(i):
        c = str(params[i]["class_idx"])
        return load_rgb(ROOT / "train" / c / f"{i:06d}.png"), part_masks(ROOT / "train_part_map" / c / f"{i:06d}.png")

    rows = []
    for L, name in ENCODERS.items():
        cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": {
            "name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
            "model_name": name, "pretrained": "webli", "device": DEVICE}})
        enc = instantiate_encoder(cfg)
        center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, enc)
        q = orthogonalize_queries({p: demean_query(build_concept_query(QUERY_TEXT[p], enc), center) for p in PARTS})
        q = {p: v.to(DEVICE) for p, v in q.items()}
        cal_maps = {p: [] for p in PARTS}
        for i in cal:
            img, masks = load(i)
            sm = sim_maps(enc, img, q)
            for p in PARTS:
                if masks[p].any():
                    cal_maps[p].append(sm[p])
        cut = {(a, p): concept_zscore_cutoff(cal_maps[p], a) for a in ALPHAS for p in PARTS}
        acc = {(a, p): {m: [] for m in ("auroc", "mismatch", "area", "true_area", "precision", "recall", "iou",
                                        "covers_other")} for a in ALPHAS for p in PARTS}
        for n, i in enumerate(ev):
            img, masks = load(i)
            sm = sim_maps(enc, img, q)
            for p in PARTS:
                g = masks[p]
                if not g.any():
                    continue
                au = auroc(sm[p], g)
                mism = float(np.nanmean([auroc(sm[o], g) for o in PARTS if o != p]))
                others = np.zeros_like(g)
                for o in PARTS:
                    if o != p:
                        others |= masks[o]
                for a in ALPHAS:
                    m = threshold_mask(sm[p], method="fixed", cutoff=cut[(a, p)])
                    inter = float((m & g).sum()); r = acc[(a, p)]
                    r["auroc"].append(au); r["mismatch"].append(mism)
                    r["area"].append(float(m.mean())); r["true_area"].append(float(g.mean()))
                    r["precision"].append(inter / m.sum() if m.any() else 0.0)
                    r["recall"].append(inter / g.sum())
                    r["iou"].append(inter / float((m | g).sum()))
                    # share of the OTHER parts' pixels this part's mask also removes (what makes PIs indistinct)
                    r["covers_other"].append(float((m & others).sum() / others.sum()) if others.any() else np.nan)
            if (n + 1) % 50 == 0:
                print(f"  L={L}: {n + 1}/{len(ev)} evaluation images", flush=True)
        for (a, p), r in acc.items():
            rows.append({"L": L, "encoder": name, "alpha": a, "part": p, "n": len(r["auroc"]), "cutoff": cut[(a, p)]}
                        | {m: float(np.nanmean(v)) for m, v in r.items()})
        del enc

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "localization_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{'L':>5} {'alpha':>5} {'part':>5} {'auroc':>6} {'mism':>6} {'area':>6} {'true':>6} {'prec':>6} "
          f"{'rec':>6} {'iou':>6} {'other':>6}")
    for r in rows:
        print(f"{r['L']:>5} {r['alpha']:5.1f} {r['part']:>5} {r['auroc']:6.3f} {r['mismatch']:6.3f} {r['area']:6.3f} "
              f"{r['true_area']:6.3f} {r['precision']:6.3f} {r['recall']:6.3f} {r['iou']:6.3f} "
              f"{r['covers_other']:6.3f}")
    print("\nmean over parts:")
    for L in ENCODERS:
        for a in ALPHAS:
            s = [r for r in rows if r["L"] == L and r["alpha"] == a]
            print(f"  L={L:>4} alpha={a:.1f}  iou {np.mean([r['iou'] for r in s]):.3f}  "
                  f"prec {np.mean([r['precision'] for r in s]):.3f}  rec {np.mean([r['recall'] for r in s]):.3f}  "
                  f"area {np.mean([r['area'] for r in s]):.3f}  covers_other {np.mean([r['covers_other'] for r in s]):.3f}")


if __name__ == "__main__":
    main()
