"""Step 1 of the FunnyBirds experiment: does HnS's SigLIP localization find the five bird parts?

Feasibility gate before running HnS on the benchmark. Uses TRAINING images only (the 500 test images are kept
for the actual evaluation). For each query configuration (demean_query on/off x orthogonalize on/off) and part:
  - auroc:       pixel AUROC of the similarity map against the true part mask (part map colours), per image, mean
  - pointing:    fraction of images whose most similar pixel lies on the part
  - mismatch:    mean AUROC of the OTHER four parts' queries against this part's mask (control: a high value means
                 the map finds "the bird" or a fixed position, not the part)
  - pseudo-mask: HnS's thresholded mask (per-part z-score cutoff, alpha=1.0, from N_CAL calibration images, as in
                 the CelebA runs): area fraction vs the true part's, precision, recall, IoU
Only images where the part is visible count for that part.

Run from the repo root:  python scripts/funnybirds/localization_check.py [n_eval]
Writes $FUNNYBIRDS_OUT/localization_check.csv.
"""
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from scipy.stats import rankdata

sys.path.insert(0, "src")
import cards
from cards.attribution.localization import concept_zscore_cutoff, threshold_mask
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.pipeline import instantiate_encoder, orthogonalize_queries

sys.path.insert(0, str(Path(__file__).parent))
from paths import OUT, PART_COLORS, PARTS, QUERY_TEXT, ROOT

SEED, ALPHA, N_CAL = 0, 1.0, 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIGS = [(True, True), (True, False), (False, True), (False, False)]  # (demean, orthogonalize); HnS default first


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def part_masks(part_map_path: Path) -> dict[str, np.ndarray]:
    pm = np.asarray(Image.open(part_map_path).convert("RGB")).astype(np.int32)
    out = {}
    for part, cols in PART_COLORS.items():
        m = np.zeros(pm.shape[:2], bool)
        for c in cols:
            m |= np.all(pm == np.array(c), axis=-1)
        out[part] = m
    return out


@torch.no_grad()
def sim_maps(enc, image: Image.Image, queries: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Same computation as cards.attribution.localization.localize_concept, with the patches encoded once."""
    patches = enc.encode_patches(image)
    side = round(patches.shape[0] ** 0.5)
    out = {}
    for c, q in queries.items():
        sims = (patches @ q.to(patches.device)).float().cpu()
        up = F.interpolate(sims.reshape(1, 1, side, side), size=(image.height, image.width), mode="bilinear",
                           align_corners=False)
        out[c] = up[0, 0].numpy()
    return out


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    n1 = int(labels.sum()); n0 = labels.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores.ravel())
    return float((r[labels.ravel()].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    n_eval = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print("cards from", cards.__file__, flush=True)
    params = json.loads((ROOT / "dataset_train.json").read_text())
    idx = list(range(len(params)))
    random.Random(SEED).shuffle(idx)
    cal, ev = idx[:N_CAL], idx[N_CAL:N_CAL + n_eval]

    def paths(i):
        c = str(params[i]["class_idx"])
        return ROOT / "train" / c / f"{i:06d}.png", ROOT / "train_part_map" / c / f"{i:06d}.png"

    cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": {
        "name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
        "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE}})
    enc = instantiate_encoder(cfg)
    center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, enc)
    raw = {p: build_concept_query(QUERY_TEXT[p], enc) for p in PARTS}
    qsets = {}
    for dm, orth in CONFIGS:
        q = {p: demean_query(v, center) if dm else v for p, v in raw.items()}
        if orth:
            q = orthogonalize_queries(q)
        qsets[(dm, orth)] = {p: v.to(DEVICE) for p, v in q.items()}

    def maps_for(i):
        img_p, pm_p = paths(i)
        img = load_rgb(img_p)
        return {k: sim_maps(enc, img, q) for k, q in qsets.items()}, part_masks(pm_p)

    # check the shared-patch shortcut against the library function once
    from cards.attribution.localization import localize_concept
    img0 = load_rgb(paths(cal[0])[0]); k0 = CONFIGS[0]
    ref = localize_concept(enc, img0, qsets[k0]["beak"], (img0.height, img0.width))
    assert np.allclose(ref, sim_maps(enc, img0, qsets[k0])["beak"], atol=1e-5)

    cut = {k: {p: [] for p in PARTS} for k in qsets}
    for n, i in enumerate(cal):
        maps, masks = maps_for(i)
        for k in qsets:
            for p in PARTS:
                if masks[p].any():
                    cut[k][p].append(maps[k][p])
    cut = {k: {p: concept_zscore_cutoff(v, ALPHA) for p, v in d.items()} for k, d in cut.items()}
    print(f"cutoffs from {N_CAL} calibration images done", flush=True)

    acc = {(k, p): {m: [] for m in ("auroc", "pointing", "mismatch", "area", "true_area", "precision", "recall",
                                    "iou")} for k in qsets for p in PARTS}
    for n, i in enumerate(ev):
        maps, masks = maps_for(i)
        for k in qsets:
            for p in PARTS:
                g = masks[p]
                if not g.any():
                    continue
                s = maps[k][p]; a = acc[(k, p)]
                a["auroc"].append(auroc(s, g))
                a["pointing"].append(float(g.flat[int(np.argmax(s))]))
                a["mismatch"].append(float(np.nanmean([auroc(maps[k][o], g) for o in PARTS if o != p])))
                m = threshold_mask(s, method="fixed", cutoff=cut[k][p])
                inter = float((m & g).sum())
                a["area"].append(float(m.mean())); a["true_area"].append(float(g.mean()))
                a["precision"].append(inter / m.sum() if m.any() else 0.0)
                a["recall"].append(inter / g.sum())
                a["iou"].append(inter / float((m | g).sum()))
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(ev)} evaluation images", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for (k, p), a in acc.items():
        rows.append({"demean": k[0], "orthogonalize": k[1], "part": p, "n": len(a["auroc"]), "cutoff": cut[k][p]}
                    | {m: float(np.nanmean(v)) for m, v in a.items()})
    with open(OUT / "localization_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{'demean':>6} {'orth':>5} {'part':>5} {'n':>4} {'auroc':>6} {'point':>6} {'mism':>6} {'area':>6} "
          f"{'true':>6} {'prec':>6} {'rec':>6} {'iou':>6}")
    for r in rows:
        print(f"{r['demean']!s:>6} {r['orthogonalize']!s:>5} {r['part']:>5} {r['n']:>4} {r['auroc']:6.3f} "
              f"{r['pointing']:6.3f} {r['mismatch']:6.3f} {r['area']:6.3f} {r['true_area']:6.3f} "
              f"{r['precision']:6.3f} {r['recall']:6.3f} {r['iou']:6.3f}")


if __name__ == "__main__":
    main()
