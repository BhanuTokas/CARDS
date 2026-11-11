# ruff: noqa: SIM115
# One-shot research script: bare open() on small CSVs is intentional.

"""Local (per-image) CARDS and soft CARDS on the clean CelebA ground truth (official-train black box, env CARDS_ARCH).

CARDS local score: identical to scripts/celeba/analysis/local_attribution_comparison_celeba_official_train.py's
hybrid score (same SigLIP queries, z-score cutoff from 50 official-train positives drawn with the same seeded
shuffles, best-of-7 fill, per-pair fill seed): D = positive-class logit drop when the concept's pseudo-mask is filled.
Soft CARDS adds, per pair, the fill-matched null A (mean drop of 3 equal-area random smooth shapes outside the mask,
same fill) and the two-patch test per fill (first and second patch drops), exactly as celeba_soft.py does globally.
score_local.py turns these into soft = D - rho[task, fill] * A, rho estimated over all pairs (no labels).

argv[1] = output folder; writes soft_{SHARD}.csv. Sharded over (image, concept) pairs (local_common.py)."""
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

sys.path.insert(0, "scripts/celeba/run"); sys.path.insert(0, "src")
from run_cards_celeba_full import CONCEPT_QUERY_TEXT
from run_cards_celeba_masking_hybrid_official_val_zscore import CELEBA_ROOT

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.validation.broden_faithfulness import mask_region

sys.path.insert(0, str(Path(__file__).parent))
from black_boxes import ARCH, DEVICE, PREPROCESS, load
from local_common import POS, SEED, gt_pairs, hq_path, shard, shard_tag

ALPHA, N_CUTOFF = 1.0, 50
FILLS = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
N_NULL, MAX_FILL, GRID = 3, 0.8, 128
NULL_SEED, PAIR_SEED = 30_000_000, 50_000_000


def random_shape(mask, rng, exclude=None):
    """Smooth random shape with mask's pixel count, outside `exclude` (default: the mask). As in celeba_soft.py."""
    k = int(mask.sum()); h, w = mask.shape
    sigma = max(1.0, 0.25 * np.sqrt(k / mask.size) * GRID)
    field = zoom(gaussian_filter(rng.standard_normal((GRID, GRID)), sigma, mode="wrap"), (h / GRID, w / GRID), order=1)[:h, :w]
    field = field + 1e-9 * rng.standard_normal((h, w))
    field[mask if exclude is None else exclude] = -np.inf
    return field >= np.partition(field.ravel(), -k)[-k]


def cutoffs(enc, queries):
    """Per-concept z-score cutoff from 50 official-train positives, with the existing local run's shuffle sequence."""
    names = (CELEBA_ROOT / "list_attr_celeba.txt").read_text().splitlines()[1].split()
    lab = {}
    for line in (CELEBA_ROOT / "list_attr_celeba.txt").read_text().splitlines()[2:]:
        p = line.split()
        if p:
            lab[p[0]] = [v == "1" for v in p[1:]]
    part = dict(line.split() for line in open(CELEBA_ROOT / "list_eval_partition.txt"))
    train = sorted(f for f, p in part.items() if p == "0")
    rng = random.Random(SEED)
    rng.shuffle([CELEBA_ROOT / "img_align_celeba" / f for f in train])  # TCAV random sets in the original run
    out = {}
    for c in GROUNDABLE_CONCEPTS:
        j = names.index(c)
        pos = [CELEBA_ROOT / "img_align_celeba" / f for f in train if lab[f][j]]
        rng.shuffle(pos)
        maps = []
        for p in pos[:N_CUTOFF]:
            img = Image.open(p).convert("RGB")
            maps.append(localize_concept(enc, img, queries[c], (img.height, img.width)))
        out[c] = concept_zscore_cutoff(maps, ALPHA)
        print(f"  {c:20s} cutoff {out[c]:+.4f}", flush=True)
    return out


@torch.no_grad()
def main():
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create({"seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE}})
    enc = instantiate_encoder(cfg)
    center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, enc)
    q = orthogonalize_queries({c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], enc), center) for c in GROUNDABLE_CONCEPTS})
    q = {c: v.to(DEVICE) for c, v in q.items()}
    model = load().to(DEVICE).eval()
    print("black box:", ARCH, flush=True)
    cut = cutoffs(enc, q)
    pairs = gt_pairs(); todo = shard(pairs)
    print(f"{len(pairs)} pairs, {len(todo)} in this shard", flush=True)

    def logits(ims):
        return model(torch.stack([PREPROCESS(i) for i in ims]).to(DEVICE)).cpu().numpy()

    cols = ["image", "concept_name", "target_task", "gt_delta_p", "cards", "fill", "n_null", "null"]
    cols += [f"{f}_{x}" for f in FILLS for x in ("first", "second")]
    fh = open(out_dir / f"soft_{shard_tag()}.csv", "w", newline=""); w = csv.DictWriter(fh, cols); w.writeheader()
    for n, (pair_i, (name, c, tasks)) in enumerate(todo):
        ci = GROUNDABLE_CONCEPTS.index(c)
        image = Image.open(hq_path(name)).convert("RGB")
        m = threshold_mask(localize_concept(enc, image, q[c], (image.height, image.width)), method="fixed", cutoff=cut[c])
        base = {t: {"image": name, "concept_name": c, "target_task": t, "gt_delta_p": d} for t, d in tasks.items()}
        if not m.any() or m.all():
            for t in tasks:
                w.writerow(base[t] | {"cards": 0.0, "n_null": 0})
            continue
        rng = np.random.default_rng(SEED + ci * 10_000 + pair_i)
        cands = [mask_region(image, m, strategy=s, rng=rng) for s in FILLS]
        e = enc.encode_images([image] + cands).to(DEVICE)
        angs = []
        for i in range(len(FILLS)):
            d = e[0] - e[1 + i]; d = d / d.norm()
            angs.append(float(np.degrees(np.arccos(float(torch.clamp(d @ q[c], -1.0, 1.0))))))
        j = int(np.argmin(angs)); fill = FILLS[j]
        batch = [image, cands[j]]
        k, free = int(m.sum()), int((~m).sum())
        r3 = np.random.default_rng(NULL_SEED + ci * 10_000 + pair_i)
        shapes = [random_shape(m, r3) for _ in range(N_NULL)] if k <= MAX_FILL * free else []
        batch += [mask_region(image, s, strategy=fill, rng=r3) for s in shapes]
        pt = None
        if 2 * k <= MAX_FILL * free:
            r4 = np.random.default_rng(PAIR_SEED + ci * 10_000 + pair_i)
            s0 = random_shape(m, r4); s1 = random_shape(m, r4, exclude=m | s0); pt = len(batch)
            for f in FILLS:
                rf = np.random.default_rng(PAIR_SEED + 7 + ci * 10_000 + pair_i)
                one = mask_region(image, s0, strategy=f, rng=rf)
                batch += [one, mask_region(image, s1, strategy=f, rng=rf), mask_region(one, s1, strategy=f, rng=rf)]
        L = logits(batch)
        for t in tasks:
            p = POS[t]; row = base[t] | {"cards": L[0, p] - L[1, p], "fill": fill, "n_null": len(shapes)}
            if shapes:
                row["null"] = float((L[0, p] - L[2:2 + len(shapes), p]).mean())
            if pt is not None:
                for i, f in enumerate(FILLS):
                    s0i, s1i, s01i = L[pt + 3 * i, p], L[pt + 3 * i + 1, p], L[pt + 3 * i + 2, p]
                    row[f"{f}_first"], row[f"{f}_second"] = L[0, p] - s1i, s0i - s01i
            w.writerow(row)
        if (n + 1) % 100 == 0:
            fh.flush(); print(f"  {n + 1}/{len(todo)} pairs", flush=True)
    fh.close(); print("done", flush=True)


if __name__ == "__main__":
    main()
