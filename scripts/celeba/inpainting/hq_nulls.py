# ruff: noqa: C408, SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""CARDS on the clean CelebA-HQ pool, with three nulls per image. Run from the repo root.
argv[1]=output CSV. env HQ_LIMIT=n concepts (smoke; also checks the real path against the repo masking_score)."""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

sys.path.insert(0, "scripts/celeba/run"); sys.path.insert(0, "src")
from run_cards_celeba_masking_hybrid_official_train import (
    ALPHA,
    CELEBA_ROOT,
    CONCEPT_QUERY_TEXT,
    DEVICE,
    SEED,
    TASK_SLICE,
    K,
)

from cards.attribution.localization import concept_zscore_cutoff, threshold_mask
from cards.attribution.masking_mode import DEFAULT_FILL_STRATEGIES as FILLS
from cards.attribution.masking_mode import masking_score
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba import load_celebamask_hq_image_paths, load_celebamask_hq_mask
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder, orthogonalize_queries, process_concept
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.validation.broden_faithfulness import mask_region, random_placements

sys.path.insert(0, "scripts/celeba/ground_truth")
from run_celeba_official_train_faithfulness import CELEBA_HQ_ROOT as HQ

CACHE = Path(os.environ.get("CARDS_HQ_EMBED_CACHE", "embedding_cache_hq"))
OUT = Path(sys.argv[1]) if __name__ == "__main__" else None; LIMIT = int(os.environ.get("HQ_LIMIT", "0"))
N_DRAWS = 3; N_OTHER = 3; MAX_OVERLAP = 0.1; MAX_BG_FILL = 0.8; GRID = 128
ALL_REGIONS = ["skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow", "l_ear", "r_ear", "mouth", "u_lip", "l_lip",
               "hair", "hat", "ear_r", "neck_l", "neck", "cloth"]
FIELDS = ["concept", "hq_idx", "area", "bg_area", "real_A", "real_M", "moved_n", "moved_A", "moved_M", "bg_n", "bg_A", "bg_M",
          "other_n", "other_concepts", "other_area_ratio", "other_A", "other_M"]


def clean_hq_indices():
    hq_to_orig = {}
    with open(HQ / "CelebA-HQ-to-CelebA-mapping.txt") as f:
        next(f)
        for line in f:
            p = line.split()
            hq_to_orig[int(p[0])] = p[2]
    part = dict(line.split() for line in open(CELEBA_ROOT / "list_eval_partition.txt"))
    paths = load_celebamask_hq_image_paths(HQ)
    return sorted(i for i in paths if part.get(hq_to_orig[i]) in ("1", "2")), paths


def bg_blob(mask, bg, rng):
    """Equal-area random smooth shape placed only on background pixels; None if it can't fit."""
    k = int(mask.sum())
    if k == 0 or k > MAX_BG_FILL * bg.sum():
        return None
    h, w = mask.shape
    sigma = max(1.0, 0.25 * np.sqrt(k / mask.size) * GRID)
    field = zoom(gaussian_filter(rng.standard_normal((GRID, GRID)), sigma, mode="wrap"), (h / GRID, w / GRID), order=1)[:h, :w]
    field = field + 1e-9 * rng.standard_normal((h, w))
    field[~bg] = -np.inf
    return field >= np.partition(field.ravel(), -k)[-k]


@torch.no_grad()
def main():
    cfg = OmegaConf.create({"seed": SEED, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": str(CACHE), "retrieval": {"strategy": "naive"}, "k": K,
        "masking_hybrid": {"threshold_method": "zscore", "alpha": ALPHA}})
    enc = instantiate_encoder(cfg)
    center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, enc)
    Q = orthogonalize_queries({c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], enc), center)
                               for c in GROUNDABLE_CONCEPTS})
    spec = BACKBONES["celeba_official_train_attractive_male"]
    model = spec.load_native().to(DEVICE).eval()
    idxs, hq_paths = clean_hq_indices()
    pc = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": cfg.encoder, "cache_dir": str(CACHE)})
    pc.dataset = {"name": "celebamask_hq_clean_official_train_nonoverlap", "root": str(HQ)}
    pc.pool_source = "all"
    pool = load_or_build_pool(CACHE, cache_key_for(pc), [(hq_paths[i], 0) for i in idxs], enc)
    print("pool", len(pool.paths), flush=True)
    res = {c: process_concept(cfg, enc, pool, c, query=Q[c]) for c in GROUNDABLE_CONCEPTS}
    print("retrieval done", flush=True)

    patches, imgs = {}, {}

    def img(i):
        if i not in imgs:
            imgs[i] = Image.open(pool.paths[i]).convert("RGB")
        return imgs[i]

    def simmap(i, q):
        """Same math as localize_concept, with the patch embeddings cached per image."""
        if i not in patches:
            patches[i] = enc.encode_patches(img(i)).float().cpu()
        p = patches[i]
        g = round(p.shape[0] ** 0.5)
        im = img(i)
        s = (p @ q.float().cpu()).reshape(1, 1, g, g)
        return F.interpolate(s, size=(im.height, im.width), mode="bilinear", align_corners=False)[0, 0].numpy()

    cut = {}
    for n, c in enumerate(GROUNDABLE_CONCEPTS):
        cut[c] = concept_zscore_cutoff([simmap(i, res[c].query) for i in res[c].present_indices], ALPHA)
        imgs.clear()
        print(f"  cutoffs {n + 1}/26", flush=True)

    def logit(im):
        return model(spec.preprocess(im.convert("RGB")).unsqueeze(0).to(DEVICE))[0].cpu()

    def best(image, mask, rng, q):
        cands = [mask_region(image, mask, strategy=s, rng=rng) for s in FILLS]
        e = enc.encode_images([image] + cands)
        cos = []
        for k in range(len(FILLS)):
            d = e[0] - e[1 + k]
            cos.append(float(torch.clamp((d / d.norm()) @ q.to(e[0].device), -1, 1)))
        return cands[int(np.argmax(cos))]  # max cosine == min angle, as in masking_score

    def eff(o, ms):
        if not ms:
            return "", ""
        L = torch.stack([o - logit(m) for m in ms])
        return float(L[:, TASK_SLICE["Attractive"]].mean()), float(L[:, TASK_SLICE["Male"]].mean())

    done = set()
    if OUT.exists():
        done = {r["concept"] for r in csv.DictReader(open(OUT, newline=""))}
    new = not OUT.exists()
    f = open(OUT, "a", newline="")
    w = csv.DictWriter(f, FIELDS)
    if new:
        w.writeheader()
    names = GROUNDABLE_CONCEPTS[:LIMIT] if LIMIT else GROUNDABLE_CONCEPTS
    for ci, c in enumerate(GROUNDABLE_CONCEPTS):
        if c not in names or c in done:
            continue
        reals = []
        for i in res[c].present_indices:
            image = img(i)
            m = threshold_mask(simmap(i, res[c].query), method="fixed", cutoff=cut[c])
            if not m.any() or m.all():
                continue
            hq = idxs[i]
            hw = (image.height, image.width)
            o = logit(image)
            rA, rM = eff(o, [best(image, m, np.random.default_rng(SEED + ci * 10_000 + int(i)), res[c].query)])
            reals.append(rA)
            r2 = np.random.default_rng(10_000_000 + ci * 10_000 + int(i))
            places, _ = random_placements(m, r2, n_draws=N_DRAWS, strict=True)
            mA, mM = eff(o, [best(image, p, r2, res[c].query) for p in places])
            bg = ~load_celebamask_hq_mask(HQ, hq, ALL_REGIONS, target_hw=hw)
            r3 = np.random.default_rng(30_000_000 + ci * 10_000 + int(i))
            blobs = [b for b in (bg_blob(m, bg, r3) for _ in range(N_DRAWS)) if b is not None]
            bA, bM = eff(o, [best(image, b, r3, res[c].query) for b in blobs])
            # Equal-area other-concept masks: the k pixels most similar to another concept's query,
            # outside c's own mask (k = c's mask size), for 3 randomly chosen other concepts.
            r4 = np.random.default_rng(40_000_000 + ci * 10_000 + int(i))
            others = [x for x in GROUNDABLE_CONCEPTS if x != c]
            k = int(m.sum())
            cand = []
            if k <= MAX_BG_FILL * (~m).sum():
                for oc in r4.choice(others, size=N_OTHER, replace=False):
                    s = simmap(i, res[oc].query)
                    s[m] = -np.inf
                    mo = s >= np.partition(s.ravel(), -k)[-k]
                    cand.append((0.0, str(oc), mo))
            oA, oM = eff(o, [best(image, mo, r4, res[oc].query) for _, oc, mo in cand])
            w.writerow(dict(concept=c, hq_idx=hq, area=float(m.mean()), bg_area=float(bg.mean()), real_A=rA, real_M=rM,
                moved_n=len(places), moved_A=mA, moved_M=mM, bg_n=len(blobs), bg_A=bA, bg_M=bM, other_n=len(cand),
                other_concepts="|".join(oc for _, oc, _ in cand),
                other_area_ratio=float(np.mean([np.exp(d) for d, _, _ in cand])) if cand else "", other_A=oA, other_M=oM))
            imgs.pop(i, None)
        f.flush()
        print(f"[{ci + 1}/26] {c}: real A mean {np.mean(reals):+.4f} over {len(reals)} images", flush=True)
        if LIMIT and ci == 0:
            class BB:
                @staticmethod
                def preprocess(im):
                    return spec.preprocess(im.convert("RGB"))

                def __call__(self, b):
                    return model(b.to(DEVICE))[:, TASK_SLICE["Attractive"]].cpu()
            ref = masking_score(BB(), enc, pool, res[c].present_indices, res[c].query, threshold_method="zscore",
                                alpha=ALPHA, seed=SEED, concept_idx=ci).raw_score
            print(f"  CHECK repo masking_score {ref:+.4f} vs this script {np.mean(reals):+.4f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
