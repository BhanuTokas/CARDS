# ruff: noqa: SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""HnS soft correction on CelebA (official-train black box, official val pool). Run from a CARDS checkout's root.

Same retrieval, masks, best-of-7 fill and seeds as the paper's CARDS run (checked against the stored scores).
Per image: real drop D (Attractive, Male logits) and the fill best-of-7 chose; the fill-matched null A = mean drop
of 3 equal-area random smooth shapes outside the mask filled with that fill; and, for every fill, the two-patch
test: s1 on the intact image (first patch) and s1 added to an image already carrying s0 (second patch).
Black box = env CARDS_ARCH (resnet18 default, vit, convnext; black_boxes.py). argv[1] = output CSV. env SOFT_LIMIT=n concepts for a smoke run.
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

sys.path.insert(0, "scripts/celeba/run"); sys.path.insert(0, "src")
from run_cards_celeba_masking_hybrid_official_train import (
    ALPHA,
    CELEBA_ROOT,
    CONCEPT_QUERY_TEXT,
    CONCEPT_TO_IDX,
    DEVICE,
    SEED,
    TASK_SLICE,
    K,
    build_clean_official_val_paths,
)

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.attribution.masking_mode import DEFAULT_FILL_STRATEGIES as FILLS
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder, orthogonalize_queries, process_concept
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.validation.broden_faithfulness import mask_region

sys.path.insert(0, str(Path(__file__).parent))
from black_boxes import ARCH, SUFFIX, load

N_NULL = 3; MAX_FILL = 0.8; GRID = 128
NULL_SEED, PAIR_SEED = 30_000_000, 50_000_000
OUT = sys.argv[1]; LIMIT = int(os.environ.get("SOFT_LIMIT", "0"))
_STORED_CSV = Path(f"results/cards_celeba_masking_hybrid_official_train{SUFFIX}_raw_scores.csv")  # sanity print only
STORED = ({(r["task"], r["concept_name"]): float(r["hybrid_raw_score"]) for r in csv.DictReader(open(_STORED_CSV))}
          if _STORED_CSV.exists() else {})


def random_shape(mask, rng, exclude=None):
    """Smooth random shape with mask's pixel count, outside `exclude` (default: the mask). Same as BAM's run_hns_bam_nulls."""
    k = int(mask.sum()); h, w = mask.shape
    sigma = max(1.0, 0.25 * np.sqrt(k / mask.size) * GRID)
    field = zoom(gaussian_filter(rng.standard_normal((GRID, GRID)), sigma, mode="wrap"), (h / GRID, w / GRID), order=1)[:h, :w]
    field = field + 1e-9 * rng.standard_normal((h, w))
    field[mask if exclude is None else exclude] = -np.inf
    return field >= np.partition(field.ravel(), -k)[-k]


@torch.no_grad()
def main():
    cfg = OmegaConf.create({"seed": SEED, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache", "retrieval": {"strategy": "naive"}, "k": K,
        "masking_hybrid": {"threshold_method": "zscore", "alpha": ALPHA}})
    enc = instantiate_encoder(cfg)
    center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, enc)
    raw = {c: build_concept_query(CONCEPT_QUERY_TEXT[c], enc) for c in GROUNDABLE_CONCEPTS}
    queries = orthogonalize_queries({c: demean_query(q, center) for c, q in raw.items()})
    spec = BACKBONES["celeba_official_train_attractive_male"]  # preprocess only (identical for all three)
    model = load().to(DEVICE).eval()
    print("black box:", ARCH, flush=True)
    pairs = [(p, 0) for p in build_clean_official_val_paths()]
    pc = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": cfg.encoder, "cache_dir": "embedding_cache"})
    pc.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}; pc.pool_source = "val"
    pool = load_or_build_pool(Path(pc.cache_dir), cache_key_for(pc), pairs, enc)
    print("pool", len(pool.paths), flush=True)

    def task_logits(ims):
        out = []
        for b in range(0, len(ims), 64):
            x = torch.stack([spec.preprocess(i.convert("RGB")) for i in ims[b:b + 64]]).to(DEVICE)
            y = model(x).detach().cpu()
            out.append(torch.stack([y[:, TASK_SLICE["Attractive"]].reshape(len(x), -1)[:, 0],
                                    y[:, TASK_SLICE["Male"]].reshape(len(x), -1)[:, 0]], 1))
        return torch.cat(out).numpy()  # (n, 2): Attractive, Male

    def best(image, mask, rng, q):
        cands = [mask_region(image, mask, strategy=s, rng=rng) for s in FILLS]
        e = enc.encode_images([image] + cands)
        angs = []
        for i in range(len(FILLS)):
            d = e[0] - e[1 + i]; d = d / d.norm()
            angs.append(float(np.degrees(np.arccos(float(torch.clamp(d @ q.to(e[0].device), -1, 1))))))
        j = int(np.argmin(angs))
        return cands[j], FILLS[j]

    cols = ["concept", "image_idx", "area", "fill", "real_A", "real_M", "n_null", "null_A", "null_M"]
    cols += [f"{f}_{x}_{t}" for f in FILLS for x in ("first", "second") for t in ("A", "M")]
    fh = open(OUT, "w", newline=""); w = csv.DictWriter(fh, cols); w.writeheader()
    names = GROUNDABLE_CONCEPTS[:LIMIT] if LIMIT else GROUNDABLE_CONCEPTS
    for c in names:
        ci = CONCEPT_TO_IDX[c]; res = process_concept(cfg, enc, pool, c, query=queries[c])
        ims = {i: Image.open(pool.paths[i]).convert("RGB") for i in res.present_indices}
        maps = {i: localize_concept(enc, ims[i], res.query, (ims[i].height, ims[i].width)) for i in ims}
        cut = concept_zscore_cutoff(list(maps.values()), ALPHA)
        realA, realM, n_null = [], [], 0
        for i in res.present_indices:
            m = threshold_mask(maps[i], method="fixed", cutoff=cut)
            if not m.any() or m.all():
                continue
            image = ims[i]
            masked, fill = best(image, m, np.random.default_rng(SEED + ci * 10_000 + int(i)), res.query)
            batch, row = [image, masked], {"concept": c, "image_idx": int(i), "area": float(m.mean()), "fill": fill}
            k = int(m.sum()); free = int((~m).sum())
            r3 = np.random.default_rng(NULL_SEED + ci * 10_000 + int(i))
            shapes = [random_shape(m, r3) for _ in range(N_NULL)] if k <= MAX_FILL * free else []
            batch += [mask_region(image, s, strategy=fill, rng=r3) for s in shapes]
            pair = None
            if 2 * k <= MAX_FILL * free:
                r4 = np.random.default_rng(PAIR_SEED + ci * 10_000 + int(i))
                s0 = random_shape(m, r4); s1 = random_shape(m, r4, exclude=m | s0)
                pair = len(batch)
                for f in FILLS:
                    rf = np.random.default_rng(PAIR_SEED + 7 + ci * 10_000 + int(i))
                    one = mask_region(image, s0, strategy=f, rng=rf)
                    batch += [one, mask_region(image, s1, strategy=f, rng=rf), mask_region(one, s1, strategy=f, rng=rf)]
            L = task_logits(batch)
            d = L[0] - L[1]
            row |= {"real_A": d[0], "real_M": d[1], "n_null": len(shapes)}
            realA.append(d[0]); realM.append(d[1])
            if shapes:
                a = (L[0] - L[2:2 + len(shapes)]).mean(0)
                row |= {"null_A": a[0], "null_M": a[1]}; n_null += 1
            if pair is not None:
                for j, f in enumerate(FILLS):
                    s0i, s1i, s01i = L[pair + 3 * j], L[pair + 3 * j + 1], L[pair + 3 * j + 2]
                    first, second = L[0] - s1i, s0i - s01i
                    row |= {f"{f}_first_A": first[0], f"{f}_first_M": first[1],
                            f"{f}_second_A": second[0], f"{f}_second_M": second[1]}
            w.writerow(row)
        fh.flush()
        print(f"{c}: real A {np.mean(realA):+.4f} (stored {STORED.get(('Attractive', c), float('nan')):+.4f})  "
              f"M {np.mean(realM):+.4f} (stored {STORED.get(('Male', c), float('nan')):+.4f})  images with a null {n_null}/{len(realA)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
