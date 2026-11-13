# ruff: noqa: SIM115
"""Local (per-image) Hide and Seek (CARDS masking hybrid) on the COCO-Stuff local records (common.py), ResNet18.

Same configuration as the COCO headline runs (run_cards_coco_stuff_model_variants.py, ablate_patch_count.py): SigLIP
ViT-B-16 (webli), concept queries de-meaned with the generic text center and Loewdin-orthogonalized over every
non-dropped Broden concept, per-concept z-score cutoff (alpha 1.5) pooled over the top-50 COCO val2017 images
retrieved for the concept, best-of-7 fill (smallest angle between the embedding shift and the query).
Local score of (image, concept, class) = class logit(original) - class logit(concept pseudo-mask filled); 0 when the
pseudo-mask is empty or covers the whole image.

argv[1] = cutoffs: compute the cutoffs of the record concepts once -> $COCO_LOCAL_OUT/cards_cutoffs.json (run before
          the sharded scoring so the shards do not race on the val2017 embedding cache).
argv[1] = score:   sharded over (image, concept) groups -> cards_{SHARD}.csv."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import common as C

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.coco_stuff import COCO_STUFF_20_CLASSES
from cards.data.datasets import load_coco_stuff
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import mask_region

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K, ALPHA = 50, 1.5
FILLS = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
CUTOFFS = C.OUT / "cards_cutoffs.json"


def setup():
    cfg = OmegaConf.create({"seed": C.SEED, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache"})
    enc = instantiate_encoder(cfg)
    center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, enc)
    bank = C.all_concepts()
    q = orthogonalize_queries({c: demean_query(build_concept_query(c, enc), center) for c in bank})
    return cfg, enc, bank, q


@torch.no_grad()
def cutoffs():
    cfg, enc, _, q = setup()
    cfg.dataset = {"name": "coco_stuff", "root": str(C.COCO_ROOT)}; cfg.pool_source = "val"
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg),
                              load_coco_stuff(C.COCO_ROOT, COCO_STUFF_20_CLASSES, split="val"), enc)
    out = {}
    for c in sorted({r["concept_name"] for r in C.records()}):
        idx, _ = retrieve_top_bottom_k(pool, q[c], K); maps = []
        for i in idx:
            img = Image.open(pool.paths[i]).convert("RGB")
            maps.append(localize_concept(enc, img, q[c], (img.height, img.width)))
        out[c] = float(concept_zscore_cutoff(maps, ALPHA))
        print(f"  {c:22s} cutoff {out[c]:+.4f}", flush=True)
    CUTOFFS.write_text(json.dumps(out, indent=1))
    print(f"wrote {CUTOFFS}", flush=True)


@torch.no_grad()
def score():
    cut = json.loads(CUTOFFS.read_text())
    _, enc, bank, q = setup()
    q = {c: v.to(DEVICE) for c, v in q.items()}
    spec = BACKBONES["coco_stuff"]; model = spec.load_native().to(DEVICE).eval(); pre = spec.preprocess
    cls_idx = {k: i for i, k in enumerate(COCO_STUFF_20_CLASSES)}
    todo = C.shard(C.groups(C.records()))
    print(f"{len(todo)} (image, concept) groups in this shard", flush=True)
    fh = open(C.OUT / f"cards_{C.shard_tag()}.csv", "w", newline=""); w = csv.writer(fh)
    w.writerow([*C.BASE_COLS, "cards", "fill", "mask_frac"])
    for n, (gi, ((name, c), recs)) in enumerate(todo):
        image = Image.open(C.image_path(name)).convert("RGB")
        m = threshold_mask(localize_concept(enc, image, q[c], (image.height, image.width)), method="fixed", cutoff=cut[c])
        if not m.any() or m.all():
            for r in recs:
                w.writerow([*C.base(r), 0.0, "", float(m.mean())])
            continue
        rng = np.random.default_rng(C.SEED + bank.index(c) * 10_000 + gi)
        cands = [mask_region(image, m, strategy=s, rng=rng) for s in FILLS]
        e = enc.encode_images([image] + cands).to(DEVICE)
        angs = []
        for i in range(len(FILLS)):
            d = e[0] - e[1 + i]; d = d / (d.norm() + 1e-8)
            angs.append(float(np.degrees(np.arccos(float(torch.clamp(d @ q[c], -1.0, 1.0))))))
        j = int(np.argmin(angs))
        L = model(torch.stack([pre(image), pre(cands[j])]).to(DEVICE)).cpu().numpy()
        for r in recs:
            k = cls_idx[r["class_name"]]
            w.writerow([*C.base(r), float(L[0, k] - L[1, k]), FILLS[j], float(m.mean())])
        if (n + 1) % 100 == 0:
            fh.flush(); print(f"  {n + 1}/{len(todo)} groups", flush=True)
    fh.close(); print("done", flush=True)


if __name__ == "__main__":
    {"cutoffs": cutoffs, "score": score}[sys.argv[1]]()
