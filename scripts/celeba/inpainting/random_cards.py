# ruff: noqa: SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""CARDS with each pseudo-mask moved off its localized region. Run from repo root.
argv[1]=output CSV; env RC_LIMIT=n concepts (smoke)."""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

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
from cards.validation.broden_faithfulness import mask_region, random_placements

N_DRAWS = 3
OUT = sys.argv[1] if __name__ == "__main__" else None; LIMIT = int(os.environ.get("RC_LIMIT", "0"))
STORED = {(r["task"], r["concept_name"]): float(r["hybrid_raw_score"]) for r in
          csv.DictReader(open("results/cards_celeba_masking_hybrid_official_train_raw_scores.csv"))}

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
    spec = BACKBONES["celeba_official_train_attractive_male"]
    model = spec.load_native().to(DEVICE).eval()
    pairs = [(p, 0) for p in build_clean_official_val_paths()]
    pc = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": cfg.encoder, "cache_dir": "embedding_cache"})
    pc.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}; pc.pool_source = "val"
    pool = load_or_build_pool(Path(pc.cache_dir), cache_key_for(pc), pairs, enc)
    print("pool", len(pool.paths), flush=True)

    def logits(ims):
        x = torch.stack([spec.preprocess(i.convert("RGB")) for i in ims]).to(DEVICE)
        return model(x).detach().cpu()

    def best(image, mask, rng, q):
        cands = [mask_region(image, mask, strategy=s, rng=rng) for s in FILLS]
        e = enc.encode_images([image] + cands)
        angs = []
        for i in range(len(FILLS)):
            d = e[0] - e[1 + i]; d = d / d.norm()
            angs.append(float(np.degrees(np.arccos(float(torch.clamp(d @ q.to(e[0].device), -1, 1))))))
        return cands[int(np.argmin(angs))]

    f = open(OUT, "w", newline=""); w = csv.writer(f)
    w.writerow(["concept", "image_idx", "area", "n_valid", "n_square", "real_A", "real_M", "moved_A", "moved_M"])
    names = GROUNDABLE_CONCEPTS[:: max(1, len(GROUNDABLE_CONCEPTS) // LIMIT)][:LIMIT] if LIMIT else GROUNDABLE_CONCEPTS
    for c in names:
        ci = CONCEPT_TO_IDX[c]; res = process_concept(cfg, enc, pool, c, query=queries[c])
        ims = {i: Image.open(pool.paths[i]).convert("RGB") for i in res.present_indices}
        maps = {i: localize_concept(enc, ims[i], res.query, (ims[i].height, ims[i].width)) for i in ims}
        cut = concept_zscore_cutoff(list(maps.values()), ALPHA)
        realA, realM = [], []
        for i in res.present_indices:
            m = threshold_mask(maps[i], method="fixed", cutoff=cut)
            if not m.any() or m.all():
                continue
            rng = np.random.default_rng(SEED + ci * 10_000 + int(i))
            masked = best(ims[i], m, rng, res.query)
            r2 = np.random.default_rng(10_000_000 + ci * 10_000 + int(i))
            places, nsq = random_placements(m, r2, n_draws=N_DRAWS, strict=True)
            out = logits([ims[i], masked] + [best(ims[i], p, r2, res.query) for p in places])
            dA = (out[0, TASK_SLICE["Attractive"]] - out[1:, TASK_SLICE["Attractive"]]).numpy()
            dM = (out[0, TASK_SLICE["Male"]] - out[1:, TASK_SLICE["Male"]]).numpy()
            realA.append(dA[0]); realM.append(dM[0])
            w.writerow([c, int(i), float(m.mean()), len(places), nsq, dA[0], dM[0],
                        dA[1:].mean() if places else "", dM[1:].mean() if places else ""])
        f.flush()
        print(f"{c}: real A {np.mean(realA):+.4f} (stored {STORED[('Attractive', c)]:+.4f})  "
              f"M {np.mean(realM):+.4f} (stored {STORED[('Male', c)]:+.4f})", flush=True)
if __name__ == "__main__":
    main()
