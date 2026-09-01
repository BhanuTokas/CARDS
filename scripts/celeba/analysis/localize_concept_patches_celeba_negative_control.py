"""Negative control for localize_concept_patches_celeba.py's own AUROC=0.855
result -- checks whether the patch-similarity map is genuinely text/concept
-specific, or just spatially biased (e.g. nose/eye regions sit near image
center in aligned face crops, so ANY query might score well there by
positional prior alone, not real text grounding).

Reuses the identical images/masks/mechanism as the main script, but scores
each concept's real mask against a DIFFERENT, MISMATCHED concept's own text
query (Black_Hair mask vs. Eyeglasses query, etc. -- every concept paired
with the pilot's own "next" concept in a fixed cyclic shift, so every mask
gets exactly one deliberately-wrong query). If AUROC collapses to ~0.5 here
while the matched-query run stayed at 0.855, that's real evidence the effect
is text-driven, not positional.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.celeba import load_celebamask_hq_image_paths, load_celebamask_hq_mask, split_celebamask_hq
from cards.data.celeba_attributes import ATTRIBUTE_TO_REGIONS, PILOT_CONCEPTS, TARGET_CLASSES, load_attribute_labels, load_attribute_names
from cards.pipeline import instantiate_encoder
from localize_concept_patches_celeba import patch_similarity_grid, upsample_to_mask

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PER_CONCEPT = 25


def main():
    rng_py = random.Random(SEED)

    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]
    _, val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices)

    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
    })
    encoder = instantiate_encoder(cfg)
    model, preprocess = encoder.model, encoder.preprocess
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    # Fixed cyclic mismatch: mask_concept's real mask scored against the
    # NEXT concept in PILOT_CONCEPTS' own text query.
    mismatch_pairs = [(c, PILOT_CONCEPTS[(i + 1) % len(PILOT_CONCEPTS)]) for i, c in enumerate(PILOT_CONCEPTS)]

    all_aurocs = []
    for mask_concept, query_concept in mismatch_pairs:
        region_names = ATTRIBUTE_TO_REGIONS[mask_concept]
        query_text = CONCEPT_QUERY_TEXT[query_concept]
        t_c = build_concept_query(query_text, encoder)
        t_c = demean_query(t_c, text_center)

        candidates = list(val_hq)
        rng_py.shuffle(candidates)

        aurocs = []
        for hq_idx in candidates:
            if len(aurocs) >= N_PER_CONCEPT:
                break
            mask = load_celebamask_hq_mask(CELEBA_HQ_ROOT, hq_idx, region_names)
            if not mask.any() or mask.all():
                continue
            image = Image.open(image_paths_by_idx[hq_idx]).convert("RGB")
            sim_grid = patch_similarity_grid(model, preprocess, image, t_c)
            sim_map = upsample_to_mask(sim_grid, mask.shape)
            aurocs.append(roc_auc_score(mask.flatten(), sim_map.flatten()))

        arr = np.array(aurocs)
        all_aurocs.extend(aurocs)
        print(
            f"mask={mask_concept:<20s} query={query_concept:<20s} (MISMATCHED)  n={len(arr):>3d}  "
            f"AUROC mean={arr.mean():.3f} std={arr.std():.3f}", flush=True,
        )

    arr = np.array(all_aurocs)
    t_stat, p_val = stats.ttest_1samp(arr, 0.5)
    print(f"\nOVERALL MISMATCHED  n={len(arr)}  AUROC mean={arr.mean():.3f} std={arr.std():.3f}  "
          f"t-test vs 0.5: t={t_stat:.3f} p={p_val:.4g}", flush=True)


if __name__ == "__main__":
    main()
