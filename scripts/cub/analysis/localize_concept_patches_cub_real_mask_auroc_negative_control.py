"""Negative control for localize_concept_patches_cub_real_mask_auroc.py's
own strong AUROC=0.878 result -- same discipline as v78's CelebA
negative control: checks whether the patch-similarity map is genuinely
text/part-specific, or partly just "the bird's own foreground salience"
(plausible here given the visual examples showed masks often spreading
across the whole bird rather than tightly on one part). Scores each
part's real CUB70 mask against a DIFFERENT, deliberately mismatched
part's query (fixed cyclic shift: beak mask vs wing query, wing mask vs
tail query, tail mask vs eye query, eye mask vs beak query).
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "celeba" / "analysis"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from localize_concept_patches_celeba import patch_similarity_grid, upsample_to_mask

from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.cub_parts import load_cub70_mask, load_images_txt
from cards.pipeline import instantiate_encoder

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
CUB70_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB70_part_segmentation")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_PER_PART = 25

PART_TO_QUERY = {
    "beak": "a bird with a dagger bill",
    "left_wing": "a bird with brown wings",
    "tail": "a bird with a notched_tail",
    "left_eye": "a bird with black eyes",
}
PART_NAMES = list(PART_TO_QUERY)


def main():
    rng_py = random.Random(SEED)

    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
    })
    encoder = instantiate_encoder(cfg)
    siglip_model, siglip_preprocess = encoder.model, encoder.preprocess
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    image_paths = load_images_txt(CUB_ROOT)
    stem_to_id = {Path(p).stem: image_id for image_id, p in image_paths.items()}

    by_part: dict[str, list] = {}
    for class_dir in sorted((CUB70_ROOT / "AnnotationMasksPerclass").iterdir(), key=lambda p: int(p.name)):
        for mask_file in class_dir.glob("*.png"):
            name = mask_file.stem
            for part_name in PART_TO_QUERY:
                suffix = f"_{part_name}"
                if name.endswith(suffix):
                    stem = name[: -len(suffix)]
                    image_id = stem_to_id.get(stem)
                    if image_id is not None:
                        by_part.setdefault(part_name, []).append((image_id, class_dir.name, stem))
                    break

    mismatch_pairs = [(p, PART_NAMES[(i + 1) % len(PART_NAMES)]) for i, p in enumerate(PART_NAMES)]

    all_aurocs = []
    for mask_part, query_part in mismatch_pairs:
        query_text = PART_TO_QUERY[query_part]
        t_c = build_concept_query(query_text, encoder)
        t_c = demean_query(t_c, text_center)

        candidates = list(by_part.get(mask_part, []))
        rng_py.shuffle(candidates)

        aurocs = []
        for image_id, class_dir, stem in candidates:
            if len(aurocs) >= N_PER_PART:
                break
            real_mask = load_cub70_mask(CUB70_ROOT, class_dir, stem, mask_part)
            if real_mask is None or not real_mask.any() or real_mask.all():
                continue
            image = Image.open(image_paths[image_id]).convert("RGB")
            if real_mask.shape != (image.height, image.width):
                continue

            sim_grid = patch_similarity_grid(siglip_model, siglip_preprocess, image, t_c)
            sim_map = upsample_to_mask(sim_grid, real_mask.shape)
            aurocs.append(roc_auc_score(real_mask.flatten(), sim_map.flatten()))

        arr = np.array(aurocs)
        all_aurocs.extend(aurocs)
        print(f"mask={mask_part:<12s} query={query_part:<12s} (MISMATCHED)  n={len(arr):>3d}  "
              f"AUROC mean={arr.mean():.3f} std={arr.std():.3f}", flush=True)

    arr = np.array(all_aurocs)
    t_stat, p_val = stats.ttest_1samp(arr, 0.5)
    print(f"\nOVERALL MISMATCHED  n={len(arr)}  AUROC mean={arr.mean():.3f} std={arr.std():.3f}  "
          f"t-test vs 0.5: t={t_stat:.3f} p={p_val:.4g}", flush=True)


if __name__ == "__main__":
    main()
