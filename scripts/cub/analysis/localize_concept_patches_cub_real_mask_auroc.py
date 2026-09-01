"""CUB analogue of v78's localize_concept_patches_celeba.py AUROC check,
prompted directly ("What is the AUROC for the masks?"). Scores SigLIP's
patch-similarity pseudo-mask (the same mechanism v67's CUB masking
hybrid uses) against REAL CUB70 part segmentation masks (Behzadi-
Khormouji & Oramas WACV 2023) -- the same "real ground truth, not the
keypoint-patch approximation" discipline used throughout the CUB track
for validating masks (see cards.data.cub_parts). Only the 4 direct-
mapped CUB70 parts with the clearest single-attribute text query are
tested: beak, left_wing, tail, left_eye.
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
from run_cards_cub_attributes import PREFIX_TEMPLATES

from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.cub_parts import load_cub70_mask, load_images_txt
from cards.pipeline import instantiate_encoder

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
CUB70_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB70_part_segmentation")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_PER_PART = 25

# part_name (CUB70's own) -> a representative attribute query that
# mentions that body part, for the patch-similarity text query.
PART_TO_QUERY = {
    "beak": "a bird with a dagger bill",
    "left_wing": "a bird with brown wings",
    "tail": "a bird with a notched_tail",
    "left_eye": "a bird with black eyes",
}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
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

    print("Loading CUB images.txt and CUB70 real-mask index...", flush=True)
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

    all_rows = []  # (part_name, image_id, auroc)
    for part_name, query_text in PART_TO_QUERY.items():
        t_c = build_concept_query(query_text, encoder)
        t_c = demean_query(t_c, text_center)

        candidates = list(by_part.get(part_name, []))
        rng_py.shuffle(candidates)

        aurocs = []
        for image_id, class_dir, stem in candidates:
            if len(aurocs) >= N_PER_PART:
                break
            real_mask = load_cub70_mask(CUB70_ROOT, class_dir, stem, part_name)
            if real_mask is None or not real_mask.any() or real_mask.all():
                continue
            image = Image.open(image_paths[image_id]).convert("RGB")
            if real_mask.shape != (image.height, image.width):
                continue

            sim_grid = patch_similarity_grid(siglip_model, siglip_preprocess, image, t_c)
            sim_map = upsample_to_mask(sim_grid, real_mask.shape)
            auroc = roc_auc_score(real_mask.flatten(), sim_map.flatten())
            aurocs.append(auroc)
            all_rows.append((part_name, image_id, auroc))

        arr = np.array(aurocs)
        if len(arr) >= 2:
            t_stat, p_val = stats.ttest_1samp(arr, 0.5)
            print(f"{part_name:<12s} n={len(arr):>3d}  AUROC mean={arr.mean():.3f} std={arr.std():.3f}  "
                  f"t-test vs 0.5: p={p_val:.4g}", flush=True)
        else:
            print(f"{part_name:<12s} n={len(arr)} (too few for a t-test)", flush=True)

    all_aurocs = np.array([r[2] for r in all_rows])
    t_stat, p_val = stats.ttest_1samp(all_aurocs, 0.5)
    print(f"\nOVERALL  n={len(all_aurocs)}  AUROC mean={all_aurocs.mean():.3f} std={all_aurocs.std():.3f}  "
          f"t-test vs 0.5 chance: t={t_stat:.3f} p={p_val:.4g}", flush=True)


if __name__ == "__main__":
    main()
