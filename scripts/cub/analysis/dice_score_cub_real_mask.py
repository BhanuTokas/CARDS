"""Dice score for the actual, DEPLOYED pseudo-mask (top-15% threshold --
the same operating point run_cards_cub_masking_hybrid_best_of_family_
full.py uses to build its masked images), prompted directly ("Should we
not use something like Dice Loss?"). AUROC (localize_concept_patches_
cub_real_mask_auroc.py) is threshold-free -- it only checks the RANKING
is right, across every possible cutoff -- but says nothing about whether
top-15% specifically is a good-sized mask for a given part. Dice = 2|P
intersect R| / (|P|+|R|) directly measures overlap AT that one
threshold, and will penalize a mask that ranks correctly but is the
wrong SIZE (e.g. top-15% is likely far larger than a beak's own true
area, so ranking can be excellent while Dice is still poor).

Also reports each part's own real-mask area fraction, to make clear
whether a low Dice score reflects a genuine localization failure or
simply a scale mismatch between the fixed 15% budget and that part's
true size.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "celeba" / "analysis"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from localize_concept_patches_celeba import patch_similarity_grid, upsample_to_mask

from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.cub_parts import load_cub70_mask, load_images_txt
from cards.pipeline import instantiate_encoder

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
CUB70_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB70_part_segmentation")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_PER_PART = 25
TOP_PCT = 15  # the actual threshold the masking hybrid uses

PART_TO_QUERY = {
    "beak": "a bird with a dagger bill",
    "left_wing": "a bird with brown wings",
    "tail": "a bird with a notched_tail",
    "left_eye": "a bird with black eyes",
}


def dice_score(pred: np.ndarray, real: np.ndarray) -> float:
    intersection = np.logical_and(pred, real).sum()
    return 2.0 * intersection / (pred.sum() + real.sum())


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

    all_dice = []
    for part_name, query_text in PART_TO_QUERY.items():
        t_c = build_concept_query(query_text, encoder)
        t_c = demean_query(t_c, text_center)

        candidates = list(by_part.get(part_name, []))
        rng_py.shuffle(candidates)

        dices, real_area_fracs, pred_area_fracs = [], [], []
        for image_id, class_dir, stem in candidates:
            if len(dices) >= N_PER_PART:
                break
            real_mask = load_cub70_mask(CUB70_ROOT, class_dir, stem, part_name)
            if real_mask is None or not real_mask.any() or real_mask.all():
                continue
            image = Image.open(image_paths[image_id]).convert("RGB")
            if real_mask.shape != (image.height, image.width):
                continue

            sim_grid = patch_similarity_grid(siglip_model, siglip_preprocess, image, t_c)
            sim_map = upsample_to_mask(sim_grid, real_mask.shape)
            thresh = np.percentile(sim_map, 100 - TOP_PCT)
            pred_mask = sim_map >= thresh

            dices.append(dice_score(pred_mask, real_mask))
            real_area_fracs.append(real_mask.mean())
            pred_area_fracs.append(pred_mask.mean())
            all_dice.append(dices[-1])

        d, r, p = np.array(dices), np.array(real_area_fracs), np.array(pred_area_fracs)
        print(f"{part_name:<12s} n={len(d):>3d}  Dice mean={d.mean():.3f} std={d.std():.3f}  "
              f"| real area={r.mean():.1%} of image  pred area={p.mean():.1%} of image "
              f"(fixed top-{TOP_PCT}%)", flush=True)

    arr = np.array(all_dice)
    print(f"\nOVERALL  n={len(arr)}  Dice mean={arr.mean():.3f} std={arr.std():.3f}", flush=True)


if __name__ == "__main__":
    main()
