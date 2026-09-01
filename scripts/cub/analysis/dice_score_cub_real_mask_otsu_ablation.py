"""Ablates the fixed top-15% threshold against Otsu's method (per-image
adaptive, no external calibration), prompted directly ("Can we ablate
with the Otsu's method?") -- the direct follow-up to v69's Dice finding
that top-15% (carried over unchanged from CelebA) is wildly oversized
for CUB's small body parts (up to ~150x for the eye). Same 4 CUB70-
real-masked parts, same 25-image-per-part sample as v69's own Dice
check, for a fair paired comparison -- both thresholds scored on the
IDENTICAL images/masks.
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

from localize_concept_patches_celeba import otsu_threshold, patch_similarity_grid, upsample_to_mask

from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.cub_parts import load_cub70_mask, load_images_txt
from cards.pipeline import instantiate_encoder

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
CUB70_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB70_part_segmentation")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_PER_PART = 25
TOP_PCT = 15

PART_TO_QUERY = {
    "beak": "a bird with a dagger bill",
    "left_wing": "a bird with brown wings",
    "tail": "a bird with a notched_tail",
    "left_eye": "a bird with black eyes",
}


def dice_score(pred: np.ndarray, real: np.ndarray) -> float:
    intersection = np.logical_and(pred, real).sum()
    denom = pred.sum() + real.sum()
    return 2.0 * intersection / denom if denom > 0 else 0.0


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

    all_dice_top, all_dice_otsu = [], []
    for part_name, query_text in PART_TO_QUERY.items():
        t_c = build_concept_query(query_text, encoder)
        t_c = demean_query(t_c, text_center)

        candidates = list(by_part.get(part_name, []))
        rng_py.shuffle(candidates)

        dice_top, dice_otsu, otsu_area_fracs, n_degenerate = [], [], [], 0
        for image_id, class_dir, stem in candidates:
            if len(dice_top) >= N_PER_PART:
                break
            real_mask = load_cub70_mask(CUB70_ROOT, class_dir, stem, part_name)
            if real_mask is None or not real_mask.any() or real_mask.all():
                continue
            image = Image.open(image_paths[image_id]).convert("RGB")
            if real_mask.shape != (image.height, image.width):
                continue

            sim_grid = patch_similarity_grid(siglip_model, siglip_preprocess, image, t_c)
            sim_map = upsample_to_mask(sim_grid, real_mask.shape)

            top_thresh = np.percentile(sim_map, 100 - TOP_PCT)
            top_mask = sim_map >= top_thresh

            otsu_t = otsu_threshold(sim_map.flatten())
            otsu_mask = sim_map >= otsu_t
            if not otsu_mask.any() or otsu_mask.all():
                n_degenerate += 1
                otsu_mask = top_mask  # fallback, matches the hybrid scripts' own degenerate-case handling

            dice_top.append(dice_score(top_mask, real_mask))
            dice_otsu.append(dice_score(otsu_mask, real_mask))
            otsu_area_fracs.append(otsu_mask.mean())

        dt, do, oa = np.array(dice_top), np.array(dice_otsu), np.array(otsu_area_fracs)
        all_dice_top.extend(dice_top)
        all_dice_otsu.extend(dice_otsu)
        print(f"{part_name:<12s} n={len(dt):>3d}  Dice top-15%={dt.mean():.3f}  Dice Otsu={do.mean():.3f}  "
              f"(Otsu area={oa.mean():.1%} of image, {n_degenerate} degenerate->fallback)", flush=True)

    dt, do = np.array(all_dice_top), np.array(all_dice_otsu)
    print(f"\nOVERALL  n={len(dt)}  Dice top-15%={dt.mean():.3f}  Dice Otsu={do.mean():.3f}", flush=True)
    n_improved = int((do > dt).sum())
    print(f"Otsu improved Dice on {n_improved}/{len(dt)} individual images", flush=True)


if __name__ == "__main__":
    main()
