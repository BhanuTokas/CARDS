"""Angle diagnostic for the K ablation, prompted directly ("Is the
average angle lower with lower K?"). v89's own K ablation
(ablate_cards_celeba_masking_hybrid_k.py) computed the same angle(diff,
t_c) quantity internally (to pick the best-aligned fill strategy per
image) but never saved it -- this is the same mechanism, K=15/30/50,
demean=True/orthogonalize=False, with the angle tracked and aggregated
this time. Recomputes K=50 too (not just 15/30) since the original K=50
baseline run also never saved its own angles.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from localize_concept_patches_celeba import patch_similarity_grid, upsample_to_mask
from run_cards_celeba_full import CONCEPT_QUERY_TEXT, TASK_POSITIVE_LOGIT_INDEX

from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS, TARGET_CLASSES
from cards.data.datasets import load_celeba
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import mask_region

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_PCT = 15
SEED = 42
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
K_VALUES = [15, 30, 50]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
    })
    encoder = instantiate_encoder(cfg)
    siglip_model, siglip_preprocess = encoder.model, encoder.preprocess
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    cfg.dataset = {"name": "celeba", "root": str(CELEBA_HQ_ROOT)}
    cfg.pool_source = "val"
    pairs = load_celeba(CELEBA_HQ_ROOT, split="val")
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images", flush=True)

    all_rows = []  # (K, concept_name, image_idx, angle_degrees)

    for K in K_VALUES:
        print(f"\n=== K={K} ===", flush=True)
        k_angles = []

        for concept_name in GROUNDABLE_CONCEPTS:
            t_c = build_concept_query(CONCEPT_QUERY_TEXT[concept_name], encoder)
            t_c = demean_query(t_c, text_center)
            t_c_dev = t_c.to(DEVICE)

            present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
            concept_angles = []

            for idx in present_indices:
                image = Image.open(pool.paths[idx]).convert("RGB")
                sim_grid = patch_similarity_grid(siglip_model, siglip_preprocess, image, t_c)
                sim_map = upsample_to_mask(sim_grid, (image.height, image.width))
                thresh = np.percentile(sim_map, 100 - TOP_PCT)
                mask = sim_map >= thresh
                if not mask.any() or mask.all():
                    continue

                rng = np.random.default_rng(SEED + CONCEPT_TO_IDX[concept_name] * 10_000 + int(idx))
                candidates = [mask_region(image, mask, strategy=s, rng=rng) for s in FILL_STRATEGIES]
                with torch.no_grad():
                    embeds = encoder.encode_images([image] + candidates).to(DEVICE)
                embed_orig = embeds[0]
                best_angle = None
                for i in range(len(FILL_STRATEGIES)):
                    diff = embed_orig - embeds[1 + i]
                    diff_unit = diff / diff.norm()
                    cos_sim = float(torch.clamp(diff_unit @ t_c_dev, -1.0, 1.0))
                    angle_deg = float(np.degrees(np.arccos(cos_sim)))
                    if best_angle is None or angle_deg < best_angle:
                        best_angle = angle_deg

                concept_angles.append(best_angle)
                k_angles.append(best_angle)
                all_rows.append((K, concept_name, int(idx), best_angle))

        arr = np.array(k_angles)
        t_stat, p_val = stats.ttest_1samp(arr, 90.0)
        print(f"K={K}: n={len(arr)} angle mean={arr.mean():.2f} std={arr.std():.2f} deg "
              f"(t-test vs 90deg: t={t_stat:.2f} p={p_val:.4g})", flush=True)

    print("\n=== Summary ===", flush=True)
    for K in K_VALUES:
        arr = np.array([a for k, _c, _i, a in all_rows if k == K])
        print(f"K={K:>3d}: n={len(arr)}  mean={arr.mean():.2f} deg  std={arr.std():.2f} deg", flush=True)

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_k_angle_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "concept_name", "image_idx", "angle_degrees"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to results/cards_celeba_masking_hybrid_k_angle_ablation.csv")


if __name__ == "__main__":
    main()
