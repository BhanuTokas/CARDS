"""zero_fill variant of run_cards_celeba_masking_hybrid.py -- prompted
directly ("why are we using blur when it has been consistently
inferior"). blur was the default there (matching every other production
ground-truth script in this track, chosen to avoid the "out-of-
distribution hole" a flat fill creates, per the RISE-paper rationale),
but CUB's own v61 finding is that blur is a WEAK perturbation: it
preserves 75-90% of a masked region's own mean color and much of its
low-frequency structure. That matters specifically for the angle
diagnostic -- if blur barely changes the image, it will barely move the
embedding, and a tiny, noise-dominated shift could land near the random
90-degree baseline REGARDLESS of whether localization and concept-
semantics are actually coupled. zero_fill is a much more aggressive,
already-validated-in-this-repo erasure (used for the full zero_fill
ground-truth rebuild, v73) -- re-running both the angle diagnostic and
the hybrid score under it checks whether blur's gentleness, not real
decoupling, explains v79's ~90-degree result.

Identical to run_cards_celeba_masking_hybrid.py in every other respect
(same pseudo-mask, same K=50/pilot-concepts scope, same ground truth) --
only `mask_region`'s strategy and the output filenames differ.
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
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from localize_concept_patches_celeba import patch_similarity_grid, upsample_to_mask
from run_cards_celeba_full import CONCEPT_QUERY_TEXT, TASK_POSITIVE_LOGIT_INDEX

from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS, PILOT_CONCEPTS, TARGET_CLASSES
from cards.data.datasets import load_celeba
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    mask_region,
    score_method_agreement,
    score_sign_agreement,
)

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
TOP_PCT = 15
FILL_STRATEGY = "zero_fill"
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}


def load_records_by_task() -> dict[str, list[FaithfulnessResult]]:
    by_task: dict[str, list[FaithfulnessResult]] = {t: [] for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "celeba_full_faithfulness.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]].append(FaithfulnessResult(
                image=row["image"], concept_number=CONCEPT_TO_IDX[row["concept_name"]], category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return by_task


def evaluate(task_name, records, scores):
    rho_result = score_method_agreement(records, scores)
    sign_result = score_sign_agreement(records, scores)
    if rho_result is not None:
        print(f"  [{task_name}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} "
              f"p={rho_result.spearman_p:.4g} | sign={sign_result.agreement_frac:.1%} "
              f"({sign_result.n_agree}/{sign_result.n_pairs}) binom_p={sign_result.binom_p:.4g}", flush=True)
    else:
        print(f"  [{task_name}] too few pairs", flush=True)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    records_by_task = load_records_by_task()

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

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()

    hybrid_scores_by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    angle_rows = []  # (concept_name, image_idx, angle_degrees)

    for concept_name in PILOT_CONCEPTS:
        query_text = CONCEPT_QUERY_TEXT[concept_name]
        t_c = build_concept_query(query_text, encoder)
        t_c = demean_query(t_c, text_center)
        t_c_dev = t_c.to(DEVICE)

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)

        delta_logits = {t: [] for t in TARGET_CLASSES}
        concept_angles = []
        n_skipped_degenerate = 0

        for idx in present_indices:
            image = Image.open(pool.paths[idx]).convert("RGB")
            sim_grid = patch_similarity_grid(siglip_model, siglip_preprocess, image, t_c)
            sim_map = upsample_to_mask(sim_grid, (image.height, image.width))
            thresh = np.percentile(sim_map, 100 - TOP_PCT)
            mask = sim_map >= thresh
            if not mask.any() or mask.all():
                n_skipped_degenerate += 1
                continue

            masked_image = mask_region(image, mask, strategy=FILL_STRATEGY)

            with torch.no_grad():
                embeds = encoder.encode_images([image, masked_image]).to(DEVICE)  # (2, C), unit-normalized
            diff = embeds[0] - embeds[1]
            diff_unit = diff / diff.norm()
            cos_sim = float(torch.clamp(diff_unit @ t_c_dev, -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos_sim)))
            concept_angles.append(angle_deg)
            angle_rows.append((concept_name, int(idx), angle_deg))

            pixels_orig = spec.preprocess(image).unsqueeze(0)
            pixels_masked = spec.preprocess(masked_image).unsqueeze(0)
            batch = torch.cat([pixels_orig, pixels_masked], dim=0).to(DEVICE)
            with torch.no_grad():
                logits = native_model(batch)  # (2, 4)

            for task_name in TARGET_CLASSES:
                task_idx = TASK_POSITIVE_LOGIT_INDEX[task_name]
                delta_logits[task_name].append((logits[0, task_idx] - logits[1, task_idx]).item())

        for task_name in TARGET_CLASSES:
            hybrid_scores_by_task[task_name][(CONCEPT_TO_IDX[concept_name], 1)] = float(np.mean(delta_logits[task_name]))

        angles_arr = np.array(concept_angles)
        print(
            f"{concept_name:<20s} n={len(angles_arr):>3d} (skipped {n_skipped_degenerate} degenerate)  "
            f"angle(diff, t_c) mean={angles_arr.mean():.2f} std={angles_arr.std():.2f} deg", flush=True,
        )

    print(f"\n=== hybrid CARDS ({FILL_STRATEGY} same-image masking, mask-free pseudo-mask) vs. real ground truth ===", flush=True)
    for task_name in TARGET_CLASSES:
        evaluate(task_name, records_by_task[task_name], hybrid_scores_by_task[task_name])

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_scores_zerofill.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "target_task", "hybrid_raw_score"])
        for task_name in TARGET_CLASSES:
            for concept_name in PILOT_CONCEPTS:
                writer.writerow([concept_name, task_name, hybrid_scores_by_task[task_name][(CONCEPT_TO_IDX[concept_name], 1)]])

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_angles_zerofill.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "image_idx", "angle_degrees"])
        writer.writerows(angle_rows)

    all_angles = np.array([a for _c, _i, a in angle_rows])
    t_stat, p_val = stats.ttest_1samp(all_angles, 90.0)
    print(
        f"\nOVERALL angle(embed(orig)-embed(masked), t_c) under {FILL_STRATEGY}: n={len(all_angles)} "
        f"mean={all_angles.mean():.2f} std={all_angles.std():.2f} deg  "
        f"t-test vs 90deg (no preferred direction): t={t_stat:.3f} p={p_val:.4g}", flush=True,
    )
    print("Saved results/cards_celeba_masking_hybrid_scores_zerofill.csv and results/cards_celeba_masking_hybrid_angles_zerofill.csv")


if __name__ == "__main__":
    main()
