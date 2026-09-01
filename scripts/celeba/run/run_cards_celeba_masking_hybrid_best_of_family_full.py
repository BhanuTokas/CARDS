"""Full 26-concept scale-up of run_cards_celeba_masking_hybrid_best_of_family.py,
prompted directly ("Can you run it on the complete set?"). Pilot scale
(n=8 concepts) left the hybrid-vs-plain-CARDS comparison underpowered
(neither reached significance); this extends the SAME best-of-N-strategy
methodology (each image picks whichever fill strategy gives the smallest
angle to +t_c as its own counterfactual) to the full GROUNDABLE_CONCEPTS
set (26 concepts x 2 tasks, up to 52 scoreable pairs vs. the pilot's 8).

First full-scale run (5 strategies: blur/zero_fill/mean_fill/hue_shift/
white_fill) found CARDS' first-ever significant CelebA number
(Attractive sign=73.1%, p=0.029) but flagged it against a real
multiple-comparisons caveat (see notes v81). Extended here to 7
strategies on request ("Can we also have a noise fill strategy?, and
noise fill then blur?") -- adds `zero_fill_noise` (per-pixel Gaussian
noise, content-erasing but an unnaturally sharp/flat texture) and
`noise_then_blur` (the same noise, then blurred -- content-erasing
without either blur's own low-frequency leakage, CUB v61, or plain
noise's unnatural sharpness). Both need an rng, seeded deterministically
per (concept, image) below for reproducibility. A bigger strategy family
means MORE selection freedom per image, so the angle number is even
more selection-biased than the 5-strategy version -- the hybrid CARDS
rho/sign-agreement-vs-ground-truth numbers remain the ones that matter,
same caveat as before.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
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

from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS, TARGET_CLASSES
from cards.data.datasets import load_celeba
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import FaithfulnessResult, mask_region, score_method_agreement, score_sign_agreement

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
TOP_PCT = 15
SEED = 42
# "zero_fill_noise" and "noise_then_blur" added on request ("Can we also
# have a noise fill strategy?, and noise fill then blur?") -- both need an
# rng, threaded per-image below for reproducibility.
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
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
    angle_rows = []  # (concept_name, image_idx, selected_strategy, angle_degrees)
    selection_counts: dict[str, Counter] = {c: Counter() for c in GROUNDABLE_CONCEPTS}

    for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
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

            rng = np.random.default_rng(SEED + concept_idx * 10_000 + int(idx))
            candidates = [mask_region(image, mask, strategy=s, rng=rng) for s in FILL_STRATEGIES]
            with torch.no_grad():
                embeds = encoder.encode_images([image] + candidates).to(DEVICE)  # (6, C)
            embed_orig = embeds[0]
            best_angle, best_i = None, None
            for i in range(len(FILL_STRATEGIES)):
                diff = embed_orig - embeds[1 + i]
                diff_unit = diff / diff.norm()
                cos_sim = float(torch.clamp(diff_unit @ t_c_dev, -1.0, 1.0))
                angle_deg = float(np.degrees(np.arccos(cos_sim)))
                if best_angle is None or angle_deg < best_angle:
                    best_angle, best_i = angle_deg, i

            selected_strategy = FILL_STRATEGIES[best_i]
            masked_image = candidates[best_i]
            selection_counts[concept_name][selected_strategy] += 1
            concept_angles.append(best_angle)
            angle_rows.append((concept_name, int(idx), selected_strategy, best_angle))

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
        counts_str = ", ".join(f"{s}={c}" for s, c in selection_counts[concept_name].most_common())
        print(
            f"[{concept_idx + 1:>2d}/{len(GROUNDABLE_CONCEPTS)}] {concept_name:<20s} n={len(angles_arr):>3d} "
            f"(skipped {n_skipped_degenerate})  best-of-5 angle mean={angles_arr.mean():.2f} "
            f"std={angles_arr.std():.2f} deg  |  {counts_str}", flush=True,
        )

    print("\n=== hybrid CARDS (best-of-5-strategy same-image masking, FULL 26-concept scale) vs. real ground truth ===", flush=True)
    for task_name in TARGET_CLASSES:
        evaluate(task_name, records_by_task[task_name], hybrid_scores_by_task[task_name])

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_scores_best_of_family_full.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "target_task", "hybrid_raw_score"])
        for task_name in TARGET_CLASSES:
            for concept_name in GROUNDABLE_CONCEPTS:
                writer.writerow([concept_name, task_name, hybrid_scores_by_task[task_name][(CONCEPT_TO_IDX[concept_name], 1)]])

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_angles_best_of_family_full.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "image_idx", "selected_strategy", "angle_degrees"])
        writer.writerows(angle_rows)

    all_angles = np.array([a for *_r, a in angle_rows])
    t_stat, p_val = stats.ttest_1samp(all_angles, 90.0)
    print(
        f"\nOVERALL best-of-5 angle(embed(orig)-embed(masked), t_c): n={len(all_angles)} "
        f"mean={all_angles.mean():.2f} std={all_angles.std():.2f} deg  "
        f"t-test vs 90deg: t={t_stat:.3f} p={p_val:.4g}  "
        f"(selection-biased by construction -- see docstring)", flush=True,
    )
    overall_counts = Counter()
    for counts in selection_counts.values():
        overall_counts.update(counts)
    print("Overall strategy selection counts:", dict(overall_counts.most_common()))
    print("Saved results/cards_celeba_masking_hybrid_scores_best_of_family_full.csv and "
          "results/cards_celeba_masking_hybrid_angles_best_of_family_full.csv")


if __name__ == "__main__":
    main()
