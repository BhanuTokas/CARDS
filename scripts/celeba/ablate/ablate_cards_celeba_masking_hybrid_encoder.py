"""Encoder ablation for the masking hybrid (v79-v82), prompted directly
("Can we do an encoder ablation?"). Mirrors `ablate_cards_celeba_
encoder.py`'s pattern (CLIP/open_clip_h/Perception re-run at the same
settings, SigLIP reused from its own production output) but for the
same-image best-of-N-strategy hybrid instead of plain cross-image CARDS
-- checks whether the hybrid's one significant result so far (SigLIP,
Attractive, sign=73.1%, p=0.029, v81/v82) is SigLIP-specific or holds
under other encoders.

This is a bigger lift than the plain-CARDS ablation: v78's per-patch
localization trick is architecture-specific. Verified directly (not
assumed) before writing this:
- SigLIP (timm TimmModel, MAP/AttentionPoolLatent head, no standalone
  linear projection) needs the length-1-sequence attn_pool trick --
  `localize_concept_patches_celeba.patch_similarity_grid`.
- CLIP and open_clip_h (plain open_clip VisionTransformer, `attn_pool=
  None`, `pool_type='tok'`) expose a genuinely separate LINEAR
  `visual.proj` applied after `visual.ln_post` -- confirmed by
  reconstruction that `ln_post(x)[:,0] @ proj`, normalized, exactly
  reproduces `encode_image(normalize=True)` (cos=1.0). Since `ln_post`
  applies elementwise to every token before pooling, applying `proj`
  directly to patch tokens is architecturally EXACT, no attn_pool trick
  needed -- `patch_similarity_grid_linear_proj`.
- Perception (PE-Core, Meta's own `pe.CLIP`) uses `pool_type='attn'` but
  ALSO exposes a separate linear `visual.proj` after its own `ln_post`
  (via `forward_features(norm=True)`) -- same linear-bypass logic as
  CLIP/open_clip_h, using PE's own public `forward_features(norm=True,
  strip_cls_token=True)` helper -- `patch_similarity_grid_perception`.

SigLIP's own v82 numbers (7-strategy family) are NOT recomputed here,
just reused directly for the comparison table.
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

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from localize_concept_patches_celeba import PATCH_SIMILARITY_FN, upsample_to_mask
from run_cards_celeba_full import CONCEPT_QUERY_TEXT, TASK_POSITIVE_LOGIT_INDEX

from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS, TARGET_CLASSES
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
SEED = 42
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}

ENCODERS_TO_RUN = {
    "clip": {"name": "clip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
             "model_name": "ViT-B-32", "pretrained": "openai", "device": DEVICE},
    "open_clip_h": {"name": "open_clip_h", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                     "model_name": "ViT-H-14", "pretrained": "laion2b_s32b_b79k", "device": DEVICE},
    "perception_encoder": {"name": "perception_encoder", "_target_": "cards.encoders.perception_encoder.PerceptionEncoder",
                            "model_name": "PE-Core-B16-224", "perception_models_path": "../perception_models", "device": DEVICE},
}


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


def evaluate(label, task_name, records, scores, results):
    rho_result = score_method_agreement(records, scores)
    sign_result = score_sign_agreement(records, scores)
    results.append((label, task_name, rho_result, sign_result))
    if rho_result is not None:
        print(f"  [{task_name}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} "
              f"p={rho_result.spearman_p:.4g} | sign={sign_result.agreement_frac:.1%} "
              f"({sign_result.n_agree}/{sign_result.n_pairs}) binom_p={sign_result.binom_p:.4g}", flush=True)
    else:
        print(f"  [{task_name}] too few pairs", flush=True)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    records_by_task = load_records_by_task()

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()

    pairs = load_celeba(CELEBA_HQ_ROOT, split="val")
    results = []
    raw_score_rows = []  # (encoder, concept_name, target_task, hybrid_raw_score) -- for cross-encoder
    # per-concept sign-agreement comparison, checking whether the SAME
    # concepts agree/disagree across encoders, not just the same count.

    for encoder_name, encoder_cfg in ENCODERS_TO_RUN.items():
        print(f"\n=== {encoder_name} (best-of-{len(FILL_STRATEGIES)}-strategy hybrid, K={K}) ===", flush=True)
        encoder = instantiate_encoder(OmegaConf.create({"encoder": encoder_cfg, "device": DEVICE}))
        model, preprocess = encoder.model, encoder.preprocess
        similarity_fn = PATCH_SIMILARITY_FN[encoder_name]
        text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

        pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": encoder_cfg, "cache_dir": "embedding_cache"})
        pool_cfg.dataset = {"name": "celeba", "root": str(CELEBA_HQ_ROOT)}
        pool_cfg.pool_source = "val"
        pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
        print(f"pool: {len(pool.paths)} images", flush=True)

        hybrid_scores_by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
        selection_counts: Counter = Counter()
        all_angles = []

        for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
            query_text = CONCEPT_QUERY_TEXT[concept_name]
            t_c = build_concept_query(query_text, encoder)
            t_c = demean_query(t_c, text_center)
            t_c_dev = t_c.to(DEVICE)

            present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
            delta_logits = {t: [] for t in TARGET_CLASSES}

            for idx in present_indices:
                image = Image.open(pool.paths[idx]).convert("RGB")
                sim_grid = similarity_fn(model, preprocess, image, t_c)
                sim_map = upsample_to_mask(sim_grid, (image.height, image.width))
                thresh = np.percentile(sim_map, 100 - TOP_PCT)
                mask = sim_map >= thresh
                if not mask.any() or mask.all():
                    continue

                rng = np.random.default_rng(SEED + concept_idx * 10_000 + int(idx))
                candidates = [mask_region(image, mask, strategy=s, rng=rng) for s in FILL_STRATEGIES]
                with torch.no_grad():
                    embeds = encoder.encode_images([image] + candidates).to(DEVICE)
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
                selection_counts[selected_strategy] += 1
                all_angles.append(best_angle)

                pixels_orig = spec.preprocess(image).unsqueeze(0)
                pixels_masked = spec.preprocess(masked_image).unsqueeze(0)
                batch = torch.cat([pixels_orig, pixels_masked], dim=0).to(DEVICE)
                with torch.no_grad():
                    logits = native_model(batch)

                for task_name in TARGET_CLASSES:
                    task_idx = TASK_POSITIVE_LOGIT_INDEX[task_name]
                    delta_logits[task_name].append((logits[0, task_idx] - logits[1, task_idx]).item())

            for task_name in TARGET_CLASSES:
                score = float(np.mean(delta_logits[task_name]))
                hybrid_scores_by_task[task_name][(CONCEPT_TO_IDX[concept_name], 1)] = score
                raw_score_rows.append((encoder_name, concept_name, task_name, score))

            print(f"  [{concept_idx + 1:>2d}/{len(GROUNDABLE_CONCEPTS)}] {concept_name}", flush=True)

        for task_name in TARGET_CLASSES:
            evaluate(encoder_name, task_name, records_by_task[task_name], hybrid_scores_by_task[task_name], results)

        angles_arr = np.array(all_angles)
        t_stat, p_val = stats.ttest_1samp(angles_arr, 90.0)
        print(f"  angle: n={len(angles_arr)} mean={angles_arr.mean():.2f} std={angles_arr.std():.2f} deg "
              f"(t={t_stat:.2f} p={p_val:.4g})", flush=True)
        print(f"  strategy selection: {dict(selection_counts.most_common())}", flush=True)

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_encoder_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["encoder", "target_task", "n_pairs", "spearman_rho", "spearman_p",
                          "sign_agreement", "n_agree", "binom_p"])
        for label, task_name, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([label, task_name, "", "", "", "", "", ""])
            else:
                writer.writerow([label, task_name, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])
    print(f"\nSaved {len(results)} (encoder, task) rows to results/cards_celeba_masking_hybrid_encoder_ablation.csv")

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_encoder_ablation_raw_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["encoder", "concept_name", "target_task", "hybrid_raw_score"])
        writer.writerows(raw_score_rows)
    print(f"Saved {len(raw_score_rows)} raw score rows to results/cards_celeba_masking_hybrid_encoder_ablation_raw_scores.csv")


if __name__ == "__main__":
    main()
