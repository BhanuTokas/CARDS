"""Encoder ablation for orthogonalize=True, prompted directly ("Can we
also run the other encoders with demean=True and orthogonalize=True?").
v87 found orthogonalize=True (K=50, demean=True) the strongest CelebA
config on SigLIP; v83 found the hybrid's OWN significant result (v81)
replicates across all 4 encoders with real per-concept convergence. This
checks whether orthogonalize=True's own improvement ALSO replicates
across encoders, or is SigLIP-specific.

Same architecture-specific localization dispatch as v83's own
`ablate_cards_celeba_masking_hybrid_encoder.py` (see that script's
docstring for the per-encoder mechanism details, reused unchanged via
`PATCH_SIMILARITY_FN`). Structural addition: `cards.pipeline.
orthogonalize_queries` needs the full 26-query set built up front per
encoder (demean=True applied first, matching every prior orthogonalize
run), jointly orthogonalized ONCE per encoder, then reused across all 26
concepts' retrieval -- same pattern as v87/v91's own SigLIP-only scripts.

K=50 only (not the K sweep) -- v91 found K=50 is specifically where
orthogonalize=True shows its strongest effect on SigLIP, so that's the
one config worth checking for cross-encoder replication first.

SigLIP's own v87 numbers (K=50/demean=True/orthogonalize=True) are NOT
recomputed, reused as a text reference.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

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
from cards.pipeline import instantiate_encoder, orthogonalize_queries
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

# SigLIP's own v87 K=50/demean=True/orthogonalize=True aggregate numbers.
V87_SIGLIP_REFERENCE = {
    "Attractive": {"rho": 0.5234, "rho_p": 0.006069, "sign": 0.808, "sign_p": 0.002494},
    "Young": {"rho": 0.0359, "rho_p": 0.8618, "sign": 0.577, "sign_p": 0.5572},
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
    raw_score_rows = []  # (encoder, concept_name, target_task, hybrid_raw_score)

    for encoder_name, encoder_cfg in ENCODERS_TO_RUN.items():
        print(f"\n=== {encoder_name} (K={K}, demean=True, orthogonalize=True) ===", flush=True)
        encoder = instantiate_encoder(OmegaConf.create({"encoder": encoder_cfg, "device": DEVICE}))
        model, preprocess = encoder.model, encoder.preprocess
        similarity_fn = PATCH_SIMILARITY_FN[encoder_name]
        text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

        pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": encoder_cfg, "cache_dir": "embedding_cache"})
        pool_cfg.dataset = {"name": "celeba", "root": str(CELEBA_HQ_ROOT)}
        pool_cfg.pool_source = "val"
        pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
        print(f"pool: {len(pool.paths)} images", flush=True)

        print("Building all 26 queries (demean=True) and jointly orthogonalizing...", flush=True)
        raw_queries = {c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], encoder), text_center) for c in GROUNDABLE_CONCEPTS}
        queries = orthogonalize_queries(raw_queries)

        hybrid_scores_by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}

        for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
            t_c = queries[concept_name]
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
                masked_image = candidates[best_i]

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

    print("\n=== siglip (K=50, demean=True, orthogonalize=True) -- v87 reference, not recomputed ===", flush=True)
    for task_name in TARGET_CLASSES:
        r = V87_SIGLIP_REFERENCE[task_name]
        print(f"  [{task_name}] rho={r['rho']:+.4f} p={r['rho_p']:.4g} | sign={r['sign']:.1%} p={r['sign_p']:.4g}", flush=True)

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_encoder_orthogonalize_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["encoder", "target_task", "n_pairs", "spearman_rho", "spearman_p",
                          "sign_agreement", "n_agree", "binom_p"])
        for label, task_name, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([label, task_name, "", "", "", "", "", ""])
            else:
                writer.writerow([label, task_name, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])
    print(f"\nSaved {len(results)} (encoder, task) rows to results/cards_celeba_masking_hybrid_encoder_orthogonalize_ablation.csv")

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_encoder_orthogonalize_ablation_raw_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["encoder", "concept_name", "target_task", "hybrid_raw_score"])
        writer.writerows(raw_score_rows)
    print(f"Saved {len(raw_score_rows)} raw score rows to results/cards_celeba_masking_hybrid_encoder_orthogonalize_ablation_raw_scores.csv")


if __name__ == "__main__":
    main()
