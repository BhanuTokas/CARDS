"""K ablation for the z-score threshold's own best config (z-score alpha=1.0,
orthogonalize=True -- v96, the new best-known result in the whole
track), prompted directly ("And a K ablation as well please!"). v89
found K=15 a clean improvement over K=50 WITHOUT orthogonalize; v91
found that trend REVERSES once orthogonalize=True is on (K=50 becomes
best). Neither of those used the z-score threshold (v96) -- this checks
whether the z-score mechanism's own K-dependence matches v91's own
reversed (larger-K-wins) pattern, or v89's original (smaller-K-wins)
one, or something else entirely.

Uses the NEW integrated library code (cards.attribution.localization),
not the old external PATCH_SIMILARITY_FN/patch_similarity_grid dispatch
v89/v91's own scripts used -- same reasoning as
ablate_cards_celeba_masking_hybrid_encoder_zscore.py's own docstring.

Orthogonalization only depends on the 26 raw query vectors, not K
(`cards.pipeline.orthogonalize_queries` needs the full query SET, not
how many images get retrieved afterward) -- built ONCE, reused across
both K=15 and K=30. K=50's own z-score alpha=1.0 numbers are v96's own
result, NOT recomputed, reused as reference.
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT, TASK_POSITIVE_LOGIT_INDEX

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
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
ALPHA = 1.0
SEED = 42
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
K_VALUES = [15, 30]

# v96's own K=50/z-score alpha=1.0/orthogonalize=True numbers, for reference.
V96_K50_REFERENCE = {
    "Attractive": {"rho": 0.6150, "rho_p": 0.00083, "sign": 0.846, "sign_p": 0.00053},
    "Young": {"rho": 0.0687, "rho_p": 0.7387, "sign": 0.538, "sign_p": 0.845},
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
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    cfg.dataset = {"name": "celeba", "root": str(CELEBA_HQ_ROOT)}
    cfg.pool_source = "val"
    pairs = load_celeba(CELEBA_HQ_ROOT, split="val")
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images", flush=True)

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()

    raw_queries = {
        c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], encoder), text_center)
        for c in GROUNDABLE_CONCEPTS
    }
    queries = orthogonalize_queries(raw_queries)

    results = []  # (K, task_name, rho_result, sign_result)
    all_rows = []  # (K, concept_name, target_task, hybrid_raw_score)

    for K in K_VALUES:
        print(f"\n=== K={K} (demean=True, orthogonalize=True, z-score alpha={ALPHA}) ===", flush=True)
        hybrid_scores_by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}

        for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
            t_c = queries[concept_name]
            t_c_dev = t_c.to(DEVICE)

            present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)

            sim_maps = []
            cached = []  # (idx, image, sim_map)
            for idx in present_indices:
                image = Image.open(pool.paths[idx]).convert("RGB")
                sim_map = localize_concept(encoder, image, t_c, (image.height, image.width))
                cached.append((idx, image, sim_map))
                sim_maps.append(sim_map)
            cutoff = concept_zscore_cutoff(sim_maps, ALPHA)

            delta_logits = {t: [] for t in TARGET_CLASSES}
            for idx, image, sim_map in cached:
                mask = threshold_mask(sim_map, method="fixed", cutoff=cutoff)
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
                score = float(np.mean(delta_logits[task_name])) if delta_logits[task_name] else 0.0
                hybrid_scores_by_task[task_name][(CONCEPT_TO_IDX[concept_name], 1)] = score
                all_rows.append((K, concept_name, task_name, score))

            print(f"  [{concept_idx + 1:>2d}/{len(GROUNDABLE_CONCEPTS)}] {concept_name}", flush=True)

        for task_name in TARGET_CLASSES:
            rho_result = score_method_agreement(records_by_task[task_name], hybrid_scores_by_task[task_name])
            sign_result = score_sign_agreement(records_by_task[task_name], hybrid_scores_by_task[task_name])
            results.append((K, task_name, rho_result, sign_result))
            if rho_result is not None:
                print(f"  [{task_name}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} "
                      f"p={rho_result.spearman_p:.4g} | sign={sign_result.agreement_frac:.1%} "
                      f"({sign_result.n_agree}/{sign_result.n_pairs}) binom_p={sign_result.binom_p:.4g}", flush=True)

    print("\n=== K=50 (demean=True, orthogonalize=True, z-score alpha=1.0) -- v96 reference, not recomputed ===", flush=True)
    for task_name in TARGET_CLASSES:
        r = V96_K50_REFERENCE[task_name]
        print(f"  [{task_name}] rho={r['rho']:+.4f} p={r['rho_p']:.4g} | sign={r['sign']:.1%} p={r['sign_p']:.4g}", flush=True)

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_k_zscore_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "target_task", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for K, task_name, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([K, task_name, "", "", "", "", "", ""])
            else:
                writer.writerow([K, task_name, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])
    print(f"\nSaved {len(results)} (K, task) rows to results/cards_celeba_masking_hybrid_k_zscore_ablation.csv")

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_k_zscore_ablation_raw_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "concept_name", "target_task", "hybrid_raw_score"])
        writer.writerows(all_rows)
    print(f"Saved {len(all_rows)} raw score rows to results/cards_celeba_masking_hybrid_k_zscore_ablation_raw_scores.csv")


if __name__ == "__main__":
    main()
