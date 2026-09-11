"""Ablation of a THIRD localization-threshold family -- a per-concept
z-score cutoff (mean + alpha*std, pooled across a concept's own present-set
similarity maps) -- prompted directly, following up on the top_pct/Otsu
ablation ("What if, alternatively we computed a threshold for each
concept based on some statistics?" -> "I am inclined towards per concept
statistics. Please add the ablation.").

Unlike top_pct (a fixed area budget regardless of how sharp the
concept's real peak is) or Otsu (assumes the similarity map is
genuinely bimodal, which notes/cub_correlation_investigation.md v69/v70
already found is false here -- these maps are closer to a smooth
gradient), a z-score cutoff adapts to each concept's own similarity
SCALE without assuming a particular distribution shape, and without
needing to renormalize the (deliberately non-unit-norm, see demean_query)
query to make cutoffs comparable across concepts -- the per-concept
pooling handles that automatically.

For each concept: localize ALL its K=50 present-set images FIRST (one
pass, cached), pool every pixel from every map into one distribution,
then derive that concept's own cutoff = mean + alpha*std for each alpha in
ALPHA_VALUES -- a genuinely two-pass-per-concept design (the cutoff can't be
computed from any single image being thresholded, only from the whole
present set), unlike top_pct/Otsu which are single-image operations.
Same 26-concept x 2-task re-score against the real masking-based
faithfulness ground truth, same settings as the (also user-requested)
orthogonalize=True rerun of ablate_cards_celeba_masking_hybrid_
threshold.py (demean_query=True, orthogonalize=True -- v87/v91's own
best-supported SigLIP config, K=50, SigLIP, 7-strategy fill family):
queries are jointly Lowdin-orthogonalized once, up front, same as that
sibling script.

5 alpha values x 26 concepts x 50 images each.
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
K = 50
SEED = 42
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
ALPHA_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5]


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

    # orthogonalize=True (v87/v91's SigLIP-best config) -- needs all 26
    # queries built up front, jointly Lowdin-orthogonalized ONCE, reused
    # unchanged across every alpha below (only the threshold varies).
    raw_queries = {
        c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], encoder), text_center)
        for c in GROUNDABLE_CONCEPTS
    }
    queries = orthogonalize_queries(raw_queries)

    # hybrid_scores_by_alpha[alpha][task_name][(concept_idx, 1)] = raw_score
    hybrid_scores_by_alpha: dict[float, dict[str, dict[tuple[int, int], float]]] = {
        alpha: {t: {} for t in TARGET_CLASSES} for alpha in ALPHA_VALUES
    }
    all_rows = []  # (alpha, concept_name, target_task, hybrid_raw_score)
    cutoff_rows = []  # (alpha, concept_name, cutoff, mean_area_pct)

    for c_i, concept_name in enumerate(GROUNDABLE_CONCEPTS):
        t_c = queries[concept_name]
        t_c_dev = t_c.to(DEVICE)
        concept_idx = CONCEPT_TO_IDX[concept_name]

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)

        # Pass 1: localize every present image ONCE, cache image + sim_map
        # -- the per-concept cutoff needs the whole present set's pooled
        # stats before any single image can be thresholded, and doing
        # this once (not once per alpha) avoids ALPHA_VALUES redundant forward
        # passes through the encoder for identical similarity maps.
        cached = []  # (idx, image, sim_map)
        for idx in present_indices:
            image = Image.open(pool.paths[idx]).convert("RGB")
            sim_map = localize_concept(encoder, image, t_c, (image.height, image.width))
            cached.append((idx, image, sim_map))
        sim_maps = [sm for _, _, sm in cached]

        print(f"[{c_i + 1:>2d}/{len(GROUNDABLE_CONCEPTS)}] {concept_name:<20s}", flush=True)

        # Pass 2: one cutoff per alpha, then mask/fill/score using the cached maps.
        for alpha in ALPHA_VALUES:
            cutoff = concept_zscore_cutoff(sim_maps, alpha)
            areas = []
            delta_logits = {t: [] for t in TARGET_CLASSES}
            n_skipped = 0

            for idx, image, sim_map in cached:
                mask = threshold_mask(sim_map, method="fixed", cutoff=cutoff)
                if not mask.any() or mask.all():
                    n_skipped += 1
                    continue
                areas.append(float(mask.mean()))

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

            mean_area_pct = 100.0 * float(np.mean(areas)) if areas else 0.0
            cutoff_rows.append((alpha, concept_name, cutoff, mean_area_pct))
            for task_name in TARGET_CLASSES:
                score = float(np.mean(delta_logits[task_name])) if delta_logits[task_name] else 0.0
                hybrid_scores_by_alpha[alpha][task_name][(concept_idx, 1)] = score
                all_rows.append((alpha, concept_name, task_name, score))

            print(f"    alpha={alpha:<4.1f} cutoff={cutoff:+.4f} mean_area={mean_area_pct:>5.1f}% "
                  f"n={len(present_indices) - n_skipped:>3d} skipped={n_skipped}", flush=True)

    results = []  # (alpha, task_name, rho_result, sign_result)
    for alpha in ALPHA_VALUES:
        print(f"\n=== alpha={alpha} ===", flush=True)
        for task_name in TARGET_CLASSES:
            rho_result = score_method_agreement(records_by_task[task_name], hybrid_scores_by_alpha[alpha][task_name])
            sign_result = score_sign_agreement(records_by_task[task_name], hybrid_scores_by_alpha[alpha][task_name])
            results.append((alpha, task_name, rho_result, sign_result))
            if rho_result is not None:
                print(f"  [{task_name}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} "
                      f"p={rho_result.spearman_p:.4g} | sign={sign_result.agreement_frac:.1%} "
                      f"({sign_result.n_agree}/{sign_result.n_pairs}) binom_p={sign_result.binom_p:.4g}", flush=True)

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_threshold_zscore_orthogonalize_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "target_task", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for alpha, task_name, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([alpha, task_name, "", "", "", "", "", ""])
            else:
                writer.writerow([alpha, task_name, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_threshold_zscore_orthogonalize_ablation_raw_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "concept_name", "target_task", "hybrid_raw_score"])
        writer.writerows(all_rows)

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_threshold_zscore_orthogonalize_ablation_cutoffs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "concept_name", "cutoff", "mean_area_pct"])
        writer.writerows(cutoff_rows)

    print(f"\nSaved {len(results)} (alpha, task) rows to results/cards_celeba_masking_hybrid_threshold_zscore_orthogonalize_ablation.csv")


if __name__ == "__main__":
    main()
