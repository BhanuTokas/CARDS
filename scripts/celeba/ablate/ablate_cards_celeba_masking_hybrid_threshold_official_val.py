"""v95's own top_pct/Otsu threshold sweep, rerun on the official (non-HQ)
CelebA val pool instead of CelebA-HQ's own val split -- prompted directly
("Can we run ablations with the official validation data split minus the
overlap?"), extending v98 (which only tested k=1.0 on this pool) to the
full threshold family.

Pool: the SAME 16,874-image "clean" official-val pool v98/v99 already
built and cached (official CelebA val minus HQ-train leakage minus
HQ-val overlap -- see build_clean_official_val_paths, imported directly
rather than reimplemented). Ground truth: the ONLY one available --
`results/celeba_full_faithfulness.csv`, real CelebA-HQ masks, aggregated
per (concept,task) and therefore valid regardless of which images the
method itself scores on (same reasoning as v98). Classifier: the
ORIGINAL (not celeba_attractive_young_lowres) -- v99 found resolution-
matched retraining doesn't change the official-val correlation, so
there's no reason to introduce that as a second variable here; keeping
the original classifier makes this directly comparable to v95's own
HQ-val numbers, isolating the pool as the one changed thing.

Otherwise identical to v95's own ablate_cards_celeba_masking_hybrid_
threshold.py: demean_query=True, orthogonalize=True, K=50, SigLIP, the
7-strategy fill family, same 8 configs (top_pct 2.5-30 + Otsu).
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
from run_cards_celeba_masking_hybrid_official_val_zscore import build_clean_official_val_paths

from cards.attribution.localization import localize_concept, threshold_mask
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS, TARGET_CLASSES
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

CELEBA_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebA\celeba")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
SEED = 42
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}

CONFIGS = [
    ("top_pct_2.5", "top_pct", 2.5),
    ("top_pct_5", "top_pct", 5),
    ("top_pct_10", "top_pct", 10),
    ("top_pct_15", "top_pct", 15),
    ("top_pct_20", "top_pct", 20),
    ("top_pct_25", "top_pct", 25),
    ("top_pct_30", "top_pct", 30),
    ("otsu", "otsu", None),
]


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

    official_paths = build_clean_official_val_paths()
    cfg.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}
    cfg.pool_source = "val"
    pairs = [(p, 0) for p in official_paths]
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images", flush=True)

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()

    raw_queries = {
        c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], encoder), text_center)
        for c in GROUNDABLE_CONCEPTS
    }
    queries = orthogonalize_queries(raw_queries)

    # concept-outer / config-inner (NOT config-outer like an earlier
    # version of this script and its v95 sibling) -- retrieval and
    # localization depend only on (encoder, image, t_c), never on the
    # threshold, so recomputing them once per config (8x total) was pure
    # waste; caching them once per concept and reusing across all 8
    # configs (mirrors the z-score script's own concept-outer/k-inner
    # design) removes that redundant encoder forward pass without
    # changing any score -- only the mask/fill/encode/classify step
    # below genuinely differs per config and must still run once each.
    hybrid_scores_by_config: dict[str, dict[str, dict[tuple[int, int], float]]] = {
        label: {t: {} for t in TARGET_CLASSES} for label, _, _ in CONFIGS
    }
    all_rows = []  # (label, concept_name, target_task, hybrid_raw_score)
    skip_rows = []  # (label, concept_name, n_present, n_skipped_degenerate)

    for c_i, concept_name in enumerate(GROUNDABLE_CONCEPTS):
        t_c = queries[concept_name]
        t_c_dev = t_c.to(DEVICE)
        concept_idx = CONCEPT_TO_IDX[concept_name]

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
        cached = []  # (idx, image, sim_map)
        for idx in present_indices:
            image = Image.open(pool.paths[idx]).convert("RGB")
            sim_map = localize_concept(encoder, image, t_c, (image.height, image.width))
            cached.append((idx, image, sim_map))

        print(f"[{c_i + 1:>2d}/{len(GROUNDABLE_CONCEPTS)}] {concept_name:<20s}", flush=True)

        for label, threshold_method, top_pct in CONFIGS:
            delta_logits = {t: [] for t in TARGET_CLASSES}
            n_skipped = 0

            for idx, image, sim_map in cached:
                mask = threshold_mask(sim_map, top_pct=top_pct or 15, method=threshold_method)
                if not mask.any() or mask.all():
                    n_skipped += 1
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

            skip_rows.append((label, concept_name, len(present_indices), n_skipped))
            for task_name in TARGET_CLASSES:
                score = float(np.mean(delta_logits[task_name])) if delta_logits[task_name] else 0.0
                hybrid_scores_by_config[label][task_name][(concept_idx, 1)] = score
                all_rows.append((label, concept_name, task_name, score))

    results = []  # (label, task_name, rho_result, sign_result)
    for label, threshold_method, top_pct in CONFIGS:
        print(f"\n=== {label} (method={threshold_method}, top_pct={top_pct}) ===", flush=True)
        hybrid_scores_by_task = hybrid_scores_by_config[label]
        for task_name in TARGET_CLASSES:
            rho_result = score_method_agreement(records_by_task[task_name], hybrid_scores_by_task[task_name])
            sign_result = score_sign_agreement(records_by_task[task_name], hybrid_scores_by_task[task_name])
            results.append((label, task_name, rho_result, sign_result))
            if rho_result is not None:
                print(f"  [{task_name}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} "
                      f"p={rho_result.spearman_p:.4g} | sign={sign_result.agreement_frac:.1%} "
                      f"({sign_result.n_agree}/{sign_result.n_pairs}) binom_p={sign_result.binom_p:.4g}", flush=True)

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_threshold_official_val_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "target_task", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for label, task_name, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([label, task_name, "", "", "", "", "", ""])
            else:
                writer.writerow([label, task_name, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_threshold_official_val_ablation_raw_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "concept_name", "target_task", "hybrid_raw_score"])
        writer.writerows(all_rows)

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_threshold_official_val_ablation_skip_counts.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "concept_name", "n_present", "n_skipped_degenerate"])
        writer.writerows(skip_rows)

    print(f"\nSaved {len(results)} (config, task) rows to results/cards_celeba_masking_hybrid_threshold_official_val_ablation.csv")


if __name__ == "__main__":
    main()
