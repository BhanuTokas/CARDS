"""Re-scores the masking hybrid (k=1.0 z-score, the best-known config)
against the LOW-RESOLUTION-trained classifier
(celeba_attractive_young_lowres), on BOTH image pools -- the resolution-
mismatch follow-up (notes/celeba_correlation_investigation.md). Localization/
retrieval (SigLIP) is completely UNCHANGED from every other config in this
track -- only which classifier's logits get read out for delta_logits
differs, since that's the one thing this whole follow-up is testing.

Pool A: CelebA-HQ's own 4,500-image val split (same present-set retrieval
as every other HQ-pool config), scored against the newly-regenerated
`results/celeba_full_faithfulness_lowres.csv` ground truth (same
classifier, same real masks, same 100-per-pair methodology).

Pool B: standard CelebA's own official val partition, minus HQ-train
(leakage) and HQ-val (already tested) overlap -- the same 16,874-image
"clean" pool run_cards_celeba_masking_hybrid_official_val_zscore.py
already built and cached -- scored against the SAME lowres ground truth
(the only ground truth available; no real masks exist for these 16,874
images themselves).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT, TASK_POSITIVE_LOGIT_INDEX
from run_cards_celeba_masking_hybrid_official_val_zscore import build_clean_official_val_paths

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
CELEBA_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebA\celeba")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
SEED = 42
Z_K = 1.0
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}


def load_records_by_task(csv_name: str) -> dict[str, list[FaithfulnessResult]]:
    by_task: dict[str, list[FaithfulnessResult]] = {t: [] for t in TARGET_CLASSES}
    with open(RESULTS_DIR / csv_name, newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]].append(FaithfulnessResult(
                image=row["image"], concept_number=CONCEPT_TO_IDX[row["concept_name"]], category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return by_task


def score_pool(encoder, queries, pool, native_model, task_positive_logit_index) -> dict[str, dict[tuple[int, int], float]]:
    scores_by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    for c_i, concept_name in enumerate(GROUNDABLE_CONCEPTS):
        t_c = queries[concept_name]
        t_c_dev = t_c.to(DEVICE)
        concept_idx = CONCEPT_TO_IDX[concept_name]

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)

        cached = []
        for idx in present_indices:
            image = Image.open(pool.paths[idx]).convert("RGB")
            sim_map = localize_concept(encoder, image, t_c, (image.height, image.width))
            cached.append((idx, image, sim_map))
        cutoff = concept_zscore_cutoff([sm for _, _, sm in cached], Z_K)

        delta_logits = {t: [] for t in TARGET_CLASSES}
        n_skipped = 0
        for idx, image, sim_map in cached:
            mask = threshold_mask(sim_map, method="fixed", cutoff=cutoff)
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

            pixels_orig = BACKBONES["celeba_attractive_young_lowres"].preprocess(image).unsqueeze(0)
            pixels_masked = BACKBONES["celeba_attractive_young_lowres"].preprocess(masked_image).unsqueeze(0)
            batch = torch.cat([pixels_orig, pixels_masked], dim=0).to(DEVICE)
            with torch.no_grad():
                logits = native_model(batch)

            for task_name in TARGET_CLASSES:
                task_idx = task_positive_logit_index[task_name]
                delta_logits[task_name].append((logits[0, task_idx] - logits[1, task_idx]).item())

        for task_name in TARGET_CLASSES:
            score = float(np.mean(delta_logits[task_name])) if delta_logits[task_name] else 0.0
            scores_by_task[task_name][(concept_idx, 1)] = score

        print(f"  [{c_i + 1:>2d}/{len(GROUNDABLE_CONCEPTS)}] {concept_name:<20s} "
              f"n={len(present_indices) - n_skipped:>3d} skipped={n_skipped}", flush=True)
    return scores_by_task


def evaluate_and_report(label: str, records_by_task, scores_by_task) -> list[tuple[str, str, object, object]]:
    print(f"\n=== {label} ===", flush=True)
    rows = []
    for task_name in TARGET_CLASSES:
        rho_result = score_method_agreement(records_by_task[task_name], scores_by_task[task_name])
        sign_result = score_sign_agreement(records_by_task[task_name], scores_by_task[task_name])
        rows.append((label, task_name, rho_result, sign_result))
        if rho_result is not None:
            print(f"  [{task_name}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} "
                  f"p={rho_result.spearman_p:.4g} | sign={sign_result.agreement_frac:.1%} "
                  f"({sign_result.n_agree}/{sign_result.n_pairs}) binom_p={sign_result.binom_p:.4g}", flush=True)
    return rows


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    records_by_task = load_records_by_task("celeba_full_faithfulness_lowres.csv")

    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    raw_queries = {
        c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], encoder), text_center)
        for c in GROUNDABLE_CONCEPTS
    }
    queries = orthogonalize_queries(raw_queries)

    spec = BACKBONES["celeba_attractive_young_lowres"]
    native_model = spec.load_native().to(DEVICE).eval()

    all_results = []

    # Pool A: CelebA-HQ's own val split (same pool/retrieval as every other config).
    cfg.dataset = {"name": "celeba", "root": str(CELEBA_HQ_ROOT)}
    cfg.pool_source = "val"
    hq_pairs = load_celeba(CELEBA_HQ_ROOT, split="val")
    hq_pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), hq_pairs, encoder)
    print(f"HQ-val pool: {len(hq_pool.paths)} images", flush=True)
    print("\n--- scoring HQ-val pool ---", flush=True)
    hq_scores = score_pool(encoder, queries, hq_pool, native_model, TASK_POSITIVE_LOGIT_INDEX)
    all_results += evaluate_and_report("lowres_classifier x HQ-val pool", records_by_task, hq_scores)

    # Pool B: the already-built clean official-val pool (leakage/overlap excluded).
    official_paths = build_clean_official_val_paths()
    cfg.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}
    cfg.pool_source = "val"
    official_pairs = [(p, 0) for p in official_paths]
    official_pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), official_pairs, encoder)
    print(f"\nofficial-val pool: {len(official_pool.paths)} images", flush=True)
    print("\n--- scoring official-val pool ---", flush=True)
    official_scores = score_pool(encoder, queries, official_pool, native_model, TASK_POSITIVE_LOGIT_INDEX)
    all_results += evaluate_and_report("lowres_classifier x official-val pool", records_by_task, official_scores)

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_lowres_classifier_k1_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pool", "target_task", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for label, task_name, rho_result, sign_result in all_results:
            if rho_result is None:
                writer.writerow([label, task_name, "", "", "", "", "", ""])
            else:
                writer.writerow([label, task_name, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])
    print(f"\nSaved {len(all_results)} rows to results/cards_celeba_masking_hybrid_lowres_classifier_k1_ablation.csv")


if __name__ == "__main__":
    main()
