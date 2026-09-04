"""End-to-end check of the max-gap adaptive present-set idea (prompted
directly, "Can we do a quick run using Max-gap strategy on the complete
pipeline. I want to see if it improves the rho or not?") against the
masking hybrid's best-known HQ-val config exactly (alpha=1.0 z-score,
orthogonalize=True, K=50, SigLIP -- v96, rho=+0.6150 Attractive) --
mirrors `ablate_cards_celeba_masking_hybrid_threshold_zscore.py` (the
script that produced that baseline) with ONE change: present_indices
comes from max-gap-adaptive selection (`min(l, K)`, l = the largest
local drop in sorted similarity within ranks [MIN_RANK, MAX_RANK])
instead of a flat `retrieve_top_bottom_k(..., K)`.

Diagnostic context this follows up on (`check_retrieval_maxgap_gmm_
celeba.py` + `check_maxgap_confound_celeba.py`): max-gap's precision@K
"win" over flat top-K turned out to be a tautology of shrinking k on an
already-good ranking, not evidence of a real cluster boundary -- for
"clean" concepts precision was flat 1.0 across k=5-50 (nothing to find),
for "broken" concepts (Narrow_Eyes, Pale_Skin) the wins were noise at
tiny denominators, and for Big_Nose max-gap's own l was WORSE than just
using a smaller fixed k. This script tests directly whether that holds
at the level that actually matters -- correlation with real masking-
based faithfulness -- rather than reasoning it through from the proxy
diagnostic alone.
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
ALPHA = 1.0  # the best-known config's own threshold (v96)
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
MIN_RANK, MAX_RANK = 5, 300  # identical window to check_retrieval_maxgap_gmm_celeba.py


def max_gap_present_indices(pool, t_c: torch.Tensor, k_cap: int) -> list[int]:
    """min(l, k_cap) present indices, where l = 1 + the rank at which the
    largest consecutive similarity drop occurs within [MIN_RANK, MAX_RANK]
    -- identical logic to check_retrieval_maxgap_gmm_celeba.py's
    max_gap_l, inlined here to keep this script self-contained."""
    sims = (pool.embeddings @ t_c).numpy()
    order = np.argsort(-sims)
    sims_sorted = sims[order]
    window = sims_sorted[MIN_RANK : MAX_RANK + 1]
    gaps = window[:-1] - window[1:]
    l = MIN_RANK + int(np.argmax(gaps)) + 1
    n_select = min(l, k_cap)
    return order[:n_select].tolist(), l


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

    hybrid_scores: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    all_rows = []  # (concept_name, target_task, hybrid_raw_score)
    l_rows = []  # (concept_name, l, n_used)

    for c_i, concept_name in enumerate(GROUNDABLE_CONCEPTS):
        t_c = queries[concept_name]
        t_c_dev = t_c.to(DEVICE)
        concept_idx = CONCEPT_TO_IDX[concept_name]

        present_indices, l = max_gap_present_indices(pool, t_c, K)
        l_rows.append((concept_name, l, len(present_indices)))

        cached = []
        for idx in present_indices:
            image = Image.open(pool.paths[idx]).convert("RGB")
            sim_map = localize_concept(encoder, image, t_c, (image.height, image.width))
            cached.append((idx, image, sim_map))
        sim_maps = [sm for _, _, sm in cached]
        cutoff = concept_zscore_cutoff(sim_maps, ALPHA)

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
            hybrid_scores[task_name][(concept_idx, 1)] = score
            all_rows.append((concept_name, task_name, score))

        print(f"[{c_i + 1:>2d}/{len(GROUNDABLE_CONCEPTS)}] {concept_name:<20s} l={l:<4d} "
              f"n_used={len(present_indices):<3d} skipped={n_skipped}", flush=True)

    print("\n=== max-gap-selected present-set sizes (l, capped at K=50) ===")
    for concept_name, l, n_used in l_rows:
        print(f"  {concept_name:<20s} l={l:<4d} n_used={n_used}")
    print(f"  mean l={np.mean([l for _, l, _ in l_rows]):.1f}  "
          f"median l={np.median([l for _, l, _ in l_rows]):.1f}  "
          f"(flat K=50 baseline always used n_used=50)")

    print("\n=== max-gap retrieval vs. real faithfulness ground truth ===")
    results = []
    for task_name in TARGET_CLASSES:
        rho_result = score_method_agreement(records_by_task[task_name], hybrid_scores[task_name])
        sign_result = score_sign_agreement(records_by_task[task_name], hybrid_scores[task_name])
        results.append((task_name, rho_result, sign_result))
        if rho_result is not None:
            print(f"  [{task_name}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} "
                  f"p={rho_result.spearman_p:.4g} | sign={sign_result.agreement_frac:.1%} "
                  f"({sign_result.n_agree}/{sign_result.n_pairs}) binom_p={sign_result.binom_p:.4g}", flush=True)

    print("\n=== baseline for comparison (flat K=50, v96) ===")
    print("  Attractive: rho=+0.6150 (p=0.00083)  sign=84.6% (p=0.00053)")
    print("  Young:      (see results/cards_celeba_masking_hybrid_threshold_zscore_orthogonalize_ablation.csv, alpha=1.0 row)")

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_maxgap_retrieval.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_task", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for task_name, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([task_name, "", "", "", "", "", ""])
            else:
                writer.writerow([task_name, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_maxgap_retrieval_raw_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "target_task", "hybrid_raw_score"])
        writer.writerows(all_rows)

    print(f"\nSaved results to results/cards_celeba_masking_hybrid_maxgap_retrieval.csv")


if __name__ == "__main__":
    main()
