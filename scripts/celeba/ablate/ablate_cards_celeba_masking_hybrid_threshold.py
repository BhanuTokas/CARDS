"""Ablation of the masking hybrid's own localization threshold --
top_pct value AND Otsu's method -- prompted directly ("Where was the 15%
threshold decided on?" -> "Can we run an ablation with Otsu's and at
different PCT level?").

TOP_PCT=15 was never actually tuned on CelebA: it was picked as a
plausible round number when the pseudo-mask was first built (v79), and
only ever checked indirectly via threshold-free AUROC (v78) -- Dice
against real masks (which WOULD penalize a wrong area budget) was only
ever computed on CUB (v69), where it exposed TOP_PCT=15 as ~150x
oversized for tiny body parts. This ablation finally sweeps the value on
CelebA itself, scored the same way every other axis ablation in this
track has been (v86-v93): full 26-concept x 2-task hybrid re-score
against the real masking-based faithfulness ground truth
(`results/celeba_full_faithfulness.csv`), holding every other setting at
the v82 baseline (demean_query=True, orthogonalize=False, K=50, SigLIP,
the same 7-strategy fill family).

top_pct=15 itself is NOT recomputed -- v82's own existing raw scores
(`results/cards_celeba_masking_hybrid_scores_best_of_family_full.csv`)
already ARE that data point, reused as baseline rather than redone,
matching this track's own established discipline (e.g. the query
ablation script's demean=True reuse).

Uses the NEW integrated library code (cards.attribution.localization,
already unit-tested and pipeline-wired -- see notes/
celeba_correlation_investigation.md v94) for localization/thresholding
specifically, rather than re-duplicating that logic inline a 12th time
-- this ablation doubles as a fresh real-usage exercise of that code.

6 configs (top_pct in [5, 10, 20, 25, 30] + otsu) x 26 concepts x 50
images each -- the same per-image cost as one full hybrid run, x6.
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

from cards.attribution.localization import localize_concept, threshold_mask
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
SEED = 42
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}

# (label, threshold_method, top_pct) -- top_pct is ignored by "otsu".
# 15 is NOT included: v82's existing raw scores already are that point.
CONFIGS = [
    ("top_pct_5", "top_pct", 5),
    ("top_pct_10", "top_pct", 10),
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


def load_baseline_top_pct_15() -> dict[str, dict[tuple[int, int], float]]:
    """v82's existing top_pct=15 raw scores, reused (not recomputed) as
    this sweep's own middle data point."""
    scores_by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_scores_best_of_family_full.csv", newline="") as f:
        for row in csv.DictReader(f):
            concept_idx = CONCEPT_TO_IDX[row["concept_name"]]
            scores_by_task[row["target_task"]][(concept_idx, 1)] = float(row["hybrid_raw_score"])
    return scores_by_task


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

    # Every config uses the SAME queries (demean=True/orth=False, v82's
    # own baseline) -- only threshold_method/top_pct varies -- so build
    # them once, outside the config loop.
    queries: dict[str, torch.Tensor] = {}
    for concept_name in GROUNDABLE_CONCEPTS:
        t_c = build_concept_query(CONCEPT_QUERY_TEXT[concept_name], encoder)
        queries[concept_name] = demean_query(t_c, text_center)

    results = []  # (label, task_name, rho_result, sign_result)
    all_rows = []  # (label, concept_name, target_task, hybrid_raw_score)
    skip_rows = []  # (label, concept_name, n_present, n_skipped_degenerate)

    for label, threshold_method, top_pct in CONFIGS:
        print(f"\n=== {label} (method={threshold_method}, top_pct={top_pct}) ===", flush=True)
        hybrid_scores_by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}

        for c_i, concept_name in enumerate(GROUNDABLE_CONCEPTS):
            t_c = queries[concept_name]
            t_c_dev = t_c.to(DEVICE)
            concept_idx = CONCEPT_TO_IDX[concept_name]

            present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
            delta_logits = {t: [] for t in TARGET_CLASSES}
            n_skipped = 0

            for idx in present_indices:
                image = Image.open(pool.paths[idx]).convert("RGB")
                sim_map = localize_concept(encoder, image, t_c, (image.height, image.width))
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
                hybrid_scores_by_task[task_name][(concept_idx, 1)] = score
                all_rows.append((label, concept_name, task_name, score))

            print(f"  [{c_i + 1:>2d}/{len(GROUNDABLE_CONCEPTS)}] {concept_name:<20s} "
                  f"n={len(present_indices) - n_skipped:>3d} skipped={n_skipped}", flush=True)

        for task_name in TARGET_CLASSES:
            rho_result = score_method_agreement(records_by_task[task_name], hybrid_scores_by_task[task_name])
            sign_result = score_sign_agreement(records_by_task[task_name], hybrid_scores_by_task[task_name])
            results.append((label, task_name, rho_result, sign_result))
            if rho_result is not None:
                print(f"  [{task_name}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} "
                      f"p={rho_result.spearman_p:.4g} | sign={sign_result.agreement_frac:.1%} "
                      f"({sign_result.n_agree}/{sign_result.n_pairs}) binom_p={sign_result.binom_p:.4g}", flush=True)

    # Fold in the reused, NOT-recomputed top_pct=15 baseline for a complete table.
    baseline_scores_by_task = load_baseline_top_pct_15()
    print("\n=== top_pct_15 (method=top_pct, top_pct=15) -- REUSED from v82, not recomputed ===", flush=True)
    for task_name in TARGET_CLASSES:
        rho_result = score_method_agreement(records_by_task[task_name], baseline_scores_by_task[task_name])
        sign_result = score_sign_agreement(records_by_task[task_name], baseline_scores_by_task[task_name])
        results.append(("top_pct_15", task_name, rho_result, sign_result))
        if rho_result is not None:
            print(f"  [{task_name}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} "
                  f"p={rho_result.spearman_p:.4g} | sign={sign_result.agreement_frac:.1%} "
                  f"({sign_result.n_agree}/{sign_result.n_pairs}) binom_p={sign_result.binom_p:.4g}", flush=True)

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_threshold_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "target_task", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for label, task_name, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([label, task_name, "", "", "", "", "", ""])
            else:
                writer.writerow([label, task_name, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_threshold_ablation_raw_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "concept_name", "target_task", "hybrid_raw_score"])
        writer.writerows(all_rows)

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_threshold_ablation_skip_counts.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "concept_name", "n_present", "n_skipped_degenerate"])
        writer.writerows(skip_rows)

    print(f"\nSaved {len(results)} (config, task) rows to results/cards_celeba_masking_hybrid_threshold_ablation.csv")


if __name__ == "__main__":
    main()
