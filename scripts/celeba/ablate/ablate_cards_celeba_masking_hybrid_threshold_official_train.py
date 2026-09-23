"""z-score threshold (alpha) ablation for the masking hybrid on the
official-train CelebA classifier (ResNet18) -- prompted directly
("Can you also do separate ablations on K, alpha and prompt variations
with the official set?").

Fixed config: SigLIP, orthogonalize=True, demean_query=**False**, K=50
-- the NEW best-known setting for this classifier (see `ablate_cards_
celeba_masking_hybrid_k_official_train.py`'s own docstring for the
full reasoning).

alpha=1.0's own cell is REUSED, not recomputed -- exactly the (siglip,
orth=True, demean=False) cell from the joint encoder ablation. Only
alpha in {0.5, 1.5, 2.0, 2.5} are computed fresh.

Hand-rolled (concept-outer, alpha-inner), NOT `cards.pipeline`'s
`process_concept` -- unlike the K/prompt ablations, localization only
depends on (concept, K, encoder, query), not alpha, so retrieval +
patch-similarity localization is cached ONCE per concept and reused
across all 5 alpha thresholds; only the masking/scoring step re-runs
per alpha. Mirrors `ablate_cards_celeba_masking_hybrid_threshold_
zscore_official_val.py`'s own design exactly, adapted to official-train
specifics (backbone, TaskBlackBox, 2 tasks x 2 GT versions).
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT
from run_cards_celeba_masking_hybrid_official_val_zscore import build_clean_official_val_paths

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.concepts.prompts import build_concept_query
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
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

CELEBA_ROOT = Path(os.environ.get("CELEBA_ROOT", r"C:\Users\btokas\Projects\Datasets\CelebA\celeba"))
RESULTS_DIR = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
CACHE_DIR = Path(os.environ.get("CARDS_CACHE_DIR", "embedding_cache"))
BACKBONE_NAME = "celeba_official_train_attractive_male"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
SEED = 42
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
TASKS = ["Attractive", "Male"]
TASK_SLICE = {"Attractive": 1, "Male": 3}
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
GT_VERSIONS = {
    "non_overlapping": "celeba_official_train_faithfulness_non_overlapping.csv",
    "previously_used": "celeba_official_train_faithfulness_previously_used.csv",
}
ALPHA_VALUES_NEW = [0.5, 1.5, 2.0, 2.5]
ALPHA_REUSED = 1.0
REUSED_RAW_CSV = RESULTS_DIR / "cards_celeba_masking_hybrid_joint_orth_demean_encoder_official_train_ablation_raw_scores.csv"


def load_records(gt_filename: str, task_name: str) -> list[FaithfulnessResult]:
    records = []
    with open(RESULTS_DIR / gt_filename, newline="") as f:
        for row in csv.DictReader(f):
            if row["target_task"] != task_name:
                continue
            records.append(FaithfulnessResult(
                image=row["image"], concept_number=CONCEPT_TO_IDX[row["concept_name"]], category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return records


def load_reused_alpha1_scores() -> dict[str, dict[tuple[int, int], float]]:
    scores_by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TASKS}
    with open(REUSED_RAW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["encoder"] != "siglip" or row["orthogonalize"] != "True" or row["demean_query"] != "False":
                continue
            key = (CONCEPT_TO_IDX[row["concept_name"]], 1)
            scores_by_task[row["target_task"]][key] = float(row["hybrid_raw_score"])
    return scores_by_task


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": str(CACHE_DIR),
    })
    encoder = instantiate_encoder(cfg)

    official_paths = build_clean_official_val_paths()
    cfg.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}
    cfg.pool_source = "val"
    pairs = [(p, 0) for p in official_paths]
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images", flush=True)

    spec = BACKBONES[BACKBONE_NAME]
    native_model = spec.load_native().to(DEVICE).eval()

    raw_queries = {c: build_concept_query(CONCEPT_QUERY_TEXT[c], encoder) for c in GROUNDABLE_CONCEPTS}
    queries = orthogonalize_queries(raw_queries)  # orth=True, demean=False

    hybrid_scores_by_alpha: dict[float, dict[str, dict[tuple[int, int], float]]] = {
        alpha: {t: {} for t in TASKS} for alpha in ALPHA_VALUES_NEW
    }
    raw_rows = []  # (alpha, task, concept_name, hybrid_raw_score, source)
    cutoff_rows = []  # (alpha, concept_name, cutoff, mean_area_pct)

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
        sim_maps = [sm for _, _, sm in cached]

        print(f"[{c_i + 1:>2d}/{len(GROUNDABLE_CONCEPTS)}] {concept_name:<20s}", flush=True)

        for alpha in ALPHA_VALUES_NEW:
            cutoff = concept_zscore_cutoff(sim_maps, alpha)
            areas = []
            delta_logits = {t: [] for t in TASKS}
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

                for task_name in TASKS:
                    task_idx = TASK_SLICE[task_name]
                    delta_logits[task_name].append((logits[0, task_idx] - logits[1, task_idx]).item())

            mean_area_pct = 100.0 * float(np.mean(areas)) if areas else 0.0
            cutoff_rows.append((alpha, concept_name, cutoff, mean_area_pct))
            for task_name in TASKS:
                score = float(np.mean(delta_logits[task_name])) if delta_logits[task_name] else 0.0
                hybrid_scores_by_alpha[alpha][task_name][(concept_idx, 1)] = score
                raw_rows.append((alpha, task_name, concept_name, score, "computed"))

            print(f"    alpha={alpha:<4.1f} cutoff={cutoff:+.4f} mean_area={mean_area_pct:>5.1f}% "
                  f"n={len(present_indices) - n_skipped:>3d} skipped={n_skipped}", flush=True)

    all_rows = []  # (alpha, task, gt_version, n_pairs, rho, rho_p, sign_frac, n_agree, sign_p, source)

    print(f"\n{'=' * 20} alpha={ALPHA_REUSED} (reused from joint encoder ablation) {'=' * 20}", flush=True)
    reused_scores = load_reused_alpha1_scores()
    for task_name in TASKS:
        for c, score in reused_scores[task_name].items():
            raw_rows.append((ALPHA_REUSED, task_name, GROUNDABLE_CONCEPTS[c[0]], score, "reused"))
        for gt_name, gt_filename in GT_VERSIONS.items():
            records_task = load_records(gt_filename, task_name)
            rho_r = score_method_agreement(records_task, reused_scores[task_name])
            sign_r = score_sign_agreement(records_task, reused_scores[task_name])
            if rho_r is None:
                continue
            print(f"  [{task_name}/{gt_name}] n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} "
                  f"sign={sign_r.agreement_frac:.1%}", flush=True)
            all_rows.append((ALPHA_REUSED, task_name, gt_name, rho_r.n_pairs, rho_r.spearman_rho, rho_r.spearman_p,
                              sign_r.agreement_frac, sign_r.n_agree, sign_r.binom_p, "reused"))

    for alpha in ALPHA_VALUES_NEW:
        print(f"\n=== alpha={alpha} ===", flush=True)
        for task_name in TASKS:
            for gt_name, gt_filename in GT_VERSIONS.items():
                records_task = load_records(gt_filename, task_name)
                rho_r = score_method_agreement(records_task, hybrid_scores_by_alpha[alpha][task_name])
                sign_r = score_sign_agreement(records_task, hybrid_scores_by_alpha[alpha][task_name])
                if rho_r is None:
                    print(f"  [{task_name}/{gt_name}] too few pairs", flush=True)
                    continue
                print(f"  [{task_name}/{gt_name}] n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} "
                      f"p={rho_r.spearman_p:.4g} | sign={sign_r.agreement_frac:.1%}", flush=True)
                all_rows.append((alpha, task_name, gt_name, rho_r.n_pairs, rho_r.spearman_rho, rho_r.spearman_p,
                                  sign_r.agreement_frac, sign_r.n_agree, sign_r.binom_p, "computed"))

    out_path = RESULTS_DIR / "cards_celeba_masking_hybrid_threshold_official_train_ablation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "target_task", "gt_version", "n_pairs", "spearman_rho", "spearman_p",
                          "sign_agreement", "n_agree", "binom_p", "source"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {out_path}", flush=True)

    raw_path = RESULTS_DIR / "cards_celeba_masking_hybrid_threshold_official_train_ablation_raw_scores.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "target_task", "concept_name", "hybrid_raw_score", "source"])
        writer.writerows(raw_rows)
    print(f"Saved {len(raw_rows)} raw rows to {raw_path}", flush=True)

    cutoff_path = RESULTS_DIR / "cards_celeba_masking_hybrid_threshold_official_train_ablation_cutoffs.csv"
    with open(cutoff_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "concept_name", "cutoff", "mean_area_pct"])
        writer.writerows(cutoff_rows)
    print(f"Saved {len(cutoff_rows)} cutoff rows to {cutoff_path}", flush=True)


if __name__ == "__main__":
    main()
