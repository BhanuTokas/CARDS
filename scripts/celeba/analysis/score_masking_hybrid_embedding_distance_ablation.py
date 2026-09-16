"""Embedding-distance normalization for ConceptMask (masking hybrid) on
CelebA -- the OTHER Step 7 formula (cards.attribution.normalization.
embedding_distance_normalize: raw_score / delta_c), tested alongside the
paired-Cohen's-d normalization already ablated in score_masking_hybrid_
normalization_ablation.py, per direct request ("Can you also test
embedding distance normalization and its variants?").

Unlike the paired-Cohen's-d version, `delta_c` (Step 4's own present-vs-
absent embedding-centroid distance) is NOT something ConceptMask computes
or saves anywhere today -- masking_score() only ever touches
present_indices, never absent_indices (score_masking_hybrid_concepts's
own docstring: "ignoring absent_indices entirely"). So this script
computes delta_c fresh, per concept, via the SAME retrieval Steps 2-4
would have used (top-K present / aligned-retrieval absent from the
CelebAMask-HQ val pool, orthogonalized+demeaned query -- matching
local_attribution_comparison_celeba_corrected_gt.py's own "HQ-val,
alpha=1.0, K=50" config, the same source as the raw_score/hybrid_score
being normalized here). This is CHEAP relative to the full masking-
hybrid pipeline -- pure embedding-space centroid math on an already-
cached pool, no localization/masking/black-box inference at all.

Two variants of delta_c, both from cards.attribution.normalization:
  - Euclidean: ||centroid(present) - centroid(absent)||
  - Angular:   1 - cosine_similarity(centroid(present), centroid(absent))
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.attribution.normalization import angular_distance, embedding_distance_normalize
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS, TARGET_CLASSES
from cards.data.datasets import load_celeba
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.retrieval.aligned import aligned_retrieval
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50


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


def load_raw_scores() -> dict[str, dict]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    with open(RESULTS_DIR / "local_attribution_celeba_pairs_corrected_gt.csv", newline="") as f:
        for row in csv.DictReader(f):
            grouped[(row["concept_name"], row["target_task"])].append(float(row["hybrid_score"]))
    raw: dict[str, dict] = {t: {} for t in TARGET_CLASSES}
    for (concept_name, task_name), deltas in grouped.items():
        raw[task_name][(CONCEPT_TO_IDX[concept_name], 1)] = statistics.mean(deltas)
    return raw


def report(label: str, records_by_task, scores_by_task, method_threshold: float = 0.0):
    print(f"\n=== {label} ===", flush=True)
    for task_name in TARGET_CLASSES:
        rho_r = score_method_agreement(records_by_task[task_name], scores_by_task[task_name])
        if rho_r is None:
            print(f"  [{task_name}] too few pairs")
            continue
        sign_r = score_sign_agreement(records_by_task[task_name], scores_by_task[task_name], method_threshold=method_threshold)
        print(f"  [{task_name}] n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} (p={rho_r.spearman_p:.4g})  "
              f"pearson_r={rho_r.pearson_r:+.4f} (p={rho_r.pearson_p:.4g})  "
              f"sign={sign_r.agreement_frac:.1%} ({sign_r.n_agree}/{sign_r.n_pairs}, p={sign_r.binom_p:.4g})", flush=True)


def main():
    new_records = load_records_by_task("celeba_full_faithfulness_attribute_conditioned.csv")
    raw_scores = load_raw_scores()

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
    celeba_pairs = load_celeba(CELEBA_HQ_ROOT, split="val")
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), celeba_pairs, encoder)
    print(f"pool: {len(pool.paths)} images", flush=True)

    demeaned_queries = {c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], encoder), text_center)
                        for c in GROUNDABLE_CONCEPTS}
    orthogonalized_queries = orthogonalize_queries(demeaned_queries)

    print("\nComputing delta_c (present/absent centroid distance) per concept...", flush=True)
    delta_c_euclidean: dict[str, float] = {}
    delta_c_angular: dict[str, float] = {}
    for concept_name in GROUNDABLE_CONCEPTS:
        t_c = orthogonalized_queries[concept_name]
        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
        absent_indices = aligned_retrieval(pool, present_indices, t_c, K)
        present_centroid = pool.embeddings[present_indices].mean(dim=0)
        absent_centroid = pool.embeddings[absent_indices].mean(dim=0)
        delta_c_euclidean[concept_name] = float((present_centroid - absent_centroid).norm())
        delta_c_angular[concept_name] = angular_distance(present_centroid, absent_centroid)
        print(f"  {concept_name:<20s} delta_c_euclidean={delta_c_euclidean[concept_name]:.4f}  "
              f"delta_c_angular={delta_c_angular[concept_name]:.4f}", flush=True)

    normalized_euclidean: dict[str, dict] = {t: {} for t in TARGET_CLASSES}
    normalized_angular: dict[str, dict] = {t: {} for t in TARGET_CLASSES}
    n_skipped = 0
    for task_name in TARGET_CLASSES:
        for key, raw_score in raw_scores[task_name].items():
            concept_idx = key[0]
            concept_name = GROUNDABLE_CONCEPTS[concept_idx]
            try:
                normalized_euclidean[task_name][key] = embedding_distance_normalize(raw_score, delta_c_euclidean[concept_name])
                normalized_angular[task_name][key] = embedding_distance_normalize(raw_score, delta_c_angular[concept_name])
            except ValueError:
                n_skipped += 1
    if n_skipped:
        print(f"\n({n_skipped} (concept,task) pairs skipped -- delta_c ~zero)", flush=True)

    print("\n############## ConceptMask embedding-distance normalization ablation, corrected ground truth ##############")
    report("raw_score = mean(delta_scores)", new_records, raw_scores)
    report("normalized_score = raw_score / delta_c_euclidean", new_records, normalized_euclidean)
    report("normalized_score = raw_score / delta_c_angular", new_records, normalized_angular)


if __name__ == "__main__":
    main()
