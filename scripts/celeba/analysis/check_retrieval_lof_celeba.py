"""Density-based "identify the positive cluster" attempt, prompted
directly ("Can we try some density based metric to identify the positive
set?"), after Otsu/GMM/max-gap all failed for the same reason (see
check_retrieval_bimodality_celeba.py, check_retrieval_maxgap_gmm_celeba.py,
check_maxgap_confound_celeba.py) -- all three operate on the 1D
similarity-to-t_c SCALAR, which is genuinely unimodal/smooth for these
concepts, so there's no gap/valley/second-mode to find in that collapsed
signal no matter how it's sliced.

This tries a fundamentally different signal: Local Outlier Factor (LOF)
on the FULL embedding vectors of a wide candidate band (top CANDIDATE_N
by similarity to t_c), not the 1D projection. Idea: genuine concept-
positive images should cluster tightly together in embedding space
(similar faces look similar beyond just their t_c-alignment) and
mutually reinforce each other's local density, while near-miss retrieval
errors -- competitive on raw similarity alone -- should sit more
isolated. LOF < ~1 means "denser than its neighbors" (core-cluster-like);
LOF > ~1 means "sparser than its neighbors" (outlier-like).

Same evaluation as the prior three checks: for each of the same 6
concepts, does taking the min(l, K) LOF-selected "core" images actually
improve precision@selected against REAL CelebA attribute labels, versus
flat top-K -- and does l vary sensibly with true concept prevalence
(the same tell that damned max-gap: constant tiny l regardless of
prevalence is a red flag, not a good sign).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.neighbors import LocalOutlierFactor

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.celeba_attributes import load_attribute_labels, load_attribute_names
from cards.data.datasets import load_celeba
from cards.pipeline import instantiate_encoder
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
CANDIDATE_N = 300  # wide band to search within, same order of magnitude as max-gap's MAX_RANK
N_NEIGHBORS = 20  # LOF's own neighborhood size

CONCEPTS_TO_CHECK = [
    ("Eyeglasses", "clean"),
    ("Smiling", "clean"),
    ("Wavy_Hair", "clean"),
    ("Big_Nose", "severe dilution"),
    ("Narrow_Eyes", "broken at every K"),
    ("Pale_Skin", "broken at every K"),
]


def main():
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
    print(f"pool: {len(pool.paths)} images  (candidate band={CANDIDATE_N}, LOF n_neighbors={N_NEIGHBORS})\n", flush=True)

    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    pool_labels = np.array([attr_labels_by_file[f"{Path(p).stem}.jpg"] for p in pool.paths])
    embeddings = pool.embeddings.numpy()

    for concept_name, tag in CONCEPTS_TO_CHECK:
        t_c = build_concept_query(CONCEPT_QUERY_TEXT[concept_name], encoder)
        t_c = demean_query(t_c, text_center)
        sims = (pool.embeddings @ t_c).numpy()

        attr_idx = attr_names.index(concept_name)
        true_positive_mask = pool_labels[:, attr_idx].astype(bool)
        n_true_total = int(true_positive_mask.sum())

        order = np.argsort(-sims)
        precision_topk = float(true_positive_mask[order[:K]].mean())

        candidate_idx = order[:CANDIDATE_N]
        candidate_emb = embeddings[candidate_idx]

        lof = LocalOutlierFactor(n_neighbors=N_NEIGHBORS, metric="cosine")
        lof.fit_predict(candidate_emb)
        # negative_outlier_factor_: closer to -1 = denser/core, more negative = more outlier-like.
        # LOF "inlier score" -- rank candidates by how CORE (dense) they are, most-core first.
        core_order_within_candidates = np.argsort(-lof.negative_outlier_factor_)  # least negative (densest) first
        density_ranked_idx = candidate_idx[core_order_within_candidates]

        # l = largest gap in the (sorted-descending) LOF density score itself,
        # searched over the same style of window, reusing the density signal
        # analogous to how max-gap used the similarity signal.
        density_scores_sorted = -np.sort(-lof.negative_outlier_factor_[core_order_within_candidates])
        window_lo, window_hi = 5, min(200, CANDIDATE_N - 1)
        gaps = density_scores_sorted[window_lo:window_hi] - density_scores_sorted[window_lo + 1 : window_hi + 1]
        l = window_lo + int(np.argmax(gaps)) + 1
        n_select = min(l, K)

        precision_density = float(true_positive_mask[density_ranked_idx[:n_select]].mean())

        print(f"=== {concept_name}  ({tag}, n_true={n_true_total}/{len(pool.paths)}) ===")
        print(f"  LOF density-selected: l={l}  n_used={n_select}  precision={precision_density:.3f}   "
              f"vs. flat top-K={K} precision={precision_topk:.3f}")
        print()


if __name__ == "__main__":
    main()
