"""Checks whether P_c gets diluted with more marginal matches as K grows,
prompted directly ("does the P_C for the concept gets diluted at K=50?
Maybe it doesn't have 50 great images for some concepts?") -- v89 found
K=15 beats K=30 beats K=50 on the hybrid's own scores; this tests the
most direct explanation: for some (especially rare) concepts, maybe the
pool doesn't have 50 genuinely well-matching images, so the tail of the
top-50 retrieved set is increasingly weak/wrong matches, diluting the
average signal.

Cheap, no masking/classifier compute needed -- retrieval ranking (already
cached embeddings) against CelebA's own REAL binary attribute labels
(ground truth presence/absence, not a proxy). For each concept: rank all
4,500 val-pool images by cosine similarity to t_c, then compute
precision@K (fraction of the top-K retrieved images that ACTUALLY have
that attribute True per the real label) for K=5/10/15/20/30/40/50. A
declining precision@K as K grows is direct evidence of dilution; a flat
precision@K across K means the top-50 are equally "real" throughout.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba_attributes import (
    GROUNDABLE_CONCEPTS,
    load_attribute_labels,
    load_attribute_names,
)
from cards.data.datasets import load_celeba
from cards.pipeline import instantiate_encoder
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_CHECKPOINTS = [5, 10, 15, 20, 30, 40, 50]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

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

    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    # pool.paths are CelebA-HQ-img/<idx>.jpg -- filename key into attr_labels_by_file is "<idx>.jpg"
    pool_labels = np.array([attr_labels_by_file[f"{Path(p).stem}.jpg"] for p in pool.paths])  # (N, 40)

    rows = []  # (concept_name, K, precision_at_k, n_true_in_pool)
    print(f"\n{'concept':<20s} " + "  ".join(f"P@{k:<3d}" for k in K_CHECKPOINTS) + "   n_true_total")
    for concept_name in GROUNDABLE_CONCEPTS:
        t_c = build_concept_query(CONCEPT_QUERY_TEXT[concept_name], encoder)
        t_c = demean_query(t_c, text_center)

        sims = (pool.embeddings @ t_c).numpy()  # (N,)
        ranked_idx = np.argsort(-sims)  # descending

        attr_idx = attr_names.index(concept_name)
        n_true_total = int(pool_labels[:, attr_idx].sum())

        precisions = []
        for k in K_CHECKPOINTS:
            top_k = ranked_idx[:k]
            n_true = int(pool_labels[top_k, attr_idx].sum())
            precision = n_true / k
            precisions.append(precision)
            rows.append((concept_name, k, precision, n_true_total))

        print(f"{concept_name:<20s} " + "  ".join(f"{p:.2f}" for p in precisions) + f"   {n_true_total}")

    # aggregate mean precision@K across all 26 concepts
    print("\n=== Mean precision@K across all 26 concepts ===")
    for k in K_CHECKPOINTS:
        vals = [p for c, kk, p, _n in rows if kk == k]
        print(f"P@{k:<3d}: mean={np.mean(vals):.3f}  min={np.min(vals):.3f}  max={np.max(vals):.3f}")

    with open(RESULTS_DIR / "celeba_retrieval_dilution_check.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "K", "precision_at_k", "n_true_in_pool"])
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to results/celeba_retrieval_dilution_check.csv")


if __name__ == "__main__":
    main()
