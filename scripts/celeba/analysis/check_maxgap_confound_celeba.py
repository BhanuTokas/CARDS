"""Disentangles max-gap's apparent "win" (check_retrieval_maxgap_gmm_
celeba.py) from a tautology, prompted directly ("Let's test that!"):
precision@k is mechanically non-increasing as k shrinks for any
reasonably-good ranking, so "max-gap's l beat flat top-K=50" could just
be "any small l beats k=50" in disguise, not evidence the gap it found
is a meaningful boundary.

For each of the same 6 concepts, prints precision@k across a FIXED range
of k (5, 7, 10, 15, 20, 24, 30, 40, 50) with no gap-finding involved at
all, alongside where max-gap's own chosen l lands on that same curve. If
precision@l_gap is just an unremarkable point on an otherwise-smooth,
monotonic precision-vs-k curve, max-gap adds nothing beyond "pick a
small k" -- if it sits at a genuine local plateau/edge (a real drop in
precision right after l_gap), that's evidence it found something real.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from check_retrieval_maxgap_gmm_celeba import CONCEPTS_TO_CHECK, MAX_RANK, MIN_RANK, max_gap_l
from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.celeba_attributes import load_attribute_labels, load_attribute_names
from cards.data.datasets import load_celeba
from cards.pipeline import instantiate_encoder
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_FIXED = [5, 7, 10, 15, 20, 24, 30, 40, 50]


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
    print(f"pool: {len(pool.paths)} images  (max-gap search window: ranks {MIN_RANK}-{MAX_RANK})\n", flush=True)

    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    pool_labels = np.array([attr_labels_by_file[f"{Path(p).stem}.jpg"] for p in pool.paths])

    for concept_name, tag in CONCEPTS_TO_CHECK:
        t_c = build_concept_query(CONCEPT_QUERY_TEXT[concept_name], encoder)
        t_c = demean_query(t_c, text_center)
        sims = (pool.embeddings @ t_c).numpy()

        attr_idx = attr_names.index(concept_name)
        true_positive_mask = pool_labels[:, attr_idx].astype(bool)
        n_true_total = int(true_positive_mask.sum())

        order = np.argsort(-sims)
        sims_sorted = sims[order]
        l_gap = max_gap_l(sims_sorted)

        print(f"=== {concept_name}  ({tag}, n_true={n_true_total}/{len(pool.paths)}, max-gap l={l_gap}) ===")
        row = []
        for k in sorted(set(K_FIXED + [l_gap])):
            precision = float(true_positive_mask[order[:k]].mean())
            marker = "  <- max-gap's l" if k == l_gap else ""
            row.append(f"  P@{k:<3d} = {precision:.3f}{marker}")
        print("\n".join(row))
        print()


if __name__ == "__main__":
    main()
