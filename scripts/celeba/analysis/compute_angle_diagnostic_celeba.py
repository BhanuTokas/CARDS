"""The cos(mean(P_c)-mean(N_c), t_c) angle diagnostic (CUB v45/v46) had
never been computed for CelebA -- prompted directly ("What was the
average theta angle for P_C-N_C vs concept vector in CelebA?"). Computed
here for CelebA's production config (SigLIP, K=50, aligned_retrieval,
demean_query=True), all 26 groundable concepts (the angle depends only
on the concept's own retrieval, not on which target task is being
scored, so this is 26 values, not 52).

Cheap: pure embedding-space geometry (retrieval + `estimate_direction`),
no native-model forward passes needed -- doesn't touch the black box at
all, unlike every other script in this track.
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
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.data.datasets import load_celeba
from cards.directions.estimate import estimate_direction
from cards.pipeline import instantiate_encoder
from cards.retrieval.aligned import aligned_retrieval
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
    })
    encoder = instantiate_encoder(cfg)
    cfg.dataset = {"name": "celeba", "root": str(CELEBA_HQ_ROOT)}
    cfg.pool_source = "val"
    pairs = load_celeba(CELEBA_HQ_ROOT, split="val")
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images", flush=True)

    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    rows = []
    for concept_name in GROUNDABLE_CONCEPTS:
        query_text = CONCEPT_QUERY_TEXT[concept_name]
        t_c = build_concept_query(query_text, encoder)
        t_c = demean_query(t_c, text_center)

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
        absent_indices = aligned_retrieval(pool, present_indices, t_c, K)

        direction = estimate_direction(concept_name, pool.embeddings[present_indices], pool.embeddings[absent_indices])
        cos_sim = float(torch.clamp(t_c @ direction.unit_vector, -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos_sim)))
        rows.append((concept_name, angle))
        print(f"{concept_name:<20s}: angle={angle:.2f} deg", flush=True)

    vals = np.array([a for _c, a in rows])
    print(f"\nmean={vals.mean():.2f} deg  std={vals.std():.2f} deg  n={len(vals)}", flush=True)

    with open(RESULTS_DIR / "cards_celeba_angle_diagnostic.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "angle_degrees"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} angles to results/cards_celeba_angle_diagnostic.csv")


if __name__ == "__main__":
    main()
