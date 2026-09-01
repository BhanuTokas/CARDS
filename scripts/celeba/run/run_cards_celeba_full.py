"""Phase 7 scale-up of the CelebA plan: runs CARDS against ALL 26
groundable concepts (cards.data.celeba_attributes.GROUNDABLE_CONCEPTS),
against BOTH target classes -- extends run_cards_celeba_pilot.py's own
8-concept pilot (v70) the same way CUB's 8-part CARDS run scaled to its
87-attribute bank. Logic and hyperparameters are otherwise IDENTICAL to
the pilot script (aligned_retrieval, K=50, demean_query=True, SigLIP --
still the CUB-carried-over defaults, not yet re-ablated on CelebA, see
that script's own docstring) -- see it for the full design rationale.

Reuses the SAME retrieval pool cache the pilot and Phase 4/6 scripts
already built (identical cfg.dataset/pool_source), so this doesn't pay
for re-embedding the 4,500-image val pool.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

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
from cards.retrieval.aligned import aligned_retrieval
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
K = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# concept -> a natural-language phrase, SigLIP's text encoder doesn't need
# perfect grammar, just semantic relevance.
CONCEPT_QUERY_TEXT: dict[str, str] = {
    "Arched_Eyebrows": "a person with arched eyebrows",
    "Bushy_Eyebrows": "a person with bushy eyebrows",
    "Bags_Under_Eyes": "a person with bags under their eyes",
    "Narrow_Eyes": "a person with narrow eyes",
    "Big_Nose": "a person with a big nose",
    "Pointy_Nose": "a person with a pointy nose",
    "Big_Lips": "a person with big lips",
    "Wearing_Lipstick": "a person wearing lipstick",
    "Mouth_Slightly_Open": "a person with their mouth slightly open",
    "Smiling": "a smiling person",
    "Bald": "a bald person",
    "Bangs": "a person with bangs",
    "Black_Hair": "a person with black hair",
    "Blond_Hair": "a person with blond hair",
    "Brown_Hair": "a person with brown hair",
    "Gray_Hair": "a person with gray hair",
    "Straight_Hair": "a person with straight hair",
    "Wavy_Hair": "a person with wavy hair",
    "Receding_Hairline": "a person with a receding hairline",
    "Pale_Skin": "a person with pale skin",
    "Rosy_Cheeks": "a person with rosy cheeks",
    "Eyeglasses": "a person wearing eyeglasses",
    "Wearing_Earrings": "a person wearing earrings",
    "Wearing_Hat": "a person wearing a hat",
    "Wearing_Necklace": "a person wearing a necklace",
    "Wearing_Necktie": "a person wearing a necktie",
}

# task name -> index of that task's positive-class logit in the 4-way head
# ([0:2]=Attractive, [2:4]=Young, see train_attractive_young_classifier.py).
TASK_POSITIVE_LOGIT_INDEX: dict[str, int] = {"Attractive": 1, "Young": 3}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    missing_queries = [c for c in GROUNDABLE_CONCEPTS if c not in CONCEPT_QUERY_TEXT]
    if missing_queries:
        raise ValueError(f"no query text defined for: {missing_queries}")

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "device": DEVICE,
            "encoder": {
                "name": "siglip",
                "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                "model_name": "ViT-B-16-SigLIP",
                "pretrained": "webli",
                "device": DEVICE,
            },
            "cache_dir": "embedding_cache",
        }
    )

    print("Loading SigLIP encoder + CARDS retrieval pool (CelebAMask-HQ held-out val split)...", flush=True)
    encoder = instantiate_encoder(cfg)
    cfg.dataset = {"name": "celeba", "root": str(CELEBA_HQ_ROOT)}
    cfg.pool_source = "val"
    pairs = load_celeba(CELEBA_HQ_ROOT, split="val")
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images", flush=True)

    print("Loading native celeba_attractive_young model (the black box CARDS explains)...", flush=True)
    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()

    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    rows = []
    for concept_name in GROUNDABLE_CONCEPTS:
        query_text = CONCEPT_QUERY_TEXT[concept_name]
        t_c = build_concept_query(query_text, encoder)
        t_c = demean_query(t_c, text_center)
        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
        absent_indices = aligned_retrieval(pool, present_indices, t_c, K)

        present_paths = [pool.paths[i] for i in present_indices]
        absent_paths = [pool.paths[i] for i in absent_indices]
        present_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in present_paths]).to(DEVICE)
        absent_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in absent_paths]).to(DEVICE)

        with torch.no_grad():
            present_logits = native_model(present_batch)  # (k, 4)
            absent_logits = native_model(absent_batch)

        scores_by_task = {}
        for task_name in TARGET_CLASSES:
            idx = TASK_POSITIVE_LOGIT_INDEX[task_name]
            raw_score = (present_logits[:, idx].mean() - absent_logits[:, idx].mean()).item()
            scores_by_task[task_name] = raw_score
            rows.append((concept_name, task_name, raw_score))

        print(f"{concept_name:<20s} <- {query_text!r:<40s} "
              f"Attractive={scores_by_task['Attractive']:+.4f}  Young={scores_by_task['Young']:+.4f}", flush=True)

    with open(RESULTS_DIR / "cards_celeba_full_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "target_task", "raw_score"])
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} (concept, task) CARDS scores to results/cards_celeba_full_scores.csv")


if __name__ == "__main__":
    main()
