"""Phase 5 of the CelebA plan: runs CARDS against the 8-concept pilot,
against BOTH target classes (Attractive, Young) -- the Phase 4 masking
ground truth this will eventually be scored against (Phase 7).

Structural template: scripts/cub/run/run_cards_cub_attributes.py. Same
retrieval strategy/hyperparameter defaults CUB settled on after its own
ablation (v46/v47) -- aligned_retrieval, K=50, demean_query=True, SigLIP
encoder -- adopted here as a STARTING default, not a settled one; the
plan's own Phase 7 explicitly calls for re-ablating on CelebA before
trusting any single CARDS number, since CUB repeatedly found settings
don't reliably transfer between concept banks (v44/v47/v48).

Unlike CUB, the black box here has only 2 possible "classes of interest"
(Attractive, Young), each its own 2-way softmax block within Phase 1's
4-way head -- not 200 species. So raw_score is computed directly against
each task's own positive-class LOGIT (index 1 within [Attractive_not,
Attractive] or [Young_not, Young]), not swept across all output classes
the way CUB's script recorded a score for every one of 200 species per
attribute. This yields exactly 8 concepts x 2 tasks = 16 (concept, task)
raw_score rows, matching Phase 4's ground truth shape 1:1 -- no
downstream filtering to a ground-truth-relevant subset needed, unlike
CUB (which scored all 200 classes per attribute because any one of them
might turn out to be a positive species in the ground truth).

Retrieval pool = the same held-out val split (cards.data.datasets.
load_celeba, split="val") Phase 4's ground truth draws from -- matches
CUB's own convention of scoring CARDS and the masking ground truth
against the identical image population.
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
from cards.data.celeba_attributes import PILOT_CONCEPTS, TARGET_CLASSES
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
# perfect grammar, just semantic relevance -- matches the plan's own
# worked examples (Smiling, Eyeglasses).
CONCEPT_QUERY_TEXT: dict[str, str] = {
    "Black_Hair": "a person with black hair",
    "Bushy_Eyebrows": "a person with bushy eyebrows",
    "Big_Nose": "a person with a big nose",
    "Smiling": "a smiling person",
    "Narrow_Eyes": "a person with narrow eyes",
    "Eyeglasses": "a person wearing eyeglasses",
    "Pale_Skin": "a person with pale skin",
    "Wearing_Hat": "a person wearing a hat",
}

# task name -> index of that task's positive-class logit in the 4-way head
# ([0:2]=Attractive, [2:4]=Young, see train_attractive_young_classifier.py).
TASK_POSITIVE_LOGIT_INDEX: dict[str, int] = {"Attractive": 1, "Young": 3}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

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
    for concept_name in PILOT_CONCEPTS:
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

    with open(RESULTS_DIR / "cards_celeba_pilot_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "target_task", "raw_score"])
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} (concept, task) CARDS scores to results/cards_celeba_pilot_scores.csv")


if __name__ == "__main__":
    main()
