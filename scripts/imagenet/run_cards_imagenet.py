"""Runs CARDS itself (not yet done anywhere in Phases 1-3) on the same
ImageNet+Broden setup TCAV (Phase 2) and PCBM's surrogate (Phase 1) were
built against -- same 5 concepts (car/cat/dog/chair/bottle), same native
resnet18 as the black box, same 25-class ImageNet slice as CARDS' own
retrieval pool (per Phase 1's plan: keeping the pool within the same
closed world PCBM/TCAV are scoped to).

Retrieval only (Steps 1-3, SigLIP, matched, k=30, no de-meaning -- a
simplification, not exhaustively tuned for this new domain) reuses
CARDS' own pipeline helpers directly. Scoring reuses the efficient
"all-classes-in-one-pass" pattern from the earlier CIFAR-100/CUB work in
this investigation: one native-resnet18 forward pass on the retrieved
P_c/N_c images gives raw_score for all 25 target classes at once,
instead of instantiating a separate single-target BlackBoxModel per
class.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from build_imagenet_slice import TARGET_CLASSES  # noqa: E402
from cards.models.backbones import BACKBONES  # noqa: E402
from cards.pipeline import instantiate_encoder  # noqa: E402
from cards.retrieval.confound import matched_retrieval  # noqa: E402
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool  # noqa: E402
from cards.retrieval.pool import CandidatePool  # noqa: E402
from cards.retrieval.retrieve import retrieve_top_bottom_k  # noqa: E402
from cards.concepts.prompts import build_concept_query  # noqa: E402

IMAGENET_SLICE_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\imagenet_slice")
RESULTS_DIR = Path("results")
K = 30
DEVICE = "cpu"  # matches Phase 2/3's device choice

CLASS_NAMES = [name for _, _, name in TARGET_CLASSES]
NATIVE_LABEL_IDX = [idx for idx, _, _ in TARGET_CLASSES]  # into the native model's 1000-way output

TEST_CONCEPTS = ["car", "cat", "dog", "chair", "bottle"]  # same as Phase 2/3


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

    print("Loading SigLIP encoder + building CARDS retrieval pool (25-class val slice)...", flush=True)
    encoder = instantiate_encoder(cfg)
    pairs = []
    for local_idx, name in enumerate(CLASS_NAMES):
        for p in sorted((IMAGENET_SLICE_ROOT / "val" / name).glob("*.jpg")):
            pairs.append((p, local_idx))
    # cache_key_for expects cfg.dataset/cfg.pool_source; fake minimal values distinguishing this pool
    cfg.dataset = {"name": "imagenet_slice", "root": str(IMAGENET_SLICE_ROOT)}
    cfg.pool_source = "val"
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images", flush=True)

    print("Loading native resnet18 (the black box CARDS explains)...", flush=True)
    spec = BACKBONES["resnet18"]
    native_model = spec.load_native().to(DEVICE).eval()
    native_label_tensor = torch.tensor(NATIVE_LABEL_IDX, device=DEVICE)

    raw_scores: dict[tuple[str, int], float] = {}  # (broden_concept_name, native_class_idx) -> raw_score

    for concept in TEST_CONCEPTS:
        print(f"\n=== {concept} ===", flush=True)
        t_c = build_concept_query(concept, encoder)
        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
        absent_indices = matched_retrieval(pool, present_indices, t_c)

        present_paths = [pool.paths[i] for i in present_indices]
        absent_paths = [pool.paths[i] for i in absent_indices]
        present_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in present_paths]).to(DEVICE)
        absent_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in absent_paths]).to(DEVICE)

        with torch.no_grad():
            present_logits = native_model(present_batch)[:, native_label_tensor]  # (k, 25)
            absent_logits = native_model(absent_batch)[:, native_label_tensor]

        raw_score_all_classes = (present_logits.mean(dim=0) - absent_logits.mean(dim=0)).tolist()
        for native_idx, score in zip(NATIVE_LABEL_IDX, raw_score_all_classes):
            raw_scores[(concept, native_idx)] = score

        top3 = sorted(zip(CLASS_NAMES, raw_score_all_classes), key=lambda x: -x[1])[:3]
        print(f"top-3 classes by raw_score: {[(n, round(s, 3)) for n, s in top3]}", flush=True)

    import csv

    with open(RESULTS_DIR / "cards_imagenet_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["broden_concept", "native_class_idx", "class_name", "raw_score"])
        idx_to_name = {idx: name for idx, name in zip(NATIVE_LABEL_IDX, CLASS_NAMES)}
        for (concept, native_idx), score in raw_scores.items():
            writer.writerow([concept, native_idx, idx_to_name[native_idx], score])

    print(f"\nSaved {len(raw_scores)} (concept, class) CARDS scores to results/cards_imagenet_scores.csv")


if __name__ == "__main__":
    main()
