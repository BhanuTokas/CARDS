"""CARDS attribution across 11 shortcut-rate COCO-Stuff models.

Runs CARDS (masking-hybrid, alpha=1.5, best-of-7 fill) on the COCO val2017 pool
for each of 11 shortcut-rate ResNet18 checkpoints (0-100% in steps of 10%).
Localisation (SigLIP encoder + concept queries) is computed ONCE and cached
across all 11 models -- only the scoring step (ResNet18 logit difference) repeats.

Concepts: 103 balanced Broden concepts (from COUNTS_CSV).
Output: results/coco_stuff/shortcut/cards_shortcut_scores.csv
  rate_pct, concept_name, class_name, raw_score
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.coco_stuff import COCO_STUFF_20_CLASSES
from cards.data.datasets import list_broden_concepts, load_coco_stuff
from cards.models.coco_stuff_black_box import _coco_stuff_preprocess
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import mask_region

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COCO_ROOT   = Path("/scratch/rnair21/LLMConceptEditing_old/coco")
BRODEN_ROOT = Path("/scratch/rnair21/LLMConceptEditing_old/broden_concepts")
RESULTS_DIR = Path("results/coco_stuff")
OUT_DIR     = Path("results/coco_stuff/shortcut")
CKPT_DIR    = Path("trained_models_new/coco_stuff/shortcut")
COUNTS_CSV  = RESULTS_DIR / "coco_stuff_faithfulness_detection_counts.csv"

BRODEN_DROPPED_CONCEPTS: set[str] = {
    "air_conditioner", "apron", "awning", "bathroom_s", "beak", "bedroom_s",
    "blotchy", "bumper", "can", "candlestick", "cap", "clock", "dining_room_s",
    "drinking_glass", "eye", "eyebrow", "faucet", "figurine", "handle",
    "inside_arm", "jar", "knob", "loudspeaker", "mouse", "mouth",
    "outside_arm", "street_s",
}

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
K               = 50
ALPHA           = 1.5
SEED            = 42
RATES_PCT       = list(range(0, 101, 10))
MIN_FAITH_IMGS  = 100

FILL_STRATEGIES = [
    "blur", "zero_fill", "mean_fill", "hue_shift",
    "white_fill", "zero_fill_noise", "noise_then_blur",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_balanced_concepts(counts_csv: Path, min_images: int) -> set[str]:
    balanced: set[str] = set()
    with open(counts_csv, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["n_images"]) >= min_images:
                balanced.add(row["concept_name"])
    return balanced


def _load_shortcut_model(rate_pct: int) -> torch.nn.Module:
    import torch.nn as nn
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(COCO_STUFF_20_CLASSES))
    ckpt = CKPT_DIR / f"resnet18_coco_stuff_shortcut_{rate_pct}.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(DEVICE).eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Encoder + pool (shared across all 11 models) ---
    cfg = OmegaConf.create({
        "seed": SEED, "device": DEVICE,
        "encoder": {
            "name": "siglip",
            "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
            "model_name": "ViT-B-16-SigLIP",
            "pretrained": "webli",
            "device": DEVICE,
        },
        "cache_dir": "embedding_cache",
    })
    encoder    = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    cfg.dataset    = {"name": "coco_stuff", "root": str(COCO_ROOT)}
    cfg.pool_source = "val"
    pairs = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="val")
    pool  = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"Pool: {len(pool.paths)} COCO val2017 images", flush=True)

    preprocess = _coco_stuff_preprocess()

    # --- Balanced concepts ---
    balanced     = _load_balanced_concepts(COUNTS_CSV, MIN_FAITH_IMGS)
    all_concepts = [c for c in list_broden_concepts(BRODEN_ROOT)
                    if c not in BRODEN_DROPPED_CONCEPTS]
    concepts     = [c for c in all_concepts if c in balanced]
    concept_idx  = {c: i for i, c in enumerate(concepts)}
    print(f"All concepts: {len(all_concepts)}  scored (balanced): {len(concepts)}", flush=True)

    raw_queries = {
        c: demean_query(build_concept_query(c, encoder), text_center)
        for c in all_concepts
    }
    queries = orthogonalize_queries(raw_queries)

    # --- Localise concepts ONCE (model-independent) ---
    print("Localising all concepts on val pool (cached across models)...", flush=True)
    concept_cache: dict[str, list[tuple[int, Image.Image, np.ndarray]]] = {}
    for c_i, concept_name in enumerate(concepts):
        t_c = queries[concept_name]
        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
        cached = []
        for idx in present_indices:
            image   = Image.open(pool.paths[idx]).convert("RGB")
            sim_map = localize_concept(encoder, image, t_c, (image.height, image.width))
            cached.append((idx, image, sim_map))
        concept_cache[concept_name] = cached
        print(f"  [{c_i + 1:>3d}/{len(concepts)}] {concept_name}", flush=True)

    # --- Score each of 11 models ---
    all_rows: list[tuple] = []
    scores_by_rate: dict[int, float] = {}

    for rate_pct in RATES_PCT:
        print(f"\n=== rate={rate_pct}% ===", flush=True)
        model = _load_shortcut_model(rate_pct)
        abs_scores: list[float] = []

        for c_i, concept_name in enumerate(concepts):
            t_c     = queries[concept_name]
            t_c_dev = t_c.to(DEVICE)
            cached  = concept_cache[concept_name]
            sim_maps = [sm for _, _, sm in cached]
            cutoff   = concept_zscore_cutoff(sim_maps, ALPHA)

            for cls_idx, cls_name in enumerate(COCO_STUFF_20_CLASSES):
                delta_logits: list[float] = []
                n_skipped = 0

                for _img_i, (idx, image, sim_map) in enumerate(cached):
                    mask = threshold_mask(sim_map, method="fixed", cutoff=cutoff)
                    if not mask.any() or mask.all():
                        n_skipped += 1
                        continue

                    rng = np.random.default_rng(
                        SEED + concept_idx[concept_name] * 10_000 + int(idx)
                    )
                    candidates = [mask_region(image, mask, strategy=s, rng=rng) for s in FILL_STRATEGIES]

                    with torch.no_grad():
                        embeds = encoder.encode_images([image] + candidates).to(DEVICE)
                    embed_orig = embeds[0]

                    best_angle, best_i = None, 0
                    for i in range(len(FILL_STRATEGIES)):
                        diff      = embed_orig - embeds[1 + i]
                        diff_unit = diff / (diff.norm() + 1e-8)
                        cos_sim   = float(torch.clamp(diff_unit @ t_c_dev, -1.0, 1.0))
                        angle_deg = float(np.degrees(np.arccos(cos_sim)))
                        if best_angle is None or angle_deg < best_angle:
                            best_angle, best_i = angle_deg, i

                    masked_image  = candidates[best_i]
                    pix_orig      = preprocess(image).unsqueeze(0)
                    pix_masked    = preprocess(masked_image).unsqueeze(0)
                    batch         = torch.cat([pix_orig, pix_masked], dim=0).to(DEVICE)
                    with torch.no_grad():
                        logits = model(batch)
                    delta_logits.append((logits[0, cls_idx] - logits[1, cls_idx]).item())

                raw_score = float(np.mean(delta_logits)) if delta_logits else 0.0
                all_rows.append((rate_pct, concept_name, cls_name, raw_score))
                abs_scores.append(abs(raw_score))

        mean_abs = float(np.mean(abs_scores)) if abs_scores else 0.0
        scores_by_rate[rate_pct] = mean_abs
        print(f"  mean |raw_score| across all concepts x classes: {mean_abs:.4f}", flush=True)

    # --- Save CSV ---
    out_path = OUT_DIR / "cards_shortcut_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rate_pct", "concept_name", "class_name", "raw_score"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows -> {out_path}", flush=True)

    print("\n=== CARDS mean |raw_score| per rate ===", flush=True)
    for rate_pct in RATES_PCT:
        print(f"  rate={rate_pct:>3d}%: {scores_by_rate[rate_pct]:.4f}", flush=True)


if __name__ == "__main__":
    main()
