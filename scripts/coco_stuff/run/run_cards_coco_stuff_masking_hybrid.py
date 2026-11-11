"""CARDS masking-hybrid attribution on COCO-Stuff.

Runs the masking_hybrid pipeline on all 143 Broden concepts against a
ResNet18 classifier trained on the 20 co-occurrence-bias COCO-Stuff classes
(Singh et al., 2020).

Settings (matching validated CelebA masking-hybrid config):
- Encoder:       SigLIP ViT-B-16-SigLIP (webli)
- Pool:          all COCO val2017 images (~5k)
- Retrieval:     top-K=50 by cosine similarity to de-meaned t_c
- Demean:        True  (subtract GENERIC_REFERENCE_CONCEPTS text center)
- Orthogonalize: True  (Lowdin orthogonalization over all 143 concept queries)
- Localization:  SigLIP patch embeddings dotted against t_c
- Threshold:     z-score method -- mean + ALPHA*std pooled over all K present-
                 set sim-maps for each concept (two-pass per concept).
                 ALPHA is determined by ablate_cards_coco_stuff_alpha.py;
                 update it here after running that script.
- Masking:       best-of-7 fill strategies (smallest embedding-shift angle wins)
- Scoring:       mean(logit_orig - logit_masked) for ALL 20 classes in one pass

Outputs (results/coco_stuff/):
  cards_coco_stuff_masking_hybrid_scores.csv  -- concept_name, class_name, raw_score
  cards_coco_stuff_masking_hybrid_angles.csv  -- concept_name, image_idx, strategy, angle

Prerequisites:
  1. Train ResNet18:  scripts/coco_stuff/build/train_coco_stuff_classifier.py
  2. Run alpha ablation to determine ALPHA:
       scripts/coco_stuff/run/ablate_cards_coco_stuff_alpha.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from scipy import stats

from cards.attribution.localization import (
    concept_zscore_cutoff,
    localize_concept,
    threshold_mask,
)
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

COCO_ROOT   = Path("/scratch/rnair21/LLMConceptEditing_old/coco")
BRODEN_ROOT = Path("/scratch/rnair21/LLMConceptEditing_old/broden_concepts")
CKPT_PATH   = Path("trained_models_new/coco_stuff/resnet18_coco_stuff.pt")
RESULTS_DIR = Path("results/coco_stuff")

BRODEN_DROPPED_CONCEPTS: set[str] = {
    "air_conditioner", "apron", "awning", "bathroom_s", "beak", "bedroom_s",
    "blotchy", "bumper", "can", "candlestick", "cap", "clock", "dining_room_s",
    "drinking_glass", "eye", "eyebrow", "faucet", "figurine", "handle",
    "inside_arm", "jar", "knob", "loudspeaker", "mouse", "mouth",
    "outside_arm", "street_s",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

K     = 50
ALPHA = 1.0   # z-score std multiplier -- update after running ablate_cards_coco_stuff_alpha.py
SEED  = 42
FILL_STRATEGIES = [
    "blur", "zero_fill", "mean_fill", "hue_shift",
    "white_fill", "zero_fill_noise", "noise_then_blur",
]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Encoder ---
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
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)
    print(f"Encoder: SigLIP ViT-B-16-SigLIP  device: {DEVICE}", flush=True)

    # --- Pool ---
    cfg.dataset   = {"name": "coco_stuff", "root": str(COCO_ROOT)}
    cfg.pool_source = "val"
    pairs = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="val")
    pool  = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"Pool: {len(pool.paths)} COCO val2017 images", flush=True)

    # --- ResNet18 ---
    import torch.nn as nn
    from torchvision.models import resnet18
    raw_model = resnet18(weights=None)
    raw_model.fc = nn.Linear(raw_model.fc.in_features, len(COCO_STUFF_20_CLASSES))
    raw_model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    raw_model = raw_model.to(DEVICE).eval()
    preprocess = _coco_stuff_preprocess()
    print(f"ResNet18: {CKPT_PATH}", flush=True)

    # --- Concepts: build all queries upfront, orthogonalize jointly ---
    concepts = [c for c in list_broden_concepts(BRODEN_ROOT) if c not in BRODEN_DROPPED_CONCEPTS]
    print(f"Concepts: {len(concepts)}  orthogonalize=True  alpha={ALPHA}", flush=True)

    raw_queries = {
        c: demean_query(build_concept_query(c, encoder), text_center)
        for c in concepts
    }
    queries = orthogonalize_queries(raw_queries)

    # --- Per-concept two-pass scoring ---
    scores: dict[str, dict[str, float]] = {cls: {} for cls in COCO_STUFF_20_CLASSES}
    angle_rows: list[tuple[str, int, str, float]] = []
    selection_counts: dict[str, Counter] = {c: Counter() for c in concepts}

    for concept_idx, concept_name in enumerate(concepts):
        t_c     = queries[concept_name]
        t_c_dev = t_c.to(DEVICE)

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)

        # Pass 1: localize all present images, cache (idx, image, sim_map)
        cached: list[tuple[int, Image.Image, np.ndarray]] = []
        for idx in present_indices:
            image   = Image.open(pool.paths[idx]).convert("RGB")
            sim_map = localize_concept(encoder, image, t_c, (image.height, image.width))
            cached.append((idx, image, sim_map))
        sim_maps = [sm for _, _, sm in cached]

        # Per-concept z-score cutoff pooled across all K present-set sim-maps
        cutoff = concept_zscore_cutoff(sim_maps, ALPHA)

        # Pass 2: threshold, mask, fill, score
        delta_logits: dict[str, list[float]] = {cls: [] for cls in COCO_STUFF_20_CLASSES}
        concept_angles: list[float] = []
        n_skipped = 0

        for idx, image, sim_map in cached:
            mask = threshold_mask(sim_map, method="fixed", cutoff=cutoff)
            if not mask.any() or mask.all():
                n_skipped += 1
                continue

            rng = np.random.default_rng(SEED + concept_idx * 10_000 + int(idx))
            candidates = [mask_region(image, mask, strategy=s, rng=rng) for s in FILL_STRATEGIES]

            with torch.no_grad():
                embeds = encoder.encode_images([image] + candidates).to(DEVICE)
            embed_orig = embeds[0]

            best_angle, best_i = None, 0
            for i in range(len(FILL_STRATEGIES)):
                diff = embed_orig - embeds[1 + i]
                diff_unit = diff / diff.norm()
                cos_sim = float(torch.clamp(diff_unit @ t_c_dev, -1.0, 1.0))
                angle_deg = float(np.degrees(np.arccos(cos_sim)))
                if best_angle is None or angle_deg < best_angle:
                    best_angle, best_i = angle_deg, i

            selected_strategy = FILL_STRATEGIES[best_i]
            masked_image = candidates[best_i]
            selection_counts[concept_name][selected_strategy] += 1
            concept_angles.append(best_angle)
            angle_rows.append((concept_name, int(idx), selected_strategy, best_angle))

            pixels_orig   = preprocess(image).unsqueeze(0)
            pixels_masked = preprocess(masked_image).unsqueeze(0)
            batch = torch.cat([pixels_orig, pixels_masked], dim=0).to(DEVICE)
            with torch.no_grad():
                logits = raw_model(batch)   # (2, 20)

            for cls_idx, cls_name in enumerate(COCO_STUFF_20_CLASSES):
                delta_logits[cls_name].append(
                    (logits[0, cls_idx] - logits[1, cls_idx]).item()
                )

        for cls_name in COCO_STUFF_20_CLASSES:
            scores[cls_name][concept_name] = (
                float(np.mean(delta_logits[cls_name])) if delta_logits[cls_name] else 0.0
            )

        angles_arr  = np.array(concept_angles) if concept_angles else np.array([float("nan")])
        counts_str  = ", ".join(f"{s}={c}" for s, c in selection_counts[concept_name].most_common(3))
        mean_area   = 100.0 * float(np.mean([sm >= cutoff for sm in np.concatenate([m.flatten() for m in sim_maps])])) if sim_maps else 0.0
        print(
            f"[{concept_idx + 1:>3d}/{len(concepts)}] {concept_name:<25s} "
            f"n={len(concept_angles):>3d} (skipped {n_skipped})  "
            f"cutoff={cutoff:+.4f}  area={mean_area:.1f}%  "
            f"angle mean={angles_arr.mean():.2f}°  |  {counts_str}",
            flush=True,
        )

    # --- Save scores ---
    scores_path = RESULTS_DIR / "cards_coco_stuff_masking_hybrid_scores.csv"
    with open(scores_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "class_name", "raw_score"])
        for cls_name in COCO_STUFF_20_CLASSES:
            for concept_name in concepts:
                writer.writerow([concept_name, cls_name, scores[cls_name][concept_name]])
    print(f"\nSaved {scores_path}", flush=True)

    angles_path = RESULTS_DIR / "cards_coco_stuff_masking_hybrid_angles.csv"
    with open(angles_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "image_idx", "selected_strategy", "angle_degrees"])
        writer.writerows(angle_rows)
    print(f"Saved {angles_path}", flush=True)

    # --- Overall angle stats ---
    all_angles = np.array([a for *_, a in angle_rows])
    if len(all_angles) > 1:
        t_stat, p_val = stats.ttest_1samp(all_angles, 90.0)
        print(
            f"\nOVERALL angle(embed_diff, t_c): n={len(all_angles)} "
            f"mean={all_angles.mean():.2f}° std={all_angles.std():.2f}°  "
            f"t-test vs 90°: t={t_stat:.3f} p={p_val:.4g}",
            flush=True,
        )
    overall_counts: Counter = Counter()
    for c in selection_counts.values():
        overall_counts.update(c)
    print("Overall strategy selection counts:", dict(overall_counts.most_common()), flush=True)


if __name__ == "__main__":
    main()
