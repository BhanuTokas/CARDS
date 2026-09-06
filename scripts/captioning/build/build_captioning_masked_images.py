"""Generates the "Concept Mask" (cfg.scoring_mode == "masking_hybrid")
positive/masked image pairs for the captioning ranking experiment:
for each of the 79 task-object concepts, up to N_PER_CONCEPT ground-truth
positive images (real human object-presence annotation, NOT CARDS'
own retrieval -- deliberately, to isolate localization/masking quality
from retrieval quality for this experiment, per direct discussion) get
the concept localized via CARDS' own patch-similarity mechanism
(`localize_concept`/`threshold_mask`, no manual masks, no external
segmenter -- COCO's real instance segmentation exists but is NOT used
here, since using it would test an oracle-masking condition rather than
what CARDS actually does) and masked via the same best-of-7-fill-family
selection validated on CelebA/CUB (`masking_hybrid`'s own angle-to-query
selection, ported verbatim from `run_cards_celeba_masking_hybrid_best_
of_family_full.py`).

No black-box scoring happens here -- caption generation for both the
original and masked image sets happens externally on the user's own HPC
pipeline. This script's only job is to produce the masked image files
plus a manifest CSV joining (img_name, concept_name) to
(original_path, masked_path, selected fill strategy, angle). The actual
attribution score (mean caption-based gender-word delta between original
and masked captions) gets computed downstream once captions come back.

Scale: capped at N_PER_CONCEPT=90 positives per concept (this project's
own established convention, e.g. CelebA's N_PER_ATTRIBUTE), sampled with
a fixed seed for determinism -- NOT run on the full positive set (some
concepts like "chair" have 1,636 positives; masking + captioning all of
those across concepts would multiply HPC compute well beyond what's
needed for a ranking correlation).
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cards.attribution.localization import localize_concept, threshold_mask
from cards.attribution.masking_mode import DEFAULT_FILL_STRATEGIES
from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.captioning_bias import groundable_concepts, load_bias_captioning_records
from cards.pipeline import instantiate_encoder
from cards.validation.broden_faithfulness import mask_region

COCO_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\COCO2014")
OUT_ROOT = Path("results/captioning_bias_masked_images")
RESULTS_DIR = Path("results")
SEED = 42
N_PER_CONCEPT = 90
TOP_PCT = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def image_path_for(img_name: str) -> Path:
    subdir = "train2014" if "train2014" in img_name else "val2014"
    return COCO_ROOT / subdir / img_name


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng_py = random.Random(SEED)

    print("Loading captioning-bias records...", flush=True)
    records = load_bias_captioning_records()
    concepts = groundable_concepts(records)
    positives_by_concept: dict[str, list[str]] = {c: [] for c in concepts}
    for r in records:
        for c in r["rmdup_object_list"]:
            if c in positives_by_concept:
                positives_by_concept[c].append(r["img_name"])

    cfg = OmegaConf.create({
        "seed": SEED, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    manifest_rows = []
    for concept_idx, concept_name in enumerate(concepts):
        candidates_pool = positives_by_concept[concept_name]
        chosen = candidates_pool if len(candidates_pool) <= N_PER_CONCEPT else rng_py.sample(candidates_pool, N_PER_CONCEPT)

        query_text = f"a photo of a {concept_name}"
        t_c = build_concept_query(query_text, encoder)
        t_c = demean_query(t_c, text_center)
        t_c_dev = t_c.to(DEVICE)

        concept_out_dir = OUT_ROOT / concept_name.replace(" ", "_")
        concept_out_dir.mkdir(parents=True, exist_ok=True)

        n_skipped_degenerate = 0
        n_saved = 0
        for img_name in chosen:
            image_path = image_path_for(img_name)
            if not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")

            sim_map = localize_concept(encoder, image, t_c, (image.height, image.width))
            mask = threshold_mask(sim_map, top_pct=TOP_PCT, method="top_pct")
            if not mask.any() or mask.all():
                n_skipped_degenerate += 1
                continue

            rng = np.random.default_rng(SEED + concept_idx * 10_000 + hash(img_name) % 1_000_000)
            fill_candidates = [mask_region(image, mask, strategy=s, rng=rng) for s in DEFAULT_FILL_STRATEGIES]
            with torch.no_grad():
                embeds = encoder.encode_images([image] + fill_candidates).to(DEVICE)
            embed_orig = embeds[0]
            best_angle, best_i = None, 0
            for i in range(len(DEFAULT_FILL_STRATEGIES)):
                diff = embed_orig - embeds[1 + i]
                diff_unit = diff / diff.norm()
                cos_sim = float(torch.clamp(diff_unit @ t_c_dev, -1.0, 1.0))
                angle_deg = float(np.degrees(np.arccos(cos_sim)))
                if best_angle is None or angle_deg < best_angle:
                    best_angle, best_i = angle_deg, i

            selected_strategy = DEFAULT_FILL_STRATEGIES[best_i]
            masked_image = fill_candidates[best_i]
            masked_path = concept_out_dir / img_name
            masked_image.save(masked_path)
            n_saved += 1

            manifest_rows.append({
                "img_name": img_name, "concept_name": concept_name,
                "original_path": str(image_path), "masked_path": str(masked_path),
                "selected_strategy": selected_strategy, "angle_degrees": best_angle,
            })

        print(f"[{concept_idx + 1}/{len(concepts)}] {concept_name:<20s} "
              f"saved={n_saved} skipped_degenerate={n_skipped_degenerate} "
              f"(pool={len(candidates_pool)}, sampled={len(chosen)})", flush=True)

    manifest_path = RESULTS_DIR / "captioning_bias_masked_images_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["img_name", "concept_name", "original_path", "masked_path",
                                                "selected_strategy", "angle_degrees"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\n{len(manifest_rows)} masked images saved. Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
