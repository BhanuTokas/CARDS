"""Broden-concept variant of `build_captioning_masked_images.py` -- same
"Concept Mask" masking mechanism (CARDS' own patch-similarity
localization + best-of-7-fill-family angle selection, no manual masks,
no external segmenter), but scored against Broden's FULL 170-concept
vocabulary (`post_hoc_cbm`'s `BRODEN_CONCEPTS` directory listing)
instead of DBAC's own 79 COCO task-object concepts. Prompted directly
("I want to also explore Broden Concepts for our existing experiment"
-> "Replace/extend DBAC's own concept set" -> "Use CARDS' own
retrieval").

**Deliberately does NOT apply `BRODEN_DROPPED_CONCEPTS`** (the 27-concept
drop list from [[project-broden-label-corruption]]) -- confirmed
directly ("Actually, we do not need to drop the concepts anymore").
That list exists to protect against training CAVs on Broden's own
mislabeled POSITIVE IMAGES; this script never loads a single Broden
image or mask -- it only reuses Broden's CONCEPT NAMES as text queries,
with presence in DBAC's own images decided by CARDS' own retrieval (see
below), so the corruption in Broden's own image folders is simply not
reachable from here. All 170 concepts are used.

**Kept as a genuinely SEPARATE concept-set variant, not merged with the
existing 79-concept manifest** -- confirmed directly, since the two use
fundamentally different presence sources:
  - The existing 79 COCO concepts use DBAC's own ground-truth `rmdup_
    object_list` human annotation to decide which images are concept-
    present, specifically so retrieval quality never confounds the
    localization/masking quality this experiment is measuring.
  - Broden's 143 concepts (colors/textures/materials/objects/scene
    types) have NO such ground-truth annotation anywhere in DBAC's own
    10,780-image dataset -- there's no human label for "this image
    contains something wooden." The only way to get a present set for
    them is CARDS' own retrieval (top-K by SigLIP similarity to the
    concept's text query), same as every other CARDS concept bank.
    Confirmed directly as the accepted tradeoff (reintroduces the
    retrieval-quality confound the 79-concept design deliberately
    avoided) rather than silently pretending the two are equivalent.

Concept list sourced directly from `post_hoc_cbm` (sibling repo) rather
than duplicated locally: `BRODEN_CONCEPTS`'s own 170 concept subfolder
names, all of them. Query phrasing: `f"a photo of
{concept_name.replace('_', ' ')}"` -- Broden's folder names use
underscores for multi-word concepts (`air_conditioner`, `back_pillow`)
and a few carry a "_s" scene-suffix convention (`bathroom_s`,
`bedroom_s`, `dining_room_s`, `street_s`) that reads a little oddly
after the space-replacement (e.g. "bathroom s") but is left as-is --
not worth special-casing 4/170 concepts for a query-phrasing nicety.

Retrieval pool = ALL 10,780 DBAC images (not a ground-truth-filtered
subset, since there's no ground truth to filter by here), K=50 per
concept -- this project's standing default (CelebA/CUB/FTW all use
K=50), via the same `load_or_build_pool`/`retrieve_top_bottom_k`
machinery every other CARDS track uses, rather than the presence-list
lookup the 79-concept script uses. Everything downstream of "which
images are present" (localization, masking, fill-strategy selection,
manifest shape) is IDENTICAL to `build_captioning_masked_images.py` --
see that script's own docstring for that shared design.

No black-box scoring happens here either -- caption generation for the
masked images happens externally, same convention as the 79-concept
manifest (`generate_captions_with_logprobs.py`'s own masked-image path,
pointed at THIS script's manifest instead).

**`CAPTIONING_COCO2014_ROOT`, not `CAPTIONING_COCO_ROOT`** -- deliberately
a DIFFERENT env var than `generate_captions_with_logprobs.py`'s own
`CAPTIONING_COCO_ROOT`, which points directly at the flat `val2014/`
folder (that script's own 10,780 "original" images are all val2014-only).
This script needs the PARENT `COCO2014/` directory instead (containing
BOTH `train2014/` and `val2014/`, since DBAC's own images span both
splits -- see `image_path_for`). Reusing the same env var name would
silently break whichever script got the "wrong" semantics if both were
exported identically on a new host.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from data.constants import BRODEN_CONCEPTS

from cards.attribution.localization import localize_concept, threshold_mask
from cards.attribution.masking_mode import DEFAULT_FILL_STRATEGIES
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.captioning_bias import load_bias_captioning_records
from cards.pipeline import instantiate_encoder
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import mask_region

COCO_ROOT = Path(os.environ.get("CAPTIONING_COCO2014_ROOT", r"C:\Users\btokas\Projects\Datasets\COCO2014"))
OUT_ROOT = Path(os.environ.get("CARDS_RESULTS_DIR", "results")) / "captioning_bias_broden_masked_images"
RESULTS_DIR = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
CACHE_DIR = Path(os.environ.get("CARDS_CACHE_DIR", "embedding_cache"))
SEED = 42
K = 50
TOP_PCT = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def image_path_for(img_name: str) -> Path:
    subdir = "train2014" if "train2014" in img_name else "val2014"
    return COCO_ROOT / subdir / img_name


def broden_concepts() -> list[str]:
    return sorted(d.name for d in Path(BRODEN_CONCEPTS).iterdir() if d.is_dir())


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading captioning-bias records (retrieval pool, all images, no ground-truth filtering)...", flush=True)
    records = load_bias_captioning_records()
    all_img_names = [r["img_name"] for r in records]
    print(f"{len(all_img_names)} images in the pool.", flush=True)

    concepts = broden_concepts()
    print(f"{len(concepts)} Broden concepts (full vocabulary, no drop list applied).", flush=True)

    cfg = OmegaConf.create({
        "seed": SEED, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": str(CACHE_DIR),
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    pairs = [(image_path_for(name), 0) for name in all_img_names]  # label unused by retrieval
    pool_cfg = OmegaConf.create({"seed": SEED, "device": DEVICE, "encoder": cfg.encoder, "cache_dir": str(CACHE_DIR)})
    pool_cfg.dataset = {"name": "captioning_bias_pool", "root": str(COCO_ROOT)}
    pool_cfg.pool_source = "all"
    pool = load_or_build_pool(CACHE_DIR, cache_key_for(pool_cfg), pairs, encoder)
    print(f"Pool built/loaded: {len(pool.paths)} images.", flush=True)

    manifest_rows = []
    for concept_idx, concept_name in enumerate(concepts):
        query_text = f"a photo of {concept_name.replace('_', ' ')}"
        t_c = build_concept_query(query_text, encoder)
        t_c = demean_query(t_c, text_center)
        t_c_dev = t_c.to(DEVICE)

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)

        concept_out_dir = OUT_ROOT / concept_name
        concept_out_dir.mkdir(parents=True, exist_ok=True)

        n_skipped_degenerate = 0
        n_saved = 0
        for idx in present_indices:
            image_path = pool.paths[idx]
            img_name = image_path.name
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
                "original_path": image_path.as_posix(), "masked_path": masked_path.as_posix(),
                "selected_strategy": selected_strategy, "angle_degrees": best_angle,
            })

        print(f"[{concept_idx + 1}/{len(concepts)}] {concept_name:<20s} "
              f"saved={n_saved} skipped_degenerate={n_skipped_degenerate} (retrieved={len(present_indices)})", flush=True)

    manifest_path = RESULTS_DIR / "captioning_bias_broden_masked_images_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["img_name", "concept_name", "original_path", "masked_path",
                                                "selected_strategy", "angle_degrees"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\n{len(manifest_rows)} masked images saved. Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
