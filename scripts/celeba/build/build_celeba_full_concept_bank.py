"""Phase 7 scale-up of the CelebA plan: materializes a concept bank for
ALL 26 groundable concepts (cards.data.celeba_attributes.
GROUNDABLE_CONCEPTS) from CelebAMask-HQ's own real per-pixel masks --
extends build_celeba_pilot_concept_bank.py's own 8-concept pilot bank
(v71) the same way. Logic is IDENTICAL to the pilot script (same
positives/negatives crop-folder layout, same v20-lesson contrastive-
negative design drawn from the other 25 concepts' own crops, same
TRAIN-split-only source images) -- see that script's own docstring for
the full design rationale.

Candidate-pool sizes for the 18 new concepts were checked directly
before this run (same discipline as the pilot's own pre-check); the
tightest, Wearing_Necklace, still has 150 candidates for its rarer task
intersection -- comfortably above MAX_POSITIVES_PER_CONCEPT=150's own
target on the TRAIN split specifically (5.67x larger than val), so no
concept here is expected to fall short the way Eyeglasses/Wearing_Hat's
own train-pool numbers were checked (not assumed) to still clear 150 in
the pilot run.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cards.data.celeba import (
    load_celebamask_hq_image_paths,
    load_celebamask_hq_mask,
    split_celebamask_hq,
)
from cards.data.celeba_attributes import (
    ATTRIBUTE_TO_REGIONS,
    GROUNDABLE_CONCEPTS,
    TARGET_CLASSES,
    load_attribute_labels,
    load_attribute_names,
)

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
OUTPUT_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\celeba_full_concepts")
SEED = 42
PADDING_FRAC = 0.25
MIN_CROP_PX = 20
MAX_POSITIVES_PER_CONCEPT = 150
MAX_NEGATIVES_PER_CONCEPT = 150


def crop_with_padding(image: Image.Image, mask: np.ndarray) -> Image.Image | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = y1 - y0, x1 - x0
    pad_y, pad_x = int(h * PADDING_FRAC), int(w * PADDING_FRAC)
    y0, y1 = max(0, y0 - pad_y), min(mask.shape[0], y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(mask.shape[1], x1 + pad_x)
    if min(y1 - y0, x1 - x0) < MIN_CROP_PX:
        return None
    return image.crop((x0, y0, x1, y1))


def main():
    rng = random.Random(SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading CelebAMask-HQ metadata...", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]
    train_hq, _ = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices)
    rng.shuffle(train_hq)
    print(f"{len(train_hq)} train images available as crop source.", flush=True)
    print(f"{len(GROUNDABLE_CONCEPTS)} groundable concepts to build.", flush=True)

    # Pass 1: positive crops for every groundable concept.
    positive_crops: dict[str, list[Path]] = {c: [] for c in GROUNDABLE_CONCEPTS}
    for concept_name in GROUNDABLE_CONCEPTS:
        region_names = ATTRIBUTE_TO_REGIONS[concept_name]
        pos_dir = OUTPUT_ROOT / concept_name / "positives"
        pos_dir.mkdir(parents=True, exist_ok=True)

        n_tried = 0
        for hq_idx in train_hq:
            if len(positive_crops[concept_name]) >= MAX_POSITIVES_PER_CONCEPT:
                break
            n_tried += 1
            image_path = image_paths_by_idx[hq_idx]
            image = Image.open(image_path).convert("RGB")
            mask = load_celebamask_hq_mask(
                CELEBA_HQ_ROOT, hq_idx, region_names, target_hw=(image.height, image.width)
            )
            if not mask.any():
                continue
            crop = crop_with_padding(image, mask)
            if crop is None:
                continue
            out_path = pos_dir / f"{hq_idx}.jpg"
            crop.save(out_path, quality=90)
            positive_crops[concept_name].append(out_path)

        print(f"{concept_name:<20s} ({'+'.join(region_names)}): {len(positive_crops[concept_name])} "
              f"positive crops saved ({n_tried} candidates tried)", flush=True)

    # Pass 2: contrastive negatives -- for a given concept, drawn evenly
    # from the OTHER 25 groundable concepts' own just-materialized
    # positive crops (real crops of real, different facial regions).
    for concept_name in GROUNDABLE_CONCEPTS:
        own_regions = set(ATTRIBUTE_TO_REGIONS[concept_name])
        other_concepts = [
            c for c in GROUNDABLE_CONCEPTS if set(ATTRIBUTE_TO_REGIONS[c]) != own_regions
        ]
        pool: list[Path] = []
        for other in other_concepts:
            pool.extend(positive_crops[other])
        rng.shuffle(pool)
        neg_dir = OUTPUT_ROOT / concept_name / "negatives"
        neg_dir.mkdir(parents=True, exist_ok=True)
        chosen = pool[:MAX_NEGATIVES_PER_CONCEPT]
        for src_path in chosen:
            dest = neg_dir / f"{src_path.parent.parent.name}__{src_path.name}"
            dest.write_bytes(src_path.read_bytes())
        print(f"{concept_name:<20s}: {len(chosen)} negative crops saved (from {len(pool)} candidates "
              f"across {len(other_concepts)} other concepts)", flush=True)

    print(f"\nConcept bank written to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
