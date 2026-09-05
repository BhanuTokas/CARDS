"""Materializes a part-concept bank for CUB from CUB70-PartSegmentationDataset's
real masks -- positives/negatives crop folders in the exact layout
`post_hoc_cbm`'s own loaders expect (`<root>/<concept>/{positives,negatives}/*.jpg`),
with genuinely contrastive negatives from the start (drawn from OTHER
parts' own real crops, not an unrelated grab-bag -- applying the v20
lesson proactively instead of discovering the flaw after the fact: every
non-target part here is a real, different-concept crop from a real bird,
so there's no "general/random" pool needed the way Broden's object
categories required one).

Only the 8 CUB70 parts with an unambiguous 1:1 CUB-keypoint match are
used (see cards.data.cub_parts.CUB70_TO_CUB_PART_ID), for consistency
with v33's validated faithfulness approximation and the rest of the CUB
battery.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cards.data.cub_parts import (
    CUB70_TO_CUB_PART_ID,
    load_cub70_mask,
    load_images_txt,
)

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
CUB70_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB70_part_segmentation")
OUTPUT_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\cub_part_concepts")
SEED = 42
MIN_COVERAGE_FRAC = 0.002  # CUB70 part masks (e.g. eyes) are naturally tiny relative to Broden's object masks
PADDING_FRAC = 0.25
MIN_CROP_PX = 20
MAX_POSITIVES_PER_CONCEPT = 150
MAX_NEGATIVES_PER_CONCEPT = 150


def crop_with_padding(image: Image.Image, mask) -> Image.Image | None:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    coverage = mask.sum() / mask.size
    if coverage < MIN_COVERAGE_FRAC:
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

    print("Loading CUB images.txt...", flush=True)
    image_paths = load_images_txt(CUB_ROOT)
    stem_to_id = {p.stem: image_id for image_id, p in image_paths.items()}

    # Discover all (part, class_dir, stem) mask instances directly from the
    # extracted archive.
    all_instances: dict[str, list[tuple[str, str]]] = {p: [] for p in CUB70_TO_CUB_PART_ID}
    for class_dir in sorted((CUB70_ROOT / "AnnotationMasksPerclass").iterdir(), key=lambda p: int(p.name)):
        for mask_file in class_dir.glob("*.png"):
            name = mask_file.stem
            for part_name in CUB70_TO_CUB_PART_ID:
                suffix = f"_{part_name}"
                if name.endswith(suffix):
                    stem = name[: -len(suffix)]
                    if stem in stem_to_id:
                        all_instances[part_name].append((class_dir.name, stem))
                    break

    # Pass 1: materialize positive crops for every part.
    positive_crops: dict[str, list[Path]] = {p: [] for p in CUB70_TO_CUB_PART_ID}
    for part_name, instances in all_instances.items():
        rng.shuffle(instances)
        pos_dir = OUTPUT_ROOT / part_name / "positives"
        pos_dir.mkdir(parents=True, exist_ok=True)
        for class_dir, stem in instances:
            if len(positive_crops[part_name]) >= MAX_POSITIVES_PER_CONCEPT:
                break
            mask = load_cub70_mask(CUB70_ROOT, class_dir, stem, part_name)
            if mask is None or not mask.any():
                continue
            image_id = stem_to_id[stem]
            image_path = image_paths[image_id]
            if not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            if mask.shape != (image.height, image.width):
                continue
            crop = crop_with_padding(image, mask)
            if crop is None:
                continue
            out_path = pos_dir / f"{stem}.jpg"
            crop.save(out_path, quality=90)
            positive_crops[part_name].append(out_path)
        print(f"{part_name}: {len(positive_crops[part_name])} positive crops saved "
              f"(of {len(instances)} raw instances)", flush=True)

    # Pass 2: contrastive negatives -- for a given part, drawn evenly from
    # the OTHER 7 parts' own just-materialized positive crops (real crops
    # of real, different bird parts -- exactly the kind of hard negative
    # v20 found missing for "car").
    for part_name in CUB70_TO_CUB_PART_ID:
        other_parts = [p for p in CUB70_TO_CUB_PART_ID if p != part_name]
        pool: list[Path] = []
        for other in other_parts:
            pool.extend(positive_crops[other])
        rng.shuffle(pool)
        neg_dir = OUTPUT_ROOT / part_name / "negatives"
        neg_dir.mkdir(parents=True, exist_ok=True)
        chosen = pool[:MAX_NEGATIVES_PER_CONCEPT]
        for src_path in chosen:
            dest = neg_dir / f"{src_path.parent.parent.name}__{src_path.name}"
            dest.write_bytes(src_path.read_bytes())
        print(f"{part_name}: {len(chosen)} negative crops saved (from {len(pool)} candidates "
              f"across {len(other_parts)} other parts)", flush=True)

    print(f"\nConcept bank written to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
