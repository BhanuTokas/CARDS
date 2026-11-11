"""Fill ablation for HnS on BAM, with a two-patch test of the fill artifact.

HnS fills each mask with whichever of 7 fills moves the image embedding most
toward the concept. Here each fill is used alone, and "best" is HnS's own
best-of-7 rule (chosen per mask with the concept's query). For every image HnS
retrieves (same masks, cutoffs and seeds as run_hns_bam_masks.py) and every
fill, this writes:

  real  HnS's mask filled with this fill.
  s0    An equal-area random smooth shape outside HnS's mask.
  s1    A second one, outside both HnS's mask and s0.
  s0s1  s0 and s1 both filled, on the same image.

s0 and s1 remove nothing the concept is made of, so their drops measure the
fill artifact. Comparing drop(s0s1) - drop(s0) with drop(s1) tests whether a
second patch adds as much as the first, i.e. whether the artifact saturates.
The shapes are skipped when 2x the mask area is more than MAX_FILL of the
area outside the mask. With "best", the real image reproduces HnS's saved one.

Writes <bam-root>/hns/demean_<d>/fills/<fill>/<kind>/<concept>__<stem>.png and
<out-name>_manifest.csv (concept, file, fill, kind, masked_file, area). Model
outputs come from store_bam_outputs.py with --image-dir .../fills.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from run_hns_bam_masks import ALPHA, CONCEPTS, SEED, retrieve
from run_hns_bam_nulls import MAX_FILL, NULL_SEEDS, best_fill, random_shape

from cards.attribution.localization import (
    concept_zscore_cutoff,
    localize_concept,
    threshold_mask,
)
from cards.attribution.masking_mode import DEFAULT_FILL_STRATEGIES as FILLS
from cards.validation.broden_faithfulness import mask_region


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam-root", type=Path, default=Path(os.environ.get("BAM_ROOT", "../Datasets/BAM")))
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("CARDS_CACHE_DIR", "embedding_cache")))
    parser.add_argument("--demean", choices=["true", "false"], default="true")
    parser.add_argument("--fills", nargs="+", default=list(FILLS) + ["best"], choices=list(FILLS) + ["best"])
    parser.add_argument("--out-name", default="fills", help="manifest is <out-name>_manifest.csv")
    args = parser.parse_args()
    out_dir = args.bam_root / "hns" / f"demean_{args.demean}" / "fills"
    fills = args.fills
    for fill in fills:
        for kind in ("real", "s0", "s1", "s0s1"):
            (out_dir / fill / kind).mkdir(parents=True, exist_ok=True)

    encoder, pool, results = retrieve(args.bam_root, args.cache_dir, args.demean == "true")
    rows = []
    for ci, (concept, result) in enumerate(zip(CONCEPTS, results)):
        images = {i: Image.open(pool.paths[i]).convert("RGB") for i in result.present_indices}
        maps = {i: localize_concept(encoder, im, result.query, (im.height, im.width)) for i, im in images.items()}
        cutoff = concept_zscore_cutoff(list(maps.values()), ALPHA)
        n_shapes = 0
        for i, image in images.items():
            mask = threshold_mask(maps[i], method="fixed", cutoff=cutoff)
            if not mask.any() or mask.all():
                continue
            stem = Path(pool.paths[i]).stem
            masks = {"real": mask}
            if 2 * mask.sum() <= MAX_FILL * (~mask).sum():
                shape_rng = np.random.default_rng(NULL_SEEDS["shape"] + ci * 10_000 + int(i))
                s0 = random_shape(mask, shape_rng)
                s1 = random_shape(mask, shape_rng, exclude=mask | s0)
                masks |= {"s0": s0, "s1": s1}
                n_shapes += 1
            for fill in fills:
                def apply(img, m, rng, fill=fill, result=result):
                    if fill == "best":
                        return best_fill(encoder, img, m, rng, result.query)
                    return mask_region(img, m, strategy=fill, rng=rng)
                rng = np.random.default_rng(SEED + ci * 10_000 + int(i))
                filled = {kind: apply(image, m, rng) for kind, m in masks.items()}
                if "s0" in masks:
                    filled["s0s1"] = apply(filled["s0"], masks["s1"], rng)
                    masks["s0s1"] = masks["s0"] | masks["s1"]
                for kind, im in filled.items():
                    name = f"{fill}/{kind}/{concept}__{stem}.png"
                    im.save(out_dir / name)
                    rows.append({"concept": concept, "file": f"{stem}.jpg", "fill": fill, "kind": kind,
                                 "masked_file": name, "area": float(masks[kind].mean())})
                masks.pop("s0s1", None)
        print(f"{concept}: {len(images)} images, {n_shapes} with room for two shapes", flush=True)

    with open(out_dir.parent / f"{args.out_name}_manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"done: {len(rows)} images over fills {fills}", flush=True)


if __name__ == "__main__":
    main()
