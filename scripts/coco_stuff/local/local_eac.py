# ruff: noqa: SIM115
"""Local (per-image) EAC on the COCO-Stuff local records (common.py), ResNet18 black box.

Per present image: SAM ViT-H segments at 224 px (cached in $CARDS_SAM_CACHE), EAC segment values for all 20 classes
with exactly eac_coco.py's procedure (2,500 Bernoulli subsets seeded by the image's pool index as in eac_coco.py, PIE
surrogate on the frozen head, 50k Monte-Carlo coalitions), so an image that eac_coco.py also used gets the same values.
Local score of (image, concept, class) = sum of the class's segment values weighted by each segment's pixel share
inside the concept's Grounding DINO + SAM region, the region the ground truth masked (read from $GT_MASKS_DIR, saved
by compute_coco_stuff_faithfulness_variants.py).
Absent records are not run: EAC has no region for a concept that was not detected, so their score is 0 (set in
score_local.py). Present images without a saved region get an empty score and are counted in the log.

argv[1] = SAM checkpoint. Sharded over present images; writes $COCO_LOCAL_OUT/eac_{SHARD}.csv (resumable).
Needs segment-anything importable (PYTHONPATH) and GT_MASKS_DIR set."""
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, "scripts/coco_stuff/baselines")
import common as C

if not os.environ.get("GT_MASKS_DIR"):
    raise SystemExit("set GT_MASKS_DIR to the saved ground-truth concept masks")
sys.path.insert(0, "scripts/coco_stuff/build")
import eac_coco as E
from coco_common import CLASSES, head, load_black_box, pool_paths


def main():
    ckpt = sys.argv[1]
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    pool_index = {Path(p).name: i for i, p in enumerate(pool_paths())}
    todo = C.shard(C.groups(C.records("present"), by=("image",)))
    print(f"{len(todo)} present images in this shard", flush=True)
    model, _ = load_black_box(); w, b = head(model).weight.detach(), head(model).bias.detach()
    gen = SamAutomaticMaskGenerator(sam_model_registry["vit_h"](checkpoint=ckpt).to(E.DEVICE))
    out_path = C.OUT / f"eac_{C.shard_tag()}.csv"
    done = set()
    if out_path.exists():
        done = {r["image"] for r in csv.DictReader(open(out_path))}
    fh = open(out_path, "a", newline=""); wr = csv.writer(fh)
    if not done:
        wr.writerow([*C.BASE_COLS, "eac", "n_segments"])
    print(f"resuming with {len(done)} images done", flush=True)
    start, missing = time.time(), 0
    for n, (_, ((name,), recs)) in enumerate(todo):
        if name in done:
            continue
        path = C.image_path(name)
        regions = E.saved_regions(str(path)) or {}
        if not any(r["concept_name"] in regions for r in recs):
            missing += len(recs)
            for r in recs:
                wr.writerow([*C.base(r), "", 0])
            fh.flush(); continue
        i = pool_index[name]
        raw = np.asarray(Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
        masks = E.sam_masks(gen, path.stem, raw)
        subsets = np.random.default_rng(E.SEED + i).random((E.N_SAMPLES, len(masks))) < 0.5
        values = E.fit_values(subsets, E.model_probs(model, raw, masks, subsets), w, b)  # (20 classes, n segments)
        area = masks.sum((1, 2)).clip(min=1)
        for r in recs:
            reg = regions.get(r["concept_name"])
            if reg is None:
                missing += 1; wr.writerow([*C.base(r), "", len(masks)]); continue
            r224 = np.asarray(Image.fromarray(reg).resize((224, 224), Image.Resampling.NEAREST))
            frac = (masks & r224).sum((1, 2)) / area
            wr.writerow([*C.base(r), float(values[CLASSES.index(r["class_name"])] @ frac), len(masks)])
        fh.flush()
        if (n + 1) % 20 == 0:
            print(f"  {n + 1}/{len(todo)} images in {time.time() - start:.0f}s; {missing} records without a region",
                  flush=True)
    fh.close(); print(f"done; {missing} present records without a saved region", flush=True)


if __name__ == "__main__":
    main()
