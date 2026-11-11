# ruff: noqa: SIM115
# One-shot research script: bare open() on small CSVs is intentional.

"""Local (per-image) EAC on the clean CelebA ground truth (official-train black box, env CARDS_ARCH).

Per ground-truth image: SAM segments (cached in $CARDS_SAM_CACHE when set), EAC segment values for P(Attractive) and
P(Male) with exactly celeba_eac.py's procedure (2,500 Bernoulli subsets seeded by the image index, PIE surrogate on the
frozen head, 50k Monte-Carlo coalitions). The image's local score for a concept = sum of segment values weighted by
each segment's pixel share inside that concept's CelebAMask-HQ regions (as in celeba_eac.py), for every
(concept, task) the image has a ground-truth record for.

argv[1] = SAM checkpoint, argv[2] = output folder; writes eac_{SHARD}.csv. Sharded over images (local_common.py).
Needs segment-anything on PYTHONPATH."""
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "src"); sys.path.insert(0, str(Path(__file__).parent))
import celeba_eac as E
from local_common import gt_pairs, hq_path, shard, shard_tag
from sam_cache import sam_masks
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from cards.data.celeba import load_celebamask_hq_mask
from cards.data.celeba_attributes import ATTRIBUTE_TO_REGIONS


def main():
    ckpt, out_dir = sys.argv[1], Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
    by_image = defaultdict(list)
    for name, c, tasks in gt_pairs():
        by_image[name].append((c, tasks))
    todo = shard(list(by_image))
    print(f"{len(by_image)} images, {len(todo)} in this shard", flush=True)
    model = E.load().to(E.DEV).eval()
    print("black box:", E.ARCH, flush=True)
    w, b = E.head(model).weight.detach(), E.head(model).bias.detach()
    gen = SamAutomaticMaskGenerator(sam_model_registry["default"](checkpoint=ckpt).to(E.DEV))
    out_path = out_dir / f"eac_{shard_tag()}.csv"
    done = set()
    if out_path.exists():
        done = {r["image"] for r in csv.DictReader(open(out_path))}
    fh = open(out_path, "a", newline=""); wr = csv.writer(fh)
    if not done:
        wr.writerow(["image", "concept_name", "target_task", "gt_delta_p", "eac", "n_segments"])
    print(f"resuming with {len(done)} images done", flush=True)
    start = time.time()
    for n, (_, name) in enumerate(todo):
        if name in done:
            continue
        hq = int(Path(name).stem)
        raw = np.asarray(Image.open(hq_path(name)).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
        masks = sam_masks(gen, hq, raw)
        subsets = np.random.default_rng(E.SEED + hq).random((E.N_SAMPLES, len(masks))) < 0.5
        values = E.fit_values(subsets, E.model_probs(model, raw, masks, subsets), w, b)  # (2 tasks, n segments)
        area = masks.sum((1, 2))
        for c, tasks in by_image[name]:
            region = load_celebamask_hq_mask(E.HQ, hq, ATTRIBUTE_TO_REGIONS[c], target_hw=(224, 224))
            frac = (masks & region).sum((1, 2)) / area
            for i, t in enumerate(E.PAIRS):
                if t in tasks:
                    wr.writerow([name, c, t, tasks[t], float(values[i] @ frac), len(masks)])
        fh.flush()
        if (n + 1) % 20 == 0:
            print(f"  {n + 1}/{len(todo)} images in {time.time() - start:.0f}s", flush=True)
    fh.close(); print("done", flush=True)


if __name__ == "__main__":
    main()
