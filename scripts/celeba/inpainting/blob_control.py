# ruff: noqa: C408, SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""Random-shape, equal-area, zero-overlap control. Run from repo root. argv[1]=out CSV; env BLOB_LIMIT=n."""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

sys.path.insert(0, "src"); sys.path.insert(0, "scripts/celeba/ground_truth")
from run_celeba_official_train_faithfulness import (
    CELEBA_HQ_ROOT,
    CONCEPT_TO_IDX,
    DEVICE,
    FILL_STRATEGY,
    N_RANDOM_DRAWS,
    RESULTS_DIR,
    SEED,
    TASKS,
    load_official_train_native,
    official_train_preprocess,
)
from run_celeba_official_train_strict_control import hq_index

from cards.data.celeba import load_celebamask_hq_image_paths, load_celebamask_hq_mask
from cards.data.celeba_attributes import ATTRIBUTE_TO_REGIONS
from cards.validation.broden_faithfulness import mask_region

OUT = Path(sys.argv[1]) if __name__ == "__main__" else None; LIMIT = int(os.environ.get("BLOB_LIMIT", "0"))
GRID = 128; TOL = 1e-3
FIELDS = ["image", "concept_name", "target_task", "area", "fill_frac", "n_valid", "p0", "delta_p", "delta_p_stored", "delta_logit",
          "delta_logodds", "blob_p", "blob_logit", "blob_logodds"]

def random_blob(mask, rng):
    k = int(mask.sum()); h, w = mask.shape
    if k == 0 or k > (~mask).sum():
        return None
    sigma = max(1.0, 0.25 * np.sqrt(k / mask.size) * GRID)
    noise = gaussian_filter(rng.standard_normal((GRID, GRID)), sigma, mode="wrap")
    field = zoom(noise, (h / GRID, w / GRID), order=1)[:h, :w] + 1e-9 * rng.standard_normal((h, w))
    field[mask] = -np.inf
    t = np.partition(field.ravel(), -k)[-k]
    blob = field >= t
    return blob

@torch.no_grad()
def main():
    rows = list(csv.DictReader(open(RESULTS_DIR / "celeba_official_train_faithfulness_non_overlapping.csv", newline="")))
    by = defaultdict(dict)
    for r in rows:
        by[(r["image"], r["concept_name"])][r["target_task"]] = r
    keys = list(by)
    if LIMIT:
        keys = keys[:: max(1, len(keys) // LIMIT)][:LIMIT]
    done = set()
    if OUT.exists():
        done = {(r["image"], r["concept_name"]) for r in csv.DictReader(open(OUT, newline=""))}
    todo = [k for k in keys if k not in done]
    print(f"{len(keys)} image-concept pairs, {len(todo)} to run", flush=True)
    paths = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    model = load_official_train_native().to(DEVICE).eval(); prep = official_train_preprocess()
    new = not OUT.exists()
    f = open(OUT, "a", newline=""); w = csv.DictWriter(f, FIELDS)
    if new:
        w.writeheader()
    for n, (img, c) in enumerate(todo):
        idx = hq_index(img); image = Image.open(paths[idx]).convert("RGB"); hw = (image.height, image.width)
        mask = load_celebamask_hq_mask(CELEBA_HQ_ROOT, idx, ATTRIBUTE_TO_REGIONS[c], target_hw=hw)
        rng = np.random.default_rng(SEED + idx * 100 + CONCEPT_TO_IDX[c] + 7)
        blobs = [b for b in (random_blob(mask, rng) for _ in range(N_RANDOM_DRAWS)) if b is not None]
        ims = [image, mask_region(image, mask, strategy=FILL_STRATEGY, rng=rng)] + [mask_region(image, b, strategy=FILL_STRATEGY, rng=rng) for b in blobs]
        out = torch.cat([model(prep(i).unsqueeze(0).to(DEVICE)).cpu() for i in ims])
        for ti, task in enumerate(TASKS):
            if task not in by[(img, c)]:
                continue
            rec = by[(img, c)][task]
            l = out[:, ti * 2: ti * 2 + 2]; p = torch.softmax(l, 1)[:, 1].numpy(); pos = l[:, 1].numpy(); lo = (l[:, 1] - l[:, 0]).numpy()
            dp = float(p[0] - p[1]); stored = float(rec["delta_p"])
            if abs(dp - stored) > TOL:
                raise RuntimeError(f"{img} {c}/{task}: {dp:.5f} != {stored:.5f}")
            nb = len(blobs)
            w.writerow(dict(image=img, concept_name=c, target_task=task, area=float(mask.mean()),
                fill_frac=float(mask.sum() / max(1, (~mask).sum())), n_valid=nb, p0=float(p[0]), delta_p=dp, delta_p_stored=stored,
                delta_logit=float(pos[0] - pos[1]), delta_logodds=float(lo[0] - lo[1]),
                blob_p=float((p[0] - p[2:]).mean()) if nb else "", blob_logit=float((pos[0] - pos[2:]).mean()) if nb else "",
                blob_logodds=float((lo[0] - lo[2:]).mean()) if nb else ""))
        f.flush()
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(todo)} ({c})", flush=True)
    print("done", flush=True)

if __name__ == "__main__":
    main()
