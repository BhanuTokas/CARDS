# ruff: noqa: SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""Export the exact masks lama_gt.py inpainted (dilated real mask + its random-shape draws) for the Sol inpainting runs.
Run from the repo root. Writes BUNDLE/masks/{idx}_{concept}_m{k}.png (1-bit, image resolution; k=0 real, k>=1 random shapes)
and BUNDLE/manifest.csv (hq_idx, concept_name, n_masks). Checks n_masks-1 against lama_full.csv's n_valid."""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

sys.path.insert(0, "src"); sys.path.insert(0, "scripts/celeba/ground_truth"); sys.path.insert(0, str(Path(__file__).parent))
from blob_control import random_blob
from lama_gt import DILATE
from paths import BUNDLE, WORK
from run_celeba_official_train_faithfulness import (
    CELEBA_HQ_ROOT,
    CONCEPT_TO_IDX,
    N_RANDOM_DRAWS,
    RESULTS_DIR,
    SEED,
)
from run_celeba_official_train_strict_control import hq_index

from cards.data.celeba import load_celebamask_hq_image_paths, load_celebamask_hq_mask
from cards.data.celeba_attributes import ATTRIBUTE_TO_REGIONS

M = BUNDLE / "masks"; M.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(RESULTS_DIR / "celeba_official_train_faithfulness_non_overlapping.csv", newline="")))
pairs = list(dict.fromkeys((r["image"], r["concept_name"]) for r in rows))  # same order as lama_gt.py
lama_nv = {(r["image"], r["concept_name"]): int(r["n_valid"])
           for r in csv.DictReader(open(WORK / "lama_full.csv", newline=""))}
paths = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
with open(BUNDLE / "manifest.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["hq_idx", "concept_name", "n_masks"])
    for n, (img, c) in enumerate(pairs):
        idx = hq_index(img); hw = Image.open(paths[idx]).size[::-1]
        mask = load_celebamask_hq_mask(CELEBA_HQ_ROOT, idx, ATTRIBUTE_TO_REGIONS[c], target_hw=hw)
        dmask = binary_dilation(mask, iterations=DILATE) if mask.any() else mask
        rng = np.random.default_rng(SEED + idx * 100 + CONCEPT_TO_IDX[c] + 7)
        blobs = [b for b in (random_blob(dmask, rng) for _ in range(N_RANDOM_DRAWS)) if b is not None]
        assert len(blobs) == lama_nv[(img, c)], (img, c, len(blobs), lama_nv[(img, c)])
        for k, m in enumerate([dmask] + blobs):
            Image.fromarray(m).save(M / f"{idx}_{c}_m{k}.png", optimize=True)
        w.writerow([idx, c, 1 + len(blobs)])
        if (n + 1) % 500 == 0: print(f"  {n + 1}/{len(pairs)}", flush=True)
print("done", flush=True)
