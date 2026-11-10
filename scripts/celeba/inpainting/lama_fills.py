# ruff: noqa: SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""LaMa fills in the Sol output format (FILLS/lama_s0/{idx}_{c}_m{k}.jpg, 512 px JPEG q95) from the exported masks,
so LaMa and the diffusion inpainters are scored through the identical save + score path. Run from the repo root."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "src"); sys.path.insert(0, "scripts/celeba/ground_truth"); sys.path.insert(0, str(Path(__file__).parent))
from lama_gt import LAMA, make_inpaint
from paths import BUNDLE, FILLS
from run_celeba_official_train_faithfulness import CELEBA_HQ_ROOT, DEVICE

B = BUNDLE; OUT = FILLS / "lama_s0"; OUT.mkdir(parents=True, exist_ok=True)
inpaint = make_inpaint(torch.jit.load(str(LAMA), map_location=DEVICE).eval())
rows = list(csv.DictReader(open(B / "manifest.csv")))
for n, r in enumerate(rows):
    i, c, nm = int(r["hq_idx"]), r["concept_name"], int(r["n_masks"])
    if all((OUT / f"{i}_{c}_m{k}.jpg").exists() for k in range(nm)): continue
    image = Image.open(CELEBA_HQ_ROOT / "CelebA-HQ-img" / f"{i}.jpg").convert("RGB")
    for k in range(nm):
        m = np.array(Image.open(B / "masks" / f"{i}_{c}_m{k}.png")).astype(bool)
        inpaint(image, m).resize((512, 512), Image.LANCZOS).save(OUT / f"{i}_{c}_m{k}.jpg", quality=95)
    if (n + 1) % 500 == 0: print(f"  {n + 1}/{len(rows)}", flush=True)
print("done", flush=True)
