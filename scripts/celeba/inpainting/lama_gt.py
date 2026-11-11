# ruff: noqa: C408, SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""LaMa-inpainting ground truth + equal-area random-shape LaMa control.
Run from the repo root. argv[1]=out CSV; env LAMA_LIMIT=n (smoke). Same records/rules as blob_control.py, but fill = LaMa.
Pre-set choices: real mask dilated by DILATE px (at 1024) before inpainting; random shapes have the dilated area and avoid
the dilated mask; LaMa runs at 512 px and its fill is upsampled and composited into the original only inside the mask."""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation

sys.path.insert(0, "src"); sys.path.insert(0, "scripts/celeba/ground_truth")
sys.path.insert(0, str(Path(__file__).parent))
from blob_control import random_blob
from paths import LAMA_OUT, LAMA_WEIGHTS
from run_celeba_official_train_faithfulness import (
    CELEBA_HQ_ROOT,
    CONCEPT_TO_IDX,
    DEVICE,
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

LAMA = LAMA_WEIGHTS
SAVE = LAMA_OUT
OUT = Path(sys.argv[1]) if __name__ == "__main__" else None; LIMIT = int(os.environ.get("LAMA_LIMIT", "0"))
DILATE = 8; RES = 512; TOL = 1e-3
FIELDS = ["image", "concept_name", "target_task", "area", "fill_frac", "n_valid", "p0", "p0_stored", "delta_p", "delta_p_stored",
          "delta_logit", "delta_logodds", "blob_p", "blob_logit", "blob_logodds"]


def make_inpaint(lama):
    @torch.no_grad()
    def inpaint(image, mask):
        x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.).permute(2, 0, 1)[None].to(DEVICE)
        m = torch.from_numpy(mask.astype(np.float32))[None, None].to(DEVICE)
        xs = F.interpolate(x, size=(RES, RES), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        ms = (F.adaptive_max_pool2d(m, (RES, RES)) > 0).float()  # any covered pixel -> masked at low res
        ys = lama(xs * (1 - ms), ms).clamp(0, 1)
        y = F.interpolate(ys, size=x.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        out = x * (1 - m) + y * m
        return Image.fromarray((out[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    return inpaint


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
    SAVE.mkdir(exist_ok=True)
    paths = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    model = load_official_train_native().to(DEVICE).eval(); prep = official_train_preprocess()
    inpaint = make_inpaint(torch.jit.load(str(LAMA), map_location=DEVICE).eval())
    new = not OUT.exists()
    f = open(OUT, "a", newline=""); w = csv.DictWriter(f, FIELDS)
    if new:
        w.writeheader()
    for n, (img, c) in enumerate(todo):
        idx = hq_index(img); image = Image.open(paths[idx]).convert("RGB"); hw = (image.height, image.width)
        mask = load_celebamask_hq_mask(CELEBA_HQ_ROOT, idx, ATTRIBUTE_TO_REGIONS[c], target_hw=hw)
        dmask = binary_dilation(mask, iterations=DILATE) if mask.any() else mask
        rng = np.random.default_rng(SEED + idx * 100 + CONCEPT_TO_IDX[c] + 7)
        blobs = [b for b in (random_blob(dmask, rng) for _ in range(N_RANDOM_DRAWS)) if b is not None]
        real = inpaint(image, dmask)
        real.resize((RES, RES), Image.LANCZOS).save(SAVE / f"{idx}_{c}.jpg", quality=95)
        ims = [image, real] + [inpaint(image, b) for b in blobs]
        out = torch.cat([model(prep(i).unsqueeze(0).to(DEVICE)).cpu() for i in ims])
        for ti, task in enumerate(TASKS):
            if task not in by[(img, c)]:
                continue
            rec = by[(img, c)][task]
            l = out[:, ti * 2: ti * 2 + 2]; p = torch.softmax(l, 1)[:, 1].numpy(); pos = l[:, 1].numpy(); lo = (l[:, 1] - l[:, 0]).numpy()
            if abs(float(p[0]) - float(rec["p0"])) > TOL:  # same model, same original image as the stored GT
                raise RuntimeError(f"{img} {c}/{task}: p0 {p[0]:.5f} != stored {float(rec['p0']):.5f}")
            nb = len(blobs)
            w.writerow(dict(image=img, concept_name=c, target_task=task, area=float(dmask.mean()),
                fill_frac=float(dmask.sum() / max(1, (~dmask).sum())), n_valid=nb, p0=float(p[0]), p0_stored=float(rec["p0"]),
                delta_p=float(p[0] - p[1]), delta_p_stored=float(rec["delta_p"]),
                delta_logit=float(pos[0] - pos[1]), delta_logodds=float(lo[0] - lo[1]),
                blob_p=float((p[0] - p[2:]).mean()) if nb else "", blob_logit=float((pos[0] - pos[2:]).mean()) if nb else "",
                blob_logodds=float((lo[0] - lo[2:]).mean()) if nb else ""))
        f.flush()
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(todo)} ({c})", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
