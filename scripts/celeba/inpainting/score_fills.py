# ruff: noqa: C408, SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""Score saved fills with the ResNet18 black box. argv[1] = fill folder (e.g. ...\\fills\\sdxl_s0), argv[2] = out CSV.
Same columns as lama_full.csv. p0 = original full-resolution image (checked against the stored GT p0); fills = the saved
512 px JPEGs. delta_* = original minus concept-region fill; blob_* = mean over random-shape fills of original minus fill.
Pairs with any missing file are skipped and counted. Run from the repo root."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, "src"); sys.path.insert(0, "scripts/celeba/ground_truth")
from run_celeba_official_train_faithfulness import (
    CELEBA_HQ_ROOT,
    DEVICE,
    RESULTS_DIR,
    TASKS,
    load_official_train_native,
    official_train_preprocess,
)
from run_celeba_official_train_strict_control import hq_index

D, OUT = Path(sys.argv[1]), Path(sys.argv[2]); TOL = 1e-3
sys.path.insert(0, str(Path(__file__).parent))
from paths import BUNDLE as B  # manifest.csv + masks/

FIELDS = ["image", "concept_name", "target_task", "n_valid", "p0", "p0_stored", "delta_p", "delta_logit", "delta_logodds",
          "blob_p", "blob_logit", "blob_logodds"]
by = defaultdict(dict)
for r in csv.DictReader(open(RESULTS_DIR / "celeba_official_train_faithfulness_non_overlapping.csv", newline="")):
    by[(hq_index(r["image"]), r["concept_name"])][r["target_task"]] = r
model = load_official_train_native().to(DEVICE).eval(); prep = official_train_preprocess()
miss = 0
with torch.no_grad(), open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, FIELDS); w.writeheader()
    for n, r in enumerate(csv.DictReader(open(B / "manifest.csv"))):
        i, c, nm = int(r["hq_idx"]), r["concept_name"], int(r["n_masks"])
        files = [D / f"{i}_{c}_m{k}.jpg" for k in range(nm)]
        if not all(p.exists() for p in files): miss += 1; continue
        ims = [Image.open(CELEBA_HQ_ROOT / "CelebA-HQ-img" / f"{i}.jpg").convert("RGB")] + [Image.open(p).convert("RGB") for p in files]
        out = torch.cat([model(prep(x).unsqueeze(0).to(DEVICE)).cpu() for x in ims])  # one at a time, as the stored GT was scored
        for ti, task in enumerate(TASKS):
            if task not in by[(i, c)]: continue
            rec = by[(i, c)][task]
            l = out[:, ti * 2: ti * 2 + 2]; p = torch.softmax(l, 1)[:, 1].numpy(); pos = l[:, 1].numpy(); lo = (l[:, 1] - l[:, 0]).numpy()
            if abs(float(p[0]) - float(rec["p0"])) > TOL: raise RuntimeError(f"{i} {c}/{task}: p0 {p[0]:.5f} != {rec['p0']}")
            nb = nm - 1
            w.writerow(dict(image=rec["image"], concept_name=c, target_task=task, n_valid=nb, p0=float(p[0]), p0_stored=float(rec["p0"]),
                delta_p=float(p[0] - p[1]), delta_logit=float(pos[0] - pos[1]), delta_logodds=float(lo[0] - lo[1]),
                blob_p=float((p[0] - p[2:]).mean()) if nb else "", blob_logit=float((pos[0] - pos[2:]).mean()) if nb else "",
                blob_logodds=float((lo[0] - lo[2:]).mean()) if nb else ""))
        if (n + 1) % 1000 == 0: print(f"  {n + 1}", flush=True)
print(f"done; {miss} pairs skipped for missing files", flush=True)
