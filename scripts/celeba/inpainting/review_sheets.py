# ruff: noqa: SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""One sheet per concept: 12 random labelled-positive pairs (seed 0), original | LaMa, captioned with the detector drop."""
import collections
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from paths import LAMA_OUT, REVIEW, WORK

sys.path.insert(0, "scripts/celeba/ground_truth")
from PIL import Image, ImageDraw
from run_celeba_official_train_faithfulness import CELEBA_HQ_ROOT

R = REVIEW / "lama"; R.mkdir(parents=True, exist_ok=True)
L = LAMA_OUT; H = CELEBA_HQ_ROOT / "CelebA-HQ-img"
by = collections.defaultdict(list)
for r in csv.DictReader(open(WORK / "removal_full.csv")):
    if r["label"] == "1": by[r["concept_name"]].append(r)
T = 256
for c, v in by.items():
    rng = np.random.default_rng(0); pick = [v[i] for i in sorted(rng.choice(len(v), min(12, len(v)), replace=False))]
    t = float(pick[0]["threshold"]); so = np.median([float(r["s_orig"]) for r in v]); gap = so - t
    W = Image.new("RGB", (T * 2 * 3 + 20, (T + 22) * 4 + 30), "white"); d = ImageDraw.Draw(W)
    d.text((6, 6), f"{c}: left = original, right = LaMa. 'drop' 1.0 = pushed to the detector's 'absent' boundary; 'gone' = below it", fill="black")
    for k, r in enumerate(pick):
        x, y = (k % 3) * (T * 2 + 10), 30 + (k // 3) * (T + 22)
        W.paste(Image.open(H / f"{r['hq_idx']}.jpg").convert("RGB").resize((T, T)), (x, y))
        W.paste(Image.open(L / f"{r['hq_idx']}_{c}.jpg").convert("RGB").resize((T, T)), (x + T, y))
        dr = (float(r["s_orig"]) - float(r["s_lama"])) / gap
        d.text((x + 4, y + T + 4), f"#{r['hq_idx']}  drop {dr:+.2f}  {'gone' if float(r['s_lama']) <= t else 'still detected'}", fill="black")
    W.save(R / f"{c}.jpg", quality=92)
print("wrote", len(by), "sheets to", R)
