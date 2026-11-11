# ruff: noqa: SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""Side-by-side sheets: original | LaMa | SD-2 | SDXL | FLUX | SD-2 concept-neg | SDXL concept-neg, concept fill (m0), seed 0.
One sheet per concept, 6 random pairs (seed 0); plus one sheet of random-shape fills (m1) for 6 random pairs."""
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paths import BUNDLE, FILLS, REVIEW

sys.path.insert(0, "scripts/celeba/ground_truth")
from PIL import Image, ImageDraw
from run_celeba_official_train_faithfulness import CELEBA_HQ_ROOT

F = FILLS
B = BUNDLE
IMG = CELEBA_HQ_ROOT / "CelebA-HQ-img"
OUT = REVIEW / "inpainters"; OUT.mkdir(parents=True, exist_ok=True)
COLS = [("original", None), ("LaMa", "lama_s0"), ("SD-2", "sd2_quality_s0"), ("SDXL", "sdxl_quality_s0"), ("FLUX", "flux_none_s0"),
        ("SD-2 neg", "sd2_concept_s0"), ("SDXL neg", "sdxl_concept_s0")]
T, PAD, HEAD = 256, 22, 26
rows = list(csv.DictReader(open(B / "manifest.csv")))
by = {}
for r in rows: by.setdefault(r["concept_name"], []).append(r)


def sheet(sel, k, title, path):
    W = len(COLS) * T; im = Image.new("RGB", (W, HEAD + len(sel) * (T + PAD)), "white"); d = ImageDraw.Draw(im)
    d.text((4, 4), title, fill="black")
    for j, (name, _) in enumerate(COLS): d.text((j * T + 4, HEAD - 12), name, fill="black")
    for i, r in enumerate(sel):
        y = HEAD + i * (T + PAD); idx, c = r["hq_idx"], r["concept_name"]
        for j, (_, run) in enumerate(COLS):
            p = IMG / f"{idx}.jpg" if run is None else F / run / f"{idx}_{c}_m{k}.jpg"
            im.paste(Image.open(p).convert("RGB").resize((T, T), Image.LANCZOS), (j * T, y + PAD))
        d.text((4, y + 4), f"#{idx} {c}", fill="black")
    im.save(path, quality=90)


rng = random.Random(0)
for c, rs in sorted(by.items()):
    sheet(rng.sample(rs, min(6, len(rs))), 0, f"{c}: concept region filled (m0)", OUT / f"{c}.jpg")
sheet(rng.sample([r for r in rows if int(r["n_masks"]) > 1], 6), 1, "random-shape control fills (m1)", OUT / "_random_shape_m1.jpg")
print("done")
