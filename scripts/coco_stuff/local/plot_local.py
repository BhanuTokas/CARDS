"""Top-3 / bottom-3 local-attribution grids for every method on COCO-Stuff, in the style of the CelebA local figures
(scripts/celeba/analysis/local_attribution_plot_utils.build_presence_grid_clean, same geometry and fonts as
local_attribution_concepts.py's port): per (concept, class), present and absent records are ranked together by
|score|; one row per method shows the 3 highest-ranked images, a divider, then the 3 lowest. Thumbnails are
center-cropped to a square instead of stretched (COCO images are not square). Ties in |score| (TCAV sign, EAC on
absent records) are broken by a fixed random order, not file order, which would put present records first.

argv[1] (optional, default $COCO_LOCAL_OUT) = folder with local_merged.csv (score_local.py); grids go to
argv[1]/grids/{concept}_{class}.png. Optional argv[2:] = concept names to draw (default: all).
env LOCAL_PLOT_METHODS = comma-separated rows to draw (default all four), e.g. "cards,tcav_sign,sae" to leave out EAC,
whose regions are the ground truth's own concept masks; env LOCAL_GRIDS = output subfolder name (default grids);
env LOCAL_THUMB = crop (default) | stretch (as the paper's figures) | pad; env LOCAL_PRESENCE=1 = green / red borders for
records where the ground-truth detector found / did not find the concept."""
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from common import image_path

N = 3
DISPLAY = {"cards": "Hide and Seek (Ours)", "tcav_sign": "TCAV", "tcav_dd": "TCAV", "tcav4_dd": "TCAV (layer4)",
           "eac": "EAC", "sae": "ConceptSAE"}  # tcav_sign / tcav_dd = layer3 sign score / directional derivative
PLOT_METHODS = [m.strip() for m in os.environ.get("LOCAL_PLOT_METHODS", "cards,tcav_sign,eac,sae").split(",") if m.strip()]
assert PLOT_METHODS and set(PLOT_METHODS) <= set(DISPLAY), f"LOCAL_PLOT_METHODS must be from {sorted(DISPLAY)}"
EXPLICIT = "LOCAL_PLOT_METHODS" in os.environ  # default list: drop methods that were not run (e.g. no EAC)
GRIDS = os.environ.get("LOCAL_GRIDS", "grids")
THUMB = os.environ.get("LOCAL_THUMB", "crop")  # crop | stretch (the paper's figures) | pad
assert THUMB in ("crop", "stretch", "pad"), f"LOCAL_THUMB must be crop, stretch or pad, got {THUMB!r}"
BG, PANEL_BG, LABEL, SUB, DIVIDER, BORDER = (250, 250, 250), (255, 255, 255), (30, 30, 30), (90, 90, 90), (0, 0, 0), (60, 60, 60)
# env LOCAL_PRESENCE=1: border each thumbnail by the record's set -- green = concept detected in the image by the ground-
# truth detector (Grounding DINO), red = not detected -- and add a one-line legend under the grid.
PRESENCE = os.environ.get("LOCAL_PRESENCE", "0") == "1"
PRESENT_C, ABSENT_C = (26, 150, 65), (215, 25, 28)


def font(bold: bool, size: int):
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    dirs = [Path(r"C:\Windows\Fonts"), Path("/usr/share/fonts/truetype/dejavu")]
    try:
        import matplotlib
        dirs.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf")  # bundled DejaVu, present on any machine
    except ImportError:
        pass
    for name in names:
        for d in dirs:
            try:
                return ImageFont.truetype(str(d / name), size)
            except OSError:
                continue
    print("warning: no TrueType font found, using PIL's bitmap default")
    return ImageFont.load_default()


def thumb_at(grid, draw, item, x0, y0, thumb, border):
    path, present = item
    color = (PRESENT_C if present else ABSENT_C) if PRESENCE else BORDER
    draw.rectangle([x0, y0, x0 + thumb + 2 * border - 1, y0 + thumb + 2 * border - 1], outline=color, width=border)
    try:
        src = Image.open(path).convert("RGB")
        if THUMB == "stretch":  # as the paper's figures (by_concept_across_classes script): whole image, aspect distorted
            im = src.resize((thumb, thumb), Image.Resampling.BICUBIC)
        elif THUMB == "pad":    # whole image, aspect kept, panel-coloured bars
            im = ImageOps.pad(src, (thumb, thumb), Image.Resampling.BICUBIC, color=PANEL_BG)
        else:                   # crop: centre square, aspect kept, edges cut
            im = ImageOps.fit(src, (thumb, thumb), Image.Resampling.BICUBIC)
    except OSError:
        im = Image.new("RGB", (thumb, thumb), "gray")
    grid.paste(im, (x0 + border, y0 + border))


def render(rows, out_path: Path, thumb=150, n=N):
    left, pad, border = 10, 3, (7 if PRESENCE else 4)
    line1_h, line2_h = 40, 32
    cell_h = thumb + 2 * border
    block_w = n * (thumb + 2 * border + pad)
    divider_w = 26
    row_h = line1_h + line2_h + cell_h + pad
    width = left + 2 * block_w + divider_w + left
    legend_h = 44 if PRESENCE else 0
    grid = Image.new("RGB", (width, len(rows) * row_h + pad + legend_h), BG)
    draw = ImageDraw.Draw(grid)
    row_font, sub_font = font(True, 30), font(False, 28)
    bottom_x0 = left + block_w + divider_w
    for r, (label, top, bottom) in enumerate(rows):
        y0 = r * row_h
        draw.rectangle([0, y0, width, y0 + row_h - pad], fill=PANEL_BG if r % 2 == 0 else BG)
        draw.text((left, y0 + 4), label, fill=LABEL, font=row_font)
        sub_y, cell_y0 = y0 + line1_h + 2, y0 + line1_h + line2_h
        draw.text((left, sub_y), f"Top {n}", fill=SUB, font=sub_font)
        for c, p in enumerate(top):
            thumb_at(grid, draw, p, left + c * (thumb + 2 * border + pad), cell_y0, thumb, border)
        div_x = left + block_w + divider_w // 2 - 2  # PIL draws thick lines slightly right of x
        draw.line([(div_x, y0), (div_x, y0 + row_h - pad)], fill=DIVIDER, width=6)
        draw.text((bottom_x0, sub_y), f"Bottom {n}", fill=SUB, font=sub_font)
        for c, p in enumerate(bottom):
            thumb_at(grid, draw, p, bottom_x0 + c * (thumb + 2 * border + pad), cell_y0, thumb, border)
    if PRESENCE:  # legend: coloured swatch + text, twice
        y, x, f = len(rows) * row_h + pad + 8, left, font(False, 24)
        for col, text in ((PRESENT_C, "concept detected"), (ABSENT_C, "concept not detected")):
            draw.rectangle([x, y + 2, x + 26, y + 26], outline=col, width=5)
            draw.text((x + 36, y), text, fill=SUB, font=f)
            x += 36 + int(draw.textlength(text, font=f)) + 40
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("COCO_LOCAL_OUT", "results/coco_stuff_local/resnet18"))
    only = set(sys.argv[2:])
    groups = defaultdict(list)
    with open(d / "local_merged.csv", newline="") as f:
        rd = csv.DictReader(f)
        missing = [m for m in PLOT_METHODS if m not in rd.fieldnames]
        assert not (EXPLICIT and missing), f"not in local_merged.csv: {missing}"
        methods = [m for m in PLOT_METHODS if m not in missing]
        for r in rd:
            if not only or r["concept_name"] in only:
                groups[r["concept_name"], r["class_name"]].append(r)
    for (c, k), recs in sorted(groups.items()):
        recs = recs[:]; random.Random(f"{c}/{k}").shuffle(recs)  # random tie order
        rows = []
        for m in methods:
            ranked = sorted(recs, key=lambda r: -abs(float(r[m])))
            rows.append((DISPLAY[m], [(str(image_path(r["image"])), r["set"] == "present") for r in ranked[:N]],
                         [(str(image_path(r["image"])), r["set"] == "present") for r in ranked[-N:]]))
        render(rows, d / GRIDS / f"{c}_{k.replace(' ', '_')}.png")
    print(f"wrote {len(groups)} grids ({', '.join(methods)}) to {d / GRIDS}")


if __name__ == "__main__":
    main()
