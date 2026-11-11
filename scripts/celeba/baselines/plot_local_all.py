"""Top-3 / bottom-3 local-attribution grids for every method, in the style of the paper's figures
(results/local_attribution_top3_bottom3_by_class_official_train, made by
scripts/celeba/analysis/plot_local_attribution_top5_bottom5_by_class_official_train.py): per (concept, task), present
and absent records are ranked together by |score|; one row per method shows the 3 highest-ranked images, a divider,
then the 3 lowest. The renderer reproduces that figure's layout (fonts, sizes, divider, borders) here so this script
does not depend on later edits to the shared plot utilities. Hit rates are reported by score_local.py, not drawn.

argv[1] = folder with local_merged.csv (score_local.py); grids go to argv[1]/grids/{concept}_{task}.png.
Optional argv[2:] = concept names to draw (default: all)."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from local_common import hq_path

N = 3
PLOT_METHODS = ["cards", "tcav", "eac", "sae"]  # soft CARDS is scored by score_local.py but not drawn
DISPLAY = {"cards": "Hide and Seek (Ours)", "tcav": "TCAV", "eac": "EAC", "sae": "ConceptSAE"}
BG, PANEL_BG, LABEL, SUB, DIVIDER, BORDER = (250, 250, 250), (255, 255, 255), (30, 30, 30), (90, 90, 90), (0, 0, 0), (60, 60, 60)


def font(bold: bool, size: int):
    for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        for d in (Path(r"C:\Windows\Fonts"), Path("/usr/share/fonts/truetype/dejavu")):
            try:
                return ImageFont.truetype(str(d / name), size)
            except OSError:
                continue
    return ImageFont.load_default()


def thumb_at(grid, draw, path, x0, y0, thumb, border):
    draw.rectangle([x0, y0, x0 + thumb + 2 * border - 1, y0 + thumb + 2 * border - 1], outline=BORDER, width=border)
    try:
        im = Image.open(path).convert("RGB").resize((thumb, thumb))
    except OSError:
        im = Image.new("RGB", (thumb, thumb), "gray")
    grid.paste(im, (x0 + border, y0 + border))


def render(rows, out_path: Path, thumb=150, n=N):
    """rows: [(label, [top image paths], [bottom image paths])]. Same geometry as the paper's figure."""
    left, pad, border = 10, 3, 4
    line1_h, line2_h = 40, 32
    cell_h = thumb + 2 * border
    block_w = n * (thumb + 2 * border + pad)
    divider_w = 26
    row_h = line1_h + line2_h + cell_h + pad
    width = left + 2 * block_w + divider_w + left
    grid = Image.new("RGB", (width, len(rows) * row_h + pad), BG)
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main():
    d = Path(sys.argv[1]); only = set(sys.argv[2:])
    groups = defaultdict(list)
    with open(d / "local_merged.csv", newline="") as f:
        for r in csv.DictReader(f):
            if not only or r["concept_name"] in only:
                groups[r["concept_name"], r["target_task"]].append(r)
    for (c, t), recs in sorted(groups.items()):
        rows = []
        for m in PLOT_METHODS:
            ranked = sorted(recs, key=lambda r: -abs(float(r[m])))
            rows.append((DISPLAY[m], [str(hq_path(r["image"])) for r in ranked[:N]],
                         [str(hq_path(r["image"])) for r in ranked[-N:]]))
        render(rows, d / "grids" / f"{c}_{t}.png")
    print(f"wrote {len(groups)} grids to {d / 'grids'}")


if __name__ == "__main__":
    main()
