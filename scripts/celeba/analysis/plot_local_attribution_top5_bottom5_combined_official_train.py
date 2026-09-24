"""Official-train counterpart of `plot_local_attribution_top5_bottom5_
combined.py` -- same combined present+absent |magnitude| retrieval
grid, just pointed at the official-train classifier's own present/
absent CSVs. See that script's own docstring for the full design
rationale.

Row label for the masking-hybrid method renamed "hybrid" -> "Hide and
Seek" -- prompted directly ("I changed the name to Hide and Seek, so
the name of the technique needs to be updated"). Superseded by the
by-class version for actual current figures, but kept consistent here
too in case this one gets rerun.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RESULTS_DIR = Path("results")
PRESENT_CSV = RESULTS_DIR / "local_attribution_celeba_official_train_pairs.csv"
ABSENT_CSV = RESULTS_DIR / "local_attribution_celeba_official_train_absent_pairs.csv"
OUT_DIR = RESULTS_DIR / "local_attribution_top5_bottom5_combined_official_train"
METHODS = ["hybrid", "tcav"]
METHOD_DISPLAY_NAMES = {"hybrid": "Hide and Seek", "tcav": "tcav"}
N = 5


def build_image_grid(rows: list[tuple[str, list[tuple[str, float, bool]]]], out_path: Path, thumb=128) -> None:
    n_cols = max(len(images) for _, images in rows)
    n_rows = len(rows)
    pad, label_w, header_h, border = 4, 110, 20, 4
    grid = Image.new("RGB", (label_w + n_cols * (thumb + pad), header_h + n_rows * (thumb + pad + header_h)), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for r, (row_label, images) in enumerate(rows):
        y0 = header_h + r * (thumb + pad + header_h)
        draw.text((2, y0), row_label, fill="black", font=font)
        for c, (img_path, score, is_present) in enumerate(images):
            x0 = label_w + c * (thumb + pad)
            try:
                thumb_img = Image.open(img_path).convert("RGB").resize((thumb, thumb))
            except OSError:
                thumb_img = Image.new("RGB", (thumb, thumb), "gray")
            grid.paste(thumb_img, (x0, y0 + header_h))
            border_color = "lime" if is_present else "red"
            draw.rectangle([x0, y0 + header_h, x0 + thumb - 1, y0 + header_h + thumb - 1],
                            outline=border_color, width=border)
            sign_color = "cyan" if score >= 0 else "orange"
            draw.text((x0, y0 + header_h + thumb - 12), f"{score:+.2f}", fill=sign_color, font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main():
    candidates: dict[str, dict[str, list[tuple[str, float, bool]]]] = defaultdict(lambda: defaultdict(list))

    with open(PRESENT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            for method_name in METHODS:
                candidates[row["concept_name"]][method_name].append(
                    (row["image"], float(row[f"{method_name}_score"]), True)
                )
    with open(ABSENT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            for method_name in METHODS:
                candidates[row["concept_name"]][method_name].append(
                    (row["image"], float(row[f"{method_name}_score"]), False)
                )

    print(f"{len(candidates)} concepts found (present + absent combined, official-train classifier).", flush=True)
    for concept_name in sorted(candidates):
        rows_for_grid = []
        for method_name in METHODS:
            ranked = sorted(candidates[concept_name][method_name], key=lambda kv: -abs(kv[1]))
            top5 = ranked[:N]
            bottom5 = ranked[-N:]
            n_top_present = sum(1 for _, _, present in top5 if present)
            n_bottom_present = sum(1 for _, _, present in bottom5 if present)
            method_label = METHOD_DISPLAY_NAMES[method_name]
            rows_for_grid.append((f"{method_label} (top-5, {n_top_present}/5 present)", top5))
            rows_for_grid.append((f"{method_label} (bottom-5, {n_bottom_present}/5 present)", bottom5))
        build_image_grid(rows_for_grid, OUT_DIR / f"{concept_name}.png")
    print(f"Saved combined top-5/bottom-5 grids to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
