"""Top-5 AND bottom-5 (by |magnitude|) per-concept image grids for the
local attribution experiment, built directly from the already-scored
`local_attribution_celeba_pairs_corrected_gt.csv` -- no GPU/model
re-computation needed, since every (image, concept, task) pair's
ground_truth/hybrid/tcav score is already saved there from the last run
of `local_attribution_comparison_celeba_corrected_gt.py`. Prompted
directly ("For the local attribution experiment, can we show the top-5
and bottom-5 rated images for each concept by abs magnitude?").

Extends that script's own top-10-by-|magnitude| grid (which only shows
the 10 HIGHEST-magnitude images, mixing sign) by ALSO showing the 5
LOWEST-magnitude images per method -- the question that grid alone
can't answer: what do the images each method is LEAST confident about
actually look like (genuinely ambiguous content, or does the method
just fail there)?

Deliberately does NOT import `build_image_grid` from that script --
doing so would drag in captum/post_hoc_cbm/torch/the whole encoder
pipeline just to reuse one ~25-line PIL utility function, for a script
whose entire point is being pure CSV+PIL post-processing with no
GPU/model dependency at all. Duplicated locally instead (kept in sync
by inspection, not by import).

Images from BOTH target tasks (Attractive, Young) are pooled into one
per-concept ranking, matching the original top-10 script's own
convention (not re-litigated here). `baseline` is excluded from the
grid, same convention too -- ground_truth/hybrid/tcav only.

Row label for the masking-hybrid method renamed "hybrid" -> "Hide and
Seek" -- prompted directly ("I changed the name to Hide and Seek, so
the name of the technique needs to be updated").
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RESULTS_DIR = Path("results")
PAIRS_CSV = RESULTS_DIR / "local_attribution_celeba_pairs_corrected_gt.csv"
OUT_DIR = RESULTS_DIR / "local_attribution_top5_bottom5_by_magnitude"
METHOD_COLUMN = {"ground_truth": "gt_delta_p", "hybrid": "hybrid_score", "tcav": "tcav_score"}
METHOD_DISPLAY_NAMES = {"ground_truth": "ground_truth", "hybrid": "Hide and Seek", "tcav": "tcav"}
N = 5


def build_image_grid(rows: list[tuple[str, list[tuple[str, float]]]], out_path: Path, thumb=128) -> None:
    """rows: [(row_label, [(image_path, score), ...]), ...] -- one row per
    (method, top-5-or-bottom-5), up to 5 columns. Sign shown directly:
    green score text = positive ("helps"), red = negative ("hurts") --
    same convention as the top-10-by-magnitude grid this extends."""
    n_cols = max(len(images) for _, images in rows)
    n_rows = len(rows)
    pad, label_w, header_h = 4, 110, 20
    grid = Image.new("RGB", (label_w + n_cols * (thumb + pad), header_h + n_rows * (thumb + pad + header_h)), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for r, (row_label, images) in enumerate(rows):
        y0 = header_h + r * (thumb + pad + header_h)
        draw.text((2, y0), row_label, fill="black", font=font)
        for c, (img_path, score) in enumerate(images):
            x0 = label_w + c * (thumb + pad)
            try:
                thumb_img = Image.open(img_path).convert("RGB").resize((thumb, thumb))
            except OSError:
                thumb_img = Image.new("RGB", (thumb, thumb), "gray")
            grid.paste(thumb_img, (x0, y0 + header_h))
            sign_color = "lime" if score >= 0 else "red"
            draw.text((x0, y0 + header_h + thumb - 12), f"{score:+.2f}", fill=sign_color, font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main():
    candidates: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    with open(PAIRS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            concept_name = row["concept_name"]
            for method_name, col in METHOD_COLUMN.items():
                candidates[concept_name][method_name].append((row["image"], float(row[col])))

    print(f"{len(candidates)} concepts found in {PAIRS_CSV}", flush=True)
    for concept_name in sorted(candidates):
        rows_for_grid = []
        for method_name in METHOD_COLUMN:
            ranked = sorted(candidates[concept_name][method_name], key=lambda kv: -abs(kv[1]))
            method_label = METHOD_DISPLAY_NAMES[method_name]
            rows_for_grid.append((f"{method_label} (top-5)", ranked[:N]))
            rows_for_grid.append((f"{method_label} (bottom-5)", ranked[-N:]))
        build_image_grid(rows_for_grid, OUT_DIR / f"{concept_name}.png")
    print(f"Saved top-5/bottom-5 grids to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
