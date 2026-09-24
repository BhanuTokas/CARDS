"""Cleaner, publication-oriented replacement for `local_attribution_
comparison_celeba_corrected_gt.build_image_grid`'s original PIL renderer
(tiny default bitmap font, plain white background, small unbordered
score text) -- used by `rebuild_top5_grids_attractive_only.py` and
`rebuild_top10_grids_attractive_only_all_concepts.py` for actual
paper-figure output.

Improvements, all purely cosmetic (same underlying data/ranking):
  - TrueType fonts (Arial/Arial Bold) instead of PIL's tiny default
    bitmap font, at a readable size.
  - A colored border around each thumbnail (green=positive, red=negative)
    instead of small overlaid corner text competing with the photo --
    the numeric score is not shown at all, sign conveyed by border color
    alone.
  - Consistent padding, light panel background per row, bold row labels.

Row label for the masking-hybrid method renamed "ConceptMask (Ours)" ->
"Hide and Seek (Ours)" -- prompted directly ("I changed the name to
Hide and Seek, so the name of the technique needs to be updated").
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_FONT_DIR = Path(r"C:\Windows\Fonts")
_FONT_REGULAR = _FONT_DIR / "arial.ttf"
_FONT_BOLD = _FONT_DIR / "arialbd.ttf"

_BG = (250, 250, 250)
_PANEL_BG = (255, 255, 255)
_POSITIVE = (0, 140, 40)
_NEGATIVE = (200, 30, 30)
_LABEL_COLOR = (30, 30, 30)

_ROW_DISPLAY_NAMES = {
    "ground_truth": "Δp",
    "hybrid": "Hide and Seek (Ours)",
    "tcav": "TCAV",
}


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def build_image_grid_clean(
    rows: list[tuple[str, list[tuple[str, float]]]],
    out_path: Path,
    thumb: int = 150,
) -> None:
    """rows: [(row_key, [(image_path, score), ...]), ...] -- one row per
    method, ranked by |score| by the caller. `row_key` is mapped through
    `_ROW_DISPLAY_NAMES` for a readable row label. No in-image concept
    title -- the concept name is already stated by the caller's own LaTeX
    subfigure caption in every place this is used, so an in-image title
    would just be redundant vertical space.
    """
    n_cols = max(len(images) for _, images in rows)
    n_rows = len(rows)

    pad = 3
    border = 4
    label_w = 190
    row_h_label = 40
    cell_h = thumb + 2 * border
    row_block_h = row_h_label + cell_h + pad

    width = label_w + n_cols * (thumb + 2 * border + pad) + pad
    height = n_rows * row_block_h + pad

    grid = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(grid)

    row_font = _font(_FONT_BOLD, 28)

    for r, (row_key, images) in enumerate(rows):
        row_label = _ROW_DISPLAY_NAMES.get(row_key, row_key)
        y0 = r * row_block_h

        # light alternating panel behind the row for visual separation
        draw.rectangle([0, y0, width, y0 + row_block_h - pad], fill=_PANEL_BG if r % 2 == 0 else _BG)
        draw.text((pad, y0 + row_h_label / 2 - 14), row_label, fill=_LABEL_COLOR, font=row_font)

        for c, (img_path, score) in enumerate(images):
            x0 = label_w + c * (thumb + 2 * border + pad)
            cell_y0 = y0 + row_h_label

            color = _POSITIVE if score >= 0 else _NEGATIVE
            draw.rectangle(
                [x0, cell_y0, x0 + thumb + 2 * border - 1, cell_y0 + thumb + 2 * border - 1],
                outline=color, width=border,
            )
            try:
                thumb_img = Image.open(img_path).convert("RGB").resize((thumb, thumb))
            except OSError:
                thumb_img = Image.new("RGB", (thumb, thumb), "gray")
            grid.paste(thumb_img, (x0 + border, cell_y0 + border))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_with_retry(grid, out_path)


def _save_with_retry(grid: Image.Image, out_path: Path, attempts: int = 5) -> None:
    """Transient antivirus file-lock on this machine occasionally raises
    OSError: [Errno 22] on freshly-written files -- retry a few times
    with a short backoff before giving up for real."""
    import time
    for attempt in range(attempts):
        try:
            grid.save(out_path)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5 * (attempt + 1))


_PRESENT = (0, 150, 60)
_ABSENT = (200, 30, 30)
_DIVIDER = (120, 120, 120)
_NEUTRAL_BORDER = (60, 60, 60)


def build_presence_grid_clean(
    title: str,
    rows: list[tuple[str, list[tuple[str, bool]], list[tuple[str, bool]]]],
    out_path: Path,
    thumb: int = 150,
    n_per_block: int = 3,
) -> None:
    """Cleaner replacement for the raw-PIL-default-font top-N/bottom-N
    presence-vs-absence renderer (tiny bitmap font, on-image cyan/orange
    score text). `rows`: [(method_display_name, top_n[(image, is_present)],
    bottom_n[(image, is_present)]), ...]. Border color alone conveys
    presence/absence (green=present, red=absent), matching
    `build_image_grid_clean`'s own established convention -- no numeric
    score shown, since the qualitative point is whether top-ranked
    (highest |attribution|) images are concept-present and bottom-ranked
    (lowest |attribution|) images are concept-absent, not the exact
    magnitude.
    """
    n_rows = len(rows)
    left_margin, pad, border = 10, 3, 4
    line1_h, line2_h = 32, 26
    row_h_label = line1_h + line2_h
    cell_h = thumb + 2 * border
    block_w = n_per_block * (thumb + 2 * border + pad)
    divider_w = 18
    row_block_h = row_h_label + cell_h + pad

    width = left_margin + 2 * block_w + divider_w + pad
    height = n_rows * row_block_h + pad

    grid = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(grid)

    row_font = _font(_FONT_BOLD, 24)
    sub_font = _font(_FONT_REGULAR, 19)

    bottom_x0_base = left_margin + block_w + divider_w

    for r, (row_label, top_n, bottom_n) in enumerate(rows):
        y0 = r * row_block_h
        draw.rectangle([0, y0, width, y0 + row_block_h - pad], fill=_PANEL_BG if r % 2 == 0 else _BG)
        draw.text((left_margin, y0 + 4), row_label, fill=_LABEL_COLOR, font=row_font)

        sub_y = y0 + line1_h + 2
        cell_y0 = y0 + row_h_label

        draw.text((left_margin, sub_y), "Top-3 Present", fill=(90, 90, 90), font=sub_font)
        for c, (img_path, _is_present) in enumerate(top_n):
            x0 = left_margin + c * (thumb + 2 * border + pad)
            _paste_bordered_thumb(grid, draw, img_path, x0, cell_y0, thumb, border, _NEUTRAL_BORDER)

        div_x = left_margin + block_w + divider_w // 2
        draw.line([(div_x, y0), (div_x, y0 + row_block_h - pad)], fill=_DIVIDER, width=2)

        draw.text((bottom_x0_base, sub_y), "Top-3 Absent", fill=(90, 90, 90), font=sub_font)
        for c, (img_path, _is_present) in enumerate(bottom_n):
            x0 = bottom_x0_base + c * (thumb + 2 * border + pad)
            _paste_bordered_thumb(grid, draw, img_path, x0, cell_y0, thumb, border, _NEUTRAL_BORDER)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def _paste_bordered_thumb(grid: Image.Image, draw: ImageDraw.ImageDraw, img_path: str,
                           x0: int, y0: int, thumb: int, border: int, color: tuple[int, int, int]) -> None:
    draw.rectangle([x0, y0, x0 + thumb + 2 * border - 1, y0 + thumb + 2 * border - 1], outline=color, width=border)
    try:
        thumb_img = Image.open(img_path).convert("RGB").resize((thumb, thumb))
    except OSError:
        thumb_img = Image.new("RGB", (thumb, thumb), "gray")
    grid.paste(thumb_img, (x0 + border, y0 + border))
