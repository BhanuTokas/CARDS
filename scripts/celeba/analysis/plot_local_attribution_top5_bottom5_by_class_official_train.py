"""Per-CLASS (task) counterpart of `plot_local_attribution_top5_
bottom5_combined_official_train.py` -- same combined present+absent
|magnitude| retrieval grid, but ranked SEPARATELY per target task
(Attractive, Male) instead of pooling both into one ranking per
concept. Prompted directly ("Can you show the top and bottom 5
examples now for each class?"), the natural follow-up once pooling
Attractive+Male was shown to be confounded (score_local_attribution_
present_vs_complete_official_train.py's own class-pooled-rho check --
Male's hybrid_score has ~2.2x the spread of Attractive's, so a POOLED
magnitude ranking would be structurally biased toward surfacing Male
rows regardless of true concept relevance, the same kind of between-
task artifact already flagged for the correlation number).

One grid PER (concept, task) -- `{concept}_{task}.png` -- rather than
cramming both tasks into a single taller image, so each grid stays a
direct, readable comparison at the SAME 4-row shape as every other
grid in this track (hybrid top-5, hybrid bottom-5, tcav top-5, tcav
bottom-5).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RESULTS_DIR = Path("results")
PRESENT_CSV = RESULTS_DIR / "local_attribution_celeba_official_train_pairs.csv"
ABSENT_CSV = RESULTS_DIR / "local_attribution_celeba_official_train_absent_pairs.csv"
OUT_DIR = RESULTS_DIR / "local_attribution_top5_bottom5_by_class_official_train"
METHODS = ["hybrid", "tcav"]
TASKS = ["Attractive", "Male"]
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
    # candidates[concept][task][method] = [(image, score, is_present), ...]
    candidates: dict[str, dict[str, dict[str, list[tuple[str, float, bool]]]]] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    with open(PRESENT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            for method_name in METHODS:
                candidates[row["concept_name"]][row["target_task"]][method_name].append(
                    (row["image"], float(row[f"{method_name}_score"]), True)
                )
    with open(ABSENT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            for method_name in METHODS:
                candidates[row["concept_name"]][row["target_task"]][method_name].append(
                    (row["image"], float(row[f"{method_name}_score"]), False)
                )

    print(f"{len(candidates)} concepts found, split by class ({TASKS}).", flush=True)
    n_saved = 0
    for concept_name in sorted(candidates):
        for task_name in TASKS:
            rows_for_grid = []
            for method_name in METHODS:
                ranked = sorted(candidates[concept_name][task_name][method_name], key=lambda kv: -abs(kv[1]))
                top5 = ranked[:N]
                bottom5 = ranked[-N:]
                n_top_present = sum(1 for _, _, present in top5 if present)
                n_bottom_present = sum(1 for _, _, present in bottom5 if present)
                rows_for_grid.append((f"{method_name} (top-5, {n_top_present}/5 present)", top5))
                rows_for_grid.append((f"{method_name} (bottom-5, {n_bottom_present}/5 present)", bottom5))
            build_image_grid(rows_for_grid, OUT_DIR / f"{concept_name}_{task_name}.png")
            n_saved += 1
    print(f"Saved {n_saved} per-(concept,class) grids to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
