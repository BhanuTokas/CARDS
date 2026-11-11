"""One-off patch: add class_label column to existing coco_stuff_faithfulness.csv.

Reads the existing faithfulness CSV, looks up each (image_path, class_name)
in the COCO annotation JSON to determine whether the class was present (1) or
absent (0) in that image, and writes the patched CSV back in-place.

Run once after compute_coco_stuff_faithfulness.py has already produced the CSV.
Does NOT re-run GroundingDINO/SAM.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

COCO_ROOT   = Path("/scratch/rnair21/LLMConceptEditing_old/coco")
RESULTS_DIR = Path("results/coco_stuff")
FAITH_CSV   = RESULTS_DIR / "coco_stuff_faithfulness.csv"

from cards.data.coco_stuff import COCO_STUFF_20_CLASSES


def _load_coco_multilabel(coco_root: Path, classes: list[str], split: str) -> dict[str, set[int]]:
    """Return image_path (str) → set of present class indices."""
    ann_path = coco_root / "annotations" / f"instances_{split}2017.json"
    with open(ann_path) as f:
        data = json.load(f)

    cat_name_to_id = {cat["name"]: cat["id"] for cat in data["categories"]}
    cat_id_to_idx: dict[int, int] = {}
    for idx, cls in enumerate(classes):
        coco_id = cat_name_to_id.get(cls)
        if coco_id is not None:
            cat_id_to_idx[coco_id] = idx
    target_cat_ids = set(cat_id_to_idx)

    image_classes: dict[int, set] = defaultdict(set)
    for ann in data["annotations"]:
        if ann["category_id"] in target_cat_ids:
            image_classes[ann["image_id"]].add(cat_id_to_idx[ann["category_id"]])

    id_to_fname = {img["id"]: img["file_name"] for img in data["images"]}
    image_dir = coco_root / f"{split}2017"
    return {
        str(image_dir / id_to_fname[img_id]): cls_set
        for img_id, cls_set in image_classes.items()
    }


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

    if not FAITH_CSV.exists():
        raise FileNotFoundError(f"Faithfulness CSV not found: {FAITH_CSV}")

    multilabel_map = _load_coco_multilabel(COCO_ROOT, COCO_STUFF_20_CLASSES, "train")
    print(f"Multi-label map: {len(multilabel_map)} images", flush=True)

    cls_to_idx = {cls: idx for idx, cls in enumerate(COCO_STUFF_20_CLASSES)}

    rows: list[dict] = []
    with open(FAITH_CSV, newline="") as f:
        reader = csv.DictReader(f)
        old_fields = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    if "class_label" in old_fields:
        print("class_label already present — nothing to do.", flush=True)
        return

    n_present = n_absent = n_unknown = 0
    for row in rows:
        img_path  = row["image_path"]
        cls_name  = row["class_name"]
        cls_idx   = cls_to_idx.get(cls_name)
        cls_set   = multilabel_map.get(img_path)
        if cls_set is None or cls_idx is None:
            row["class_label"] = 0
            n_unknown += 1
        elif cls_idx in cls_set:
            row["class_label"] = 1
            n_present += 1
        else:
            row["class_label"] = 0
            n_absent += 1

    print(f"Records: {len(rows)}  present={n_present}  absent={n_absent}  unknown={n_unknown}", flush=True)

    new_fields = ["concept_name", "image_path", "class_name", "class_label",
                  "p_orig", "p_masked", "delta_p",
                  "logit_orig", "logit_masked", "delta_logit"]
    # keep any extra columns that were in the original
    for f in old_fields:
        if f not in new_fields:
            new_fields.append(f)

    with open(FAITH_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Patched → {FAITH_CSV}", flush=True)


if __name__ == "__main__":
    main()
