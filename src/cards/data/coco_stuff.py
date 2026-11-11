"""COCO-Stuff dataset loader.

Loads ALL images from a COCO split (train2017 / val2017) and assigns each
a label drawn from the 20 co-occurrence-bias classes identified by
Singh et al. (2020). Images that contain none of the 20 target classes
receive label -1; images that contain ≥1 get the lowest matching class
index (first class in the supplied `classes` list, 0-indexed).

The 20 classes below are a reconstruction from Singh et al. (2020)
"Don't Judge an Object by Its Context". Verify against
instances_val2017.json `categories[].name` before running -- if a name
doesn't match exactly, that class is silently absent from the labels.

Dataset layout expected at `root`:
    root/
        train2017/          # image files (*.jpg)
        val2017/            # image files (*.jpg)
        annotations/
            instances_train2017.json
            instances_val2017.json
"""

from __future__ import annotations

import json
from pathlib import Path

COCO_STUFF_20_CLASSES: list[str] = [
    "cup",
    "wine glass",
    "handbag",
    "apple",
    "car",
    "bus",
    "potted plant",
    "spoon",
    "microwave",
    "keyboard",
    "skis",
    "clock",
    "sports ball",
    "remote",
    "snowboard",
    "toaster",
    "hair drier",
    "tennis racket",
    "skateboard",
    "baseball glove",
]

# Co-occurrence map from Singh et al. (2020): biased class -> spurious co-occur class.
CO_OCCUR_MAP: dict[str, str] = {
    "cup":            "dining table",
    "wine glass":     "person",
    "handbag":        "person",
    "apple":          "fruit",
    "car":            "road",
    "bus":            "road",
    "potted plant":   "vase",
    "spoon":          "bowl",
    "microwave":      "oven",
    "keyboard":       "mouse",
    "skis":           "person",
    "clock":          "building",
    "sports ball":    "person",
    "remote":         "person",
    "snowboard":      "person",
    "toaster":        "ceiling",
    "hair drier":     "towel",
    "tennis racket":  "person",
    "skateboard":     "person",
    "baseball glove": "person",
}

# Subset of co-occur classes that exist as concept names in the Broden 143 set.
CO_OCCUR_BRODEN_CONCEPTS: frozenset[str] = frozenset(CO_OCCUR_MAP.values()) & frozenset({
    "person", "bowl", "oven", "building", "ceiling",
})


def load_coco_stuff_pairs(
    root: Path,
    classes: list[str],
    split: str = "val",
) -> list[tuple[Path, int]]:
    """Return sorted (image_path, label) pairs for ALL images in the split.

    `classes` is the ordered list of 20 target class names. Label is the
    0-indexed position of the first matching class in `classes` for images
    that contain ≥1 target class, or -1 for images with none.

    Annotation file: root/annotations/instances_{split}2017.json
    Image directory: root/{split}2017/
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    root = Path(root)
    ann_path = root / "annotations" / f"instances_{split}2017.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {ann_path}")

    with open(ann_path) as f:
        data = json.load(f)

    # category name → COCO category id
    cat_name_to_id: dict[str, int] = {cat["name"]: cat["id"] for cat in data["categories"]}

    # COCO category id → index into `classes` (only for target classes present in COCO)
    cat_id_to_class_idx: dict[int, int] = {}
    for idx, cls in enumerate(classes):
        coco_id = cat_name_to_id.get(cls)
        if coco_id is not None:
            cat_id_to_class_idx[coco_id] = idx

    target_cat_ids = set(cat_id_to_class_idx)

    # image_id → minimum (first) matching class index
    image_to_label: dict[int, int] = {}
    for ann in data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in target_cat_ids:
            continue
        img_id = ann["image_id"]
        cls_idx = cat_id_to_class_idx[cat_id]
        if img_id not in image_to_label or cls_idx < image_to_label[img_id]:
            image_to_label[img_id] = cls_idx

    image_dir = root / f"{split}2017"
    result: list[tuple[Path, int]] = []
    for img_info in data["images"]:
        path = image_dir / img_info["file_name"]
        label = image_to_label.get(img_info["id"], -1)
        result.append((path, label))

    return sorted(result)
