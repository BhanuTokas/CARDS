"""Raw NetDissect Broden index/label/mask reading -- the original
pixel-segmented release (`NetDissect/dataset/broden1_224/`), distinct
from `post_hoc_cbm`'s repackaged `Datasets/broden_concepts/<concept>/
{positives,negatives}/` crops this project has used elsewhere (that
repackaging has no masks at all, only whole-image crops).

Needed by Phase 3 (masking-based faithfulness metric,
`cards.validation.broden_faithfulness`) to know exactly *where* a concept
is in a given image, and reusable later by Phase 4 (the deferred
improved-concept-bank rebuild) for the same reason -- see
`notes/pcbm_correlation_investigation.md`'s v25+ entries and the parent
plan file for the full context.

Schema and mask encoding confirmed directly against the real files, not
assumed (see the plan's "Key facts" section for the empirical checks:
mask pixel value = R + 256*G, decoding to label.csv's *global* label
number -- not c_<category>.csv's separate per-category code):

- `index.csv`: `image,split,ih,iw,sh,sw,color,object,part,material,scene,texture`.
  `color`/`object`/`material` are per-pixel (one mask PNG filename per
  cell, empty if absent). `part` is per-pixel too, but can have up to 4
  simultaneous mask planes (semicolon-joined filenames) -- OR them
  together. `scene`/`texture` are per-image only (the cell holds decimal
  global label number(s) directly, semicolon-joined if multiple; no mask
  file, no finer localization possible).
- `label.csv`: `number,name,category,frequency,coverage,syns`, one row
  per *global* label number. `category` is `catname(count)`, or
  `catname1(count1);catname2(count2)` when a label spans categories.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

PIXEL_CATEGORIES = ("color", "object", "part", "material")
IMAGE_CATEGORIES = ("scene", "texture")
ALL_CATEGORIES = PIXEL_CATEGORIES + IMAGE_CATEGORIES


@dataclass
class BrodenLabel:
    number: int
    name: str
    categories: dict[str, int]  # category -> that category's own frequency count
    frequency: int
    coverage: float


@dataclass
class BrodenRecord:
    image: Path
    split: str
    ih: int
    iw: int
    sh: int
    sw: int
    mask_paths: dict[str, list[Path]] = field(default_factory=dict)  # category -> [] | [1] | [<=4 for "part"]
    per_image_labels: dict[str, list[int]] = field(default_factory=dict)  # category -> [] | [label numbers], scene/texture only


def _parse_category_field(value: str) -> dict[str, int]:
    """`"object(4743)"` -> {"object": 4743}; `"wall(15553);part(29)"` ->
    {"wall": ...} -- wait, format is `catname(count)`, semicolon-joined
    across categories the label spans."""
    result: dict[str, int] = {}
    if not value:
        return result
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, rest = chunk.partition("(")
        count = int(rest.rstrip(")"))
        result[name] = count
    return result


def load_label_table(root: Path) -> dict[int, BrodenLabel]:
    root = Path(root)
    labels: dict[int, BrodenLabel] = {}
    with open(root / "label.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            number = int(row["number"])
            labels[number] = BrodenLabel(
                number=number,
                name=row["name"],
                categories=_parse_category_field(row["category"]),
                frequency=int(row["frequency"]),
                coverage=float(row["coverage"]) if row["coverage"] else 0.0,
            )
    return labels


def _parse_mask_field(value: str, root: Path) -> list[Path]:
    if not value:
        return []
    return [root / "images" / chunk.strip() for chunk in value.split(";") if chunk.strip()]


def _parse_label_number_field(value: str) -> list[int]:
    if not value:
        return []
    return [int(chunk.strip()) for chunk in value.split(";") if chunk.strip()]


def load_index(root: Path) -> list[BrodenRecord]:
    root = Path(root)
    records: list[BrodenRecord] = []
    with open(root / "index.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mask_paths: dict[str, list[Path]] = {}
            per_image_labels: dict[str, list[int]] = {}
            for cat in PIXEL_CATEGORIES:
                mask_paths[cat] = _parse_mask_field(row[cat], root)
            for cat in IMAGE_CATEGORIES:
                per_image_labels[cat] = _parse_label_number_field(row[cat])
            records.append(
                BrodenRecord(
                    image=root / "images" / row["image"],
                    split=row["split"],
                    ih=int(row["ih"]),
                    iw=int(row["iw"]),
                    sh=int(row["sh"]),
                    sw=int(row["sw"]),
                    mask_paths=mask_paths,
                    per_image_labels=per_image_labels,
                )
            )
    return records


def decode_mask(mask_path: Path) -> np.ndarray:
    """Returns an (sh, sw) int array of global label numbers -- pixel
    value = R + 256*G (B unused), confirmed empirically against
    label.csv/c_object.csv (see module docstring)."""
    arr = np.array(Image.open(mask_path).convert("RGB"), dtype=np.int32)
    return arr[:, :, 0] + 256 * arr[:, :, 1]


def concept_pixel_mask(record: BrodenRecord, category: str, label_number: int) -> np.ndarray | None:
    """Boolean (ih, iw) mask for `label_number` in `category`, nearest-
    neighbor upsampled from the mask's own (sh, sw) resolution to the
    image's (ih, iw) resolution, OR'd across all planes present (relevant
    for "part", which can have up to 4). Returns None if `category` has
    no mask file for this record at all (image categories, or a pixel
    category the record simply has no labels in)."""
    if category not in PIXEL_CATEGORIES:
        return None
    paths = record.mask_paths.get(category, [])
    if not paths:
        return None

    combined = None
    for path in paths:
        decoded = decode_mask(path)
        plane_mask = decoded == label_number
        combined = plane_mask if combined is None else (combined | plane_mask)

    if combined is None:
        return None
    if combined.shape != (record.ih, record.iw):
        combined = np.array(
            Image.fromarray(combined.astype(np.uint8) * 255).resize(
                (record.iw, record.ih), resample=Image.NEAREST
            )
        ) > 0
    return combined


def records_with_concept(records: list[BrodenRecord], category: str, label_number: int) -> list[BrodenRecord]:
    if category in PIXEL_CATEGORIES:
        result = []
        for r in records:
            mask = concept_pixel_mask(r, category, label_number)
            if mask is not None and mask.any():
                result.append(r)
        return result
    return [r for r in records if label_number in r.per_image_labels.get(category, [])]


def build_concept_index(records: list[BrodenRecord], category: str) -> dict[int, list[BrodenRecord]]:
    """label_number -> [records containing it in `category`], decoding
    each record's mask plane(s) exactly once regardless of how many
    concepts are queried -- O(records) instead of O(records * concepts),
    which `records_with_concept` called separately per concept is not.
    Prefer this over repeated `records_with_concept` calls when checking
    more than one concept in the same category (Phase 3's faithfulness
    driver queries several)."""
    if category not in PIXEL_CATEGORIES:
        raise ValueError(f"build_concept_index only supports pixel categories, got {category!r}")
    index: dict[int, list[BrodenRecord]] = {}
    for record in records:
        paths = record.mask_paths.get(category, [])
        if not paths:
            continue
        present_labels: set[int] = set()
        for path in paths:
            present_labels.update(np.unique(decode_mask(path)).tolist())
        present_labels.discard(0)  # 0 = no label
        for label_number in present_labels:
            index.setdefault(label_number, []).append(record)
    return index


def concepts_in_category(labels: dict[int, BrodenLabel], category: str) -> list[BrodenLabel]:
    matching = [lbl for lbl in labels.values() if category in lbl.categories]
    return sorted(matching, key=lambda lbl: lbl.categories[category], reverse=True)
