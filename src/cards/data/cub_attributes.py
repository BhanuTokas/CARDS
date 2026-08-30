"""Extends the 8-part CUB70-validated concept set to CUB's own 112
official Concept-Bottleneck-Models attributes (Koh et al. 2020, the same
set used by the literature-comparable PCBM reproduction in notes v34),
by mapping each attribute's name prefix (e.g. "has_eye_color") to the CUB
keypoint(s) it describes (`parts/parts.txt`'s own 15, not just CUB70's 8),
so a keypoint-patch pseudo-mask can be placed for attributes CUB70 has no
crop coverage for at all.

Per the user's own framing: since v33 already found the keypoint-patch
approximation tracks real masking closely (r=0.936) rather than needing
one true-to-the-pixel mask, a *reasonable* patch-area estimate is enough
-- exact calibration matters less than getting the region and rough scale
right. 8 of the 15 CUB keypoints already have a real, CUB70-mask-derived
area ratio (`cards.data.cub_parts`/v36); the other 7 (back, belly,
breast, crown, forehead, nape, throat) have no real-mask ground truth
anywhere (CUB70 doesn't cover them), so their ratios below are an
explicitly-flagged, UNVALIDATED heuristic -- grouped by rough anatomical
scale against the calibrated parts, not measured.

Not every one of the 112 attributes describes a single localizable
region: `has_upperparts_color`/`has_underparts_color` (span multiple
keypoints), `has_head_pattern` (ambiguous among crown/forehead/eye), and
`has_size`/`has_shape`/`has_primary_color` (whole-bird properties) have no
single-keypoint home and are deliberately excluded, not guessed at.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

# CUB's own 15 keypoints (parts/parts.txt), part_id -> name.
CUB_PART_NAMES: dict[int, str] = {
    1: "back", 2: "beak", 3: "belly", 4: "breast", 5: "crown", 6: "forehead",
    7: "left_eye", 8: "left_leg", 9: "left_wing", 10: "nape", 11: "right_eye",
    12: "right_leg", 13: "right_wing", 14: "tail", 15: "throat",
}

# Calibrated against real CUB70 masks (notes v36's ratio_part table) --
# median(real_mask_area / silhouette_area) over up to 150 instances each.
CALIBRATED_PART_AREA_RATIO: dict[str, float] = {
    "beak": 0.0267,
    "left_eye": 0.0085,
    "right_eye": 0.0080,
    "left_leg": 0.0356,
    "right_leg": 0.0360,
    "left_wing": 0.2870,
    "right_wing": 0.2685,
    "tail": 0.1178,
}

# UNVALIDATED heuristic defaults for the 7 keypoints CUB70 has zero
# coverage for -- grouped by rough anatomical scale against the
# calibrated parts above (small head-region patches vs. medium
# torso-region patches), not measured against any real mask. Flagged
# explicitly wherever used; treat results built from these with more
# caution than the 8 calibrated parts.
HEURISTIC_PART_AREA_RATIO: dict[str, float] = {
    "crown": 0.015,     # head-region, between beak (0.027) and eye (~0.008)
    "forehead": 0.015,
    "nape": 0.015,
    "throat": 0.015,
    "back": 0.10,       # torso-region, roughly tail-scale (0.118), smaller than wing (0.28)
    "belly": 0.10,
    "breast": 0.10,
}

PART_AREA_RATIO: dict[str, float] = {**CALIBRATED_PART_AREA_RATIO, **HEURISTIC_PART_AREA_RATIO}
CALIBRATED_PARTS = frozenset(CALIBRATED_PART_AREA_RATIO)  # for tagging results by confidence

# Attribute name PREFIX (before "::") -> the CUB keypoint name(s) it
# describes. Attributes whose prefix isn't here (has_upperparts_color,
# has_underparts_color, has_head_pattern, has_size, has_shape,
# has_primary_color) have no single-keypoint home and are excluded.
ATTRIBUTE_PREFIX_TO_PARTS: dict[str, list[str]] = {
    "has_bill_shape": ["beak"],
    "has_bill_length": ["beak"],
    "has_bill_color": ["beak"],
    "has_wing_color": ["left_wing", "right_wing"],
    "has_wing_shape": ["left_wing", "right_wing"],
    "has_wing_pattern": ["left_wing", "right_wing"],
    "has_breast_pattern": ["breast"],
    "has_breast_color": ["breast"],
    "has_back_color": ["back"],
    "has_back_pattern": ["back"],
    "has_tail_shape": ["tail"],
    "has_tail_pattern": ["tail"],
    "has_upper_tail_color": ["tail"],
    "has_under_tail_color": ["tail"],
    "has_throat_color": ["throat"],
    "has_eye_color": ["left_eye", "right_eye"],
    "has_forehead_color": ["forehead"],
    "has_nape_color": ["nape"],
    "has_belly_color": ["belly"],
    "has_belly_pattern": ["belly"],
    "has_leg_color": ["left_leg", "right_leg"],
    "has_crown_color": ["crown"],
}


def load_attribute_names(path: Path) -> list[str]:
    """0-indexed concept-bank position -> attribute name, from
    attributes/new_attributes.txt ("<raw_attribute_id> <name>" per line,
    already the official 112-attribute filtered list)."""
    names = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        _, name = line.split(maxsplit=1)
        names.append(name)
    return names


def groundable_attributes(attribute_names: list[str]) -> dict[int, tuple[str, list[str]]]:
    """attribute_index -> (prefix, [part_names]) for every attribute whose
    prefix has a known single-region mapping. Attributes not present in
    the returned dict have no spatial mapping and are out of scope for
    the masking faithfulness metric."""
    result = {}
    for idx, name in enumerate(attribute_names):
        prefix = name.split("::", 1)[0]
        if prefix in ATTRIBUTE_PREFIX_TO_PARTS:
            result[idx] = (prefix, ATTRIBUTE_PREFIX_TO_PARTS[prefix])
    return result


def load_class_attributes(class_attr_dir: Path) -> dict[int, np.ndarray]:
    """1-indexed class_id -> (112,) boolean array, from post_hoc_cbm's
    class-level (not per-image) majority-vote attribute pickles -- takes
    the first record seen per class from train/val/test combined (a
    class's attribute vector is constant across its own images by
    construction, so any single record for that class carries the same
    vector; test.pkl alone already covers all 200 classes, train+val
    included only for robustness)."""
    root = Path(class_attr_dir)
    result: dict[int, np.ndarray] = {}
    for split in ("test", "train", "val"):
        with open(root / f"{split}.pkl", "rb") as f:
            records = pickle.load(f)
        for r in records:
            cid = r["class_label"] + 1  # pickle stores 0-indexed class_label; CUB's own ids are 1-indexed
            if cid not in result:
                result[cid] = np.array(r["attribute_label"], dtype=bool)
    return result
