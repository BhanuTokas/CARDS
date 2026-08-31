"""Maps CelebA's 40 official binary attributes to CelebAMask-HQ's 19
real per-pixel segmentation classes, and separates the 2 attributes used
as this track's TARGET CLASSES (Attractive, Young) from the 38 used as
CONCEPTS -- prompted directly ("How about we let attractive and young be
the target and everything else be concepts?").

This decoupling is deliberate and load-bearing: CelebAMask-HQ's regions
overlap heavily with CelebA's own attribute names (an `eye_g` mask region
and an `Eyeglasses` attribute are nearly the same thing), so testing "does
the eyeglass region matter for predicting Eyeglasses" would be close to
tautological. Since Attractive/Young are never themselves one of the 38
concept attributes, no concept is ever tested against itself as a class --
this structurally removes the circularity CUB never had to deal with
(CUB's own concept VALUES, e.g. "wing_color::brown", were never also one
of CUB's own 200 species CLASSES).

Unlike `cards.data.cub_attributes`, no prefix/value split is needed --
each CelebA attribute IS a single binary concept directly (no separate
"which value of this attribute" dimension the way CUB's color/pattern/
shape attributes needed one). Unlike CUB, no calibrated-vs-heuristic
split is needed either: every groundable attribute below maps to a REAL
CelebAMask-HQ per-pixel mask (no keypoint-patch approximation exists in
this track at all -- CelebAMask-HQ has real masks for all 30,000 images,
not just a small validated subset the way CUB70 was for CUB).

The groundable/excluded split was derived directly from CelebAMask-HQ's
own real 19 mask classes (see CELEBA_MASK_CLASSES), not assumed --
notably, CelebAMask-HQ has NO facial-hair segmentation class at all, so
5_o_Clock_Shadow/Goatee/Mustache/Sideburns/No_Beard are all excluded
despite being an intuitive concept category to include; the real class
list simply doesn't support them, a genuine finding worth stating
plainly rather than forcing an approximate fit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# CelebAMask-HQ's own 19 mask classes (CelebAMask-HQ-mask-anno/*), for
# reference and validation once real files are in hand -- "background" is
# excluded from concept use (not a facial feature).
CELEBA_MASK_CLASSES: list[str] = [
    "background", "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow",
    "l_ear", "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat", "ear_r",
    "neck_l", "neck", "cloth",
]

# The 2 attributes used as this track's target CLASSES -- never present as
# a key in ATTRIBUTE_TO_REGIONS below (excluded from the concept set by
# construction, not filtered out separately).
TARGET_CLASSES: list[str] = ["Attractive", "Young"]

# CelebA attribute name -> CelebAMask-HQ region name(s) describing it.
# Multi-region entries (e.g. eyebrows) get their masks OR'd together.
ATTRIBUTE_TO_REGIONS: dict[str, list[str]] = {
    "Arched_Eyebrows": ["l_brow", "r_brow"],
    "Bushy_Eyebrows": ["l_brow", "r_brow"],
    "Bags_Under_Eyes": ["l_eye", "r_eye"],
    "Narrow_Eyes": ["l_eye", "r_eye"],
    "Big_Nose": ["nose"],
    "Pointy_Nose": ["nose"],
    "Big_Lips": ["u_lip", "l_lip"],
    "Wearing_Lipstick": ["u_lip", "l_lip"],
    "Mouth_Slightly_Open": ["mouth"],
    "Smiling": ["mouth"],
    "Bald": ["hair"],
    "Bangs": ["hair"],
    "Black_Hair": ["hair"],
    "Blond_Hair": ["hair"],
    "Brown_Hair": ["hair"],
    "Gray_Hair": ["hair"],
    "Straight_Hair": ["hair"],
    "Wavy_Hair": ["hair"],
    "Receding_Hairline": ["hair"],
    "Pale_Skin": ["skin"],
    "Rosy_Cheeks": ["skin"],
    "Eyeglasses": ["eye_g"],
    "Wearing_Earrings": ["ear_r"],
    "Wearing_Hat": ["hat"],
    "Wearing_Necklace": ["neck_l"],
    "Wearing_Necktie": ["cloth"],  # approx -- no dedicated necktie class in CelebAMask-HQ
}

# Excluded attributes -> why, kept explicit rather than silently dropped.
EXCLUDED_ATTRIBUTES: dict[str, str] = {
    "5_o_Clock_Shadow": "no facial-hair segmentation class exists in CelebAMask-HQ at all",
    "Goatee": "no facial-hair segmentation class exists in CelebAMask-HQ at all",
    "Mustache": "no facial-hair segmentation class exists in CelebAMask-HQ at all",
    "Sideburns": "no facial-hair segmentation class exists in CelebAMask-HQ at all",
    "No_Beard": "no facial-hair segmentation class exists in CelebAMask-HQ at all",
    "Blurry": "whole-image quality property, not a region",
    "Male": "holistic/gestalt judgment, not one region",
    "Heavy_Makeup": "spans eyes+lips+skin simultaneously, no single region",
    "High_Cheekbones": "bone-structure gestalt, no distinct cheekbone segmentation class",
    "Chubby": "whole-face/body contour, not one region",
    "Double_Chin": "whole lower-face contour, not one region",
    "Oval_Face": "whole-face shape, not one region",
}

# 8-concept pilot (Phase 4-7 of the plan) -- deliberately mixes large/small
# masks, color/shape/dynamic attributes, and common/rare presence, the
# same diversity CUB's own 8-part pilot had (v33-v37).
PILOT_CONCEPTS: list[str] = [
    "Black_Hair", "Bushy_Eyebrows", "Big_Nose", "Smiling",
    "Narrow_Eyes", "Eyeglasses", "Pale_Skin", "Wearing_Hat",
]

# All 26 groundable attributes (the pilot's own 8 plus 18 more) -- the
# scale-up target once the pilot's own infrastructure was proven (v65-v71),
# mirroring how CUB's 8-part pilot (v33-v37) scaled to its full 87-
# attribute bank (v38+). Sorted alphabetically for a stable, deterministic
# script order (ATTRIBUTE_TO_REGIONS is itself insertion-ordered, not
# alphabetical -- sorting here decouples script iteration order from that
# dict's own declaration order).
GROUNDABLE_CONCEPTS: list[str] = sorted(ATTRIBUTE_TO_REGIONS)


def load_attribute_names(path: Path) -> list[str]:
    """The 40 official CelebA attribute names, in their canonical order --
    from list_attr_celeba.txt's own header line (line 2; line 1 is just
    the image count). Confirmed directly against the real, already-local
    file at Datasets/CelebA/celeba/list_attr_celeba.txt -- not assumed."""
    lines = Path(path).read_text().splitlines()
    return lines[1].split()


def load_attribute_labels(path: Path) -> dict[str, np.ndarray]:
    """image filename ("000001.jpg") -> (40,) boolean array, in
    load_attribute_names' own column order. list_attr_celeba.txt encodes
    presence/absence as 1/-1 (not 1/0); converted to bool here so callers
    never need to know that encoding detail."""
    lines = Path(path).read_text().splitlines()
    result: dict[str, np.ndarray] = {}
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split()
        filename, values = parts[0], parts[1:]
        result[filename] = np.array([v == "1" for v in values], dtype=bool)
    return result


def groundable_attributes(attribute_names: list[str]) -> dict[int, tuple[str, list[str]]]:
    """attribute_index -> (attribute_name, [region_names]) for every
    concept attribute (i.e. NOT Attractive/Young, see TARGET_CLASSES)
    with a known real-mask region mapping. Attributes not present in the
    returned dict are either a target class or structurally excluded
    (see EXCLUDED_ATTRIBUTES) -- out of scope for the masking
    faithfulness metric either way."""
    result = {}
    for idx, name in enumerate(attribute_names):
        if name in ATTRIBUTE_TO_REGIONS:
            result[idx] = (name, ATTRIBUTE_TO_REGIONS[name])
    return result
