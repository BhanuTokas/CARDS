"""Loads the DBAC/LIC bias-in-captioning dataset (`gender_obj_cap_mw_entries.pkl`,
n=10,780, matches the DBAC paper's own stated dataset size exactly -- confirmed
directly against the real file, kept locally under the user's own internal
research-repo layout for that paper's codebase) and maps it onto the same
TARGET_CLASSES / GROUNDABLE_CONCEPTS structure
`cards.data.celeba_attributes` uses.

Structural mapping, decided directly with the user: the captioning ranking
experiment attributes a captioning model's gender/race bias to task objects
in the image, mirroring the DBAC paper's own A<->T (attribute<->task)
framing. TARGET_CLASSES (Gender, Race) play the role CelebA's
Attractive/Young played; the 79 non-"person" COCO object categories play
the role CelebA's 26 groundable attributes played. Unlike CelebA, no
region-overlap circularity risk exists here (a "knife" concept and a
"Gender" target class share no vocabulary), so no exclusion list is needed
on that basis -- "person" is excluded for a different, purely statistical
reason: it is present in all 10,780 images (confirmed directly), so it has
no negative set at all and cannot support a present-vs-absent contrast.

No masking or per-pixel region is involved anywhere in this module -- concept
presence/absence is the object's own membership in `rmdup_object_list`,
already a clean human annotation, not something requiring a segmentation
mask (unlike CelebA/CUB). This is deliberate: the earlier masking-hybrid
design was dropped for captioning specifically because its negative-set
construction already relies on masking internally, which would confound a
masking-based causal test (see conversation) -- the ranking design adopted
instead needs no masks anywhere.

Gender is a clean binary in this dataset (only "Male"/"Female" observed,
confirmed directly -- no "Both" value for gender the way there is for
skin). Race ("bb_skin") has 3 valid values plus real missingness:
Light=7752, Dark=940, Unsure=560, Both=238, missing/blank=1290 -- only
Light/Dark are usable as the two classes of a binary Race target;
Unsure/Both/blank are excluded per image (not imputed), same "state what's
excluded and why" discipline as EXCLUDED_ATTRIBUTES in
`celeba_attributes.py`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

# The 2 attributes used as this track's target CLASSES -- ground-truth
# image labels here, NOT a trained classifier's predicted class the way
# CelebA's Attractive/Young were (that distinction matters for how the
# eventual black-box scoring function gets defined; loading/concept-set
# construction below is unaffected by it).
TARGET_CLASSES: list[str] = ["Gender", "Race"]

# Valid category values per target class -- only these two values of each
# are used; anything else (e.g. Race="Unsure"/"Both"/"") is excluded per
# image at load time, not imputed.
TARGET_CLASS_VALUES: dict[str, tuple[str, str]] = {
    "Gender": ("Male", "Female"),
    "Race": ("Light", "Dark"),
}

# Canonical gender-word lists, reused verbatim from the DBAC paper's own
# codebase (`gender_ratio.py`'s MASCULINE/FEMININE) rather than redefined
# here -- for the log-prob-based black-box scoring approach (score a
# caption by the model's own probability on these tokens; 0 when a caption
# contains none of them). No equivalent race word list exists anywhere in
# that codebase -- race is essentially never stated explicitly in captions,
# so this word-probability approach may not transfer to the Race target
# class the way it does to Gender.
MASCULINE_WORDS: list[str] = [
    "man", "men", "male", "father", "gentleman", "boy", "uncle",
    "husband", "actor", "prince", "waiter", "he", "his", "him",
]
FEMININE_WORDS: list[str] = [
    "woman", "women", "female", "mother", "lady", "girl", "aunt",
    "wife", "actress", "princess", "waitress", "she", "her", "hers",
]

_PKL_PATH = Path(r"C:\Users\btokas\Projects\DIC\bias_data\Human_Ann\gender_obj_cap_mw_entries.pkl")


def load_bias_captioning_records(path: Path = _PKL_PATH) -> list[dict]:
    """Raw list of 10,780 per-image dicts, exactly as pickled by the DBAC
    paper's own codebase (Python-2-era pickle; needs latin1 decoding to load
    under Python 3, confirmed directly against the real file). Each record has
    (among other fields not used here) `img_name`, `bb_gender`, `bb_skin`,
    `rmdup_object_list`, `split`, `caption_list`."""
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def groundable_concepts(records: list[dict]) -> list[str]:
    """All COCO object categories appearing in `rmdup_object_list` across
    the dataset, EXCLUDING "person" (present in all 10,780 images --
    confirmed directly -- so it has no negative set and cannot support a
    present-vs-absent contrast). Sorted alphabetically for a stable,
    deterministic script order, matching `celeba_attributes.GROUNDABLE_
    CONCEPTS`'s own convention."""
    objects: set[str] = set()
    for record in records:
        objects.update(record["rmdup_object_list"])
    objects.discard("person")
    return sorted(objects)


def build_concept_sets(records: list[dict], concept_name: str) -> tuple[list[str], list[str]]:
    """(positive_image_names, negative_image_names) for one task-object
    concept -- positive = concept_name is in the image's own
    `rmdup_object_list`, negative = it is not. No masking, no region
    lookup: object presence is already a direct human annotation."""
    positive, negative = [], []
    for record in records:
        (positive if concept_name in record["rmdup_object_list"] else negative).append(record["img_name"])
    return positive, negative


def build_target_class_sets(records: list[dict], target_class: str) -> tuple[list[str], list[str]]:
    """(class_a_image_names, class_b_image_names) for one target class
    (Gender or Race), using only the 2 valid values in TARGET_CLASS_VALUES
    -- images with any other value (missing/Unsure/Both) are excluded from
    both lists, not imputed into either."""
    field = "bb_gender" if target_class == "Gender" else "bb_skin"
    value_a, value_b = TARGET_CLASS_VALUES[target_class]
    class_a = [r["img_name"] for r in records if r[field] == value_a]
    class_b = [r["img_name"] for r in records if r[field] == value_b]
    return class_a, class_b
