"""CelebAMask-HQ image and mask loading -- every detail below was
confirmed directly against the real, extracted archive at
`Datasets/CelebAMask-HQ/` (not assumed from the dataset's own README,
which omits several of these specifics entirely):

- Images: `CelebA-HQ-img/<idx>.jpg`, `idx` in 0..29999, NOT zero-padded.
  Confirmed 1024x1024 JPEG (the README's own text only ever says "high
  resolution", not a specific number).
- Masks: `CelebAMask-HQ-mask-anno/<idx // 2000>/<idx:05d>_<region>.png`
  -- bucketed into subfolders of 2000 images each (confirmed: folder "0"
  holds indices 00000-01999). One PNG per PRESENT class per image (a
  class absent from that image -- e.g. no eyeglasses worn -- simply has
  no file, confirmed directly against image 0's own 11-of-19 present
  files). Confirmed 512x512 RGB, binary 0/255 content (all 3 channels
  identical) -- i.e. HALF the resolution of the source images, a real
  mismatch that needs resizing before masking a real 1024x1024 photo,
  not something the dataset's own docs mention needing to handle.
- Attribute labels: `CelebAMask-HQ-attribute-anno.txt` ships the SAME
  format as standard CelebA's own `list_attr_celeba.txt` (line 1 = image
  count, line 2 = the 40 attribute names, then `<idx>.jpg val1 ... val40`
  rows, -1/1 encoded) but already keyed by CelebAMask-HQ's own 0-29999
  numbering -- confirmed directly, this means `cards.data.celeba_
  attributes.load_attribute_names`/`load_attribute_labels` work on it
  UNCHANGED, with zero join/remapping step needed. The originally-planned
  `CelebA-HQ-to-CelebA-mapping.txt`-based join (mapping HQ indices back
  onto the separate, larger standard-CelebA attribute file) turned out to
  be unnecessary -- confirmed by direct inspection, not assumed to still
  be needed just because the plan anticipated it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

MASK_BUCKET_SIZE = 2000
NATIVE_MASK_SIZE = (512, 512)  # (height, width)

# Same seed/fraction Phase 1's own classifier training
# (scripts/celeba/build/train_attractive_young_classifier.py) used --
# split_celebamask_hq below reproduces that exact split so every
# downstream consumer (retrieval pool, faithfulness ground truth) draws
# only from images the classifier never trained on.
SPLIT_SEED = 42
VAL_FRACTION = 0.15


def split_celebamask_hq(
    image_paths_by_idx: dict[int, Path],
    attr_labels_by_file: dict[str, np.ndarray],
    target_indices: list[int],
    seed: int = SPLIT_SEED,
    val_fraction: float = VAL_FRACTION,
) -> tuple[list[int], list[int]]:
    """Reproduces, bit-for-bit, the stratified 85/15 split
    `train_attractive_young_classifier.py` computed for Phase 1's own
    classifier training -- same sorted-index ordering, same (Attractive,
    Young) joint stratification key, same `train_test_split` call -- so
    it is never re-derived independently and can't silently drift from
    what the classifier actually trained on. Returns (train_hq_indices,
    val_hq_indices), both lists of the original CelebAMask-HQ 0-29999
    indices (not positions).
    """
    from sklearn.model_selection import train_test_split

    indices = sorted(image_paths_by_idx)
    labels = np.array([attr_labels_by_file[f"{i}.jpg"][target_indices] for i in indices])
    strata = labels[:, 0].astype(int) * 2 + labels[:, 1].astype(int)
    train_pos, val_pos = train_test_split(
        np.arange(len(indices)), test_size=val_fraction, random_state=seed, stratify=strata
    )
    return [indices[i] for i in train_pos], [indices[i] for i in val_pos]


def load_celebamask_hq_image_paths(root: Path) -> dict[int, Path]:
    """hq_index (0-29999) -> path to that image's own CelebA-HQ-img/*.jpg."""
    img_dir = Path(root) / "CelebA-HQ-img"
    return {int(p.stem): p for p in img_dir.glob("*.jpg")}


def load_celebamask_hq_mask(
    root: Path, hq_index: int, region_names: list[str], target_hw: tuple[int, int] | None = None
) -> np.ndarray:
    """OR-combined boolean mask over one or more CelebAMask-HQ regions for
    a single image. A region with no file for this image (i.e. absent
    from the photo -- not every person wears glasses/a hat/jewelry) simply
    contributes nothing, not an error. Returns an all-False (512, 512)
    array if none of `region_names` are present in this image at all.

    `target_hw`, when given, resizes the combined mask to match the
    SOURCE IMAGE's own resolution (1024x1024, not the mask's native
    512x512) via NEAREST-neighbor interpolation -- preserves hard mask
    edges rather than blurring the boundary the way bilinear resizing
    would, which matters here since `mask_region`'s own blur/hue_shift/
    etc. strategies are applied at the boundary this mask defines.
    """
    root = Path(root)
    bucket = hq_index // MASK_BUCKET_SIZE
    combined: np.ndarray | None = None
    for region in region_names:
        path = root / "CelebAMask-HQ-mask-anno" / str(bucket) / f"{hq_index:05d}_{region}.png"
        if not path.exists():
            continue
        arr = np.array(Image.open(path).convert("L")) > 0
        combined = arr if combined is None else (combined | arr)

    if combined is None:
        combined = np.zeros(NATIVE_MASK_SIZE, dtype=bool)

    if target_hw is not None and combined.shape != target_hw:
        height, width = target_hw
        resized = Image.fromarray(combined).resize((width, height), Image.NEAREST)
        combined = np.array(resized, dtype=bool)

    return combined
