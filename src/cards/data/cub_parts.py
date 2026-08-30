"""CUB-200-2011's own part-location keypoints (`parts/part_locs.txt`) and
the CUB70-PartSegmentationDataset's real per-part pixel masks (Behzadi-
Khormouji & Oramas, WACV 2023 -- github.com/hamedbehzadi/
CUB70-PartSegmentationDataset), for validating a cheap keypoint+patch
faithfulness approximation against real segmentation on the 67 classes
where both exist (see notes/pcbm_correlation_investigation.md).

CUB's 15 keypoints (`parts/parts.txt`) and CUB70's 11 parts don't share a
vocabulary 1:1 -- only 8 have an unambiguous direct match (CUB70's
head/neck/body are each plausibly one of several CUB keypoints, e.g.
"head" could mean crown or forehead, so those are excluded rather than
guessed at):
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# CUB70 part name -> CUB's own parts.txt part id, direct 1:1 matches only.
CUB70_TO_CUB_PART_ID = {
    "beak": 2,
    "left_eye": 7,
    "left_leg": 8,
    "left_wing": 9,
    "right_eye": 11,
    "right_leg": 12,
    "right_wing": 13,
    "tail": 14,
}


@dataclass
class CubImage:
    image_id: str
    path: Path
    class_dir: str  # CUB70's class-index directory name, e.g. "1"


def load_images_txt(cub_root: Path) -> dict[str, Path]:
    root = Path(cub_root)
    result = {}
    for line in (root / "images.txt").read_text().splitlines():
        image_id, rel_path = line.split(maxsplit=1)
        result[image_id] = root / "images" / rel_path
    return result


def load_keypoints(cub_root: Path) -> dict[str, dict[int, tuple[float, float, bool]]]:
    """image_id -> {part_id: (x, y, visible)}, from part_locs.txt
    (`image_id part_id x y visible`, visible is 0/1)."""
    root = Path(cub_root)
    result: dict[str, dict[int, tuple[float, float, bool]]] = {}
    for line in (root / "parts" / "part_locs.txt").read_text().splitlines():
        image_id, part_id, x, y, visible = line.split()
        result.setdefault(image_id, {})[int(part_id)] = (float(x), float(y), visible == "1")
    return result


def load_cub70_mask(cub70_root: Path, class_dir: str, image_stem: str, part_name: str) -> np.ndarray | None:
    """Boolean mask for `part_name` on this image, or None if that part
    wasn't annotated for this image (parts CUB70's annotators judged
    occluded/absent are simply omitted, not given an empty mask file)."""
    path = Path(cub70_root) / "AnnotationMasksPerclass" / class_dir / f"{image_stem}_{part_name}.png"
    if not path.exists():
        return None
    arr = np.array(Image.open(path).convert("L"))
    return arr > 127


def load_cub_segmentation(cub_root: Path, image_id: str, image_paths: dict[str, Path]) -> np.ndarray:
    """Whole-bird foreground silhouette for `image_id` (CUB's own
    `segmentations/`, present for all 200 classes -- confirmed to mirror
    images.txt's own relative path with .jpg swapped for .png). Used to
    scale the keypoint-patch approximation to each image's own visible
    bird size on the ~133 CUB classes with no CUB70 real part mask."""
    root = Path(cub_root)
    rel_path = image_paths[image_id].relative_to(root / "images")
    seg_path = root / "segmentations" / rel_path.with_suffix(".png")
    arr = np.array(Image.open(seg_path).convert("L"))
    return arr > 127


def keypoint_patch_mask(x: float, y: float, target_area: float, image_shape: tuple[int, int]) -> np.ndarray:
    """A filled disk of `target_area` pixels centered at (x, y), clipped
    to image bounds -- the cheap approximation being validated against
    CUB70's real masks."""
    h, w = image_shape
    radius = np.sqrt(target_area / np.pi)
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    return mask
