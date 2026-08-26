"""Extends the CUB faithfulness ground truth from the 8 CUB70-validated
parts to CUB's own 112 official Concept-Bottleneck-Models attributes
(the SAME literature-standard bank used for the reproduction check in
notes v34), via cards.data.cub_attributes' prefix->keypoint mapping.

87/112 attributes have a single-keypoint home (has_upperparts_color,
has_underparts_color, has_head_pattern, has_size, has_shape,
has_primary_color are whole-bird/multi-region properties with no single
location and are excluded, not guessed at). For a paired part
(has_wing_color etc., which maps to both left_wing and right_wing), the
left-side keypoint is used as the representative location, a scope
simplification (not a claim that sides are redundant here -- unlike
CARDS' text-query retrieval, a keypoint-based mask genuinely differs by
side) made to keep this 87-attribute run's compute budget comparable to
v36's 8-part run.

Positive images for an attribute = test-split images from species whose
class-level attribute vector (post_hoc_cbm's own Koh-et-al. majority-vote
labels) has that attribute True -- a species-level, not per-image,
positive/negative split (same caveat as the official 112-concept bank
itself, see notes v34).

Uses each image's own GROUND-TRUTH species label as `compute_faithfulness`'s
`target_class` (not the model's own top-1 prediction) -- see notes v41 for
why, and run_cub_faithfulness.py's own docstring for the same change there.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cards.data.cub_attributes import (  # noqa: E402
    CALIBRATED_PARTS,
    PART_AREA_RATIO,
    groundable_attributes,
    load_attribute_names,
    load_class_attributes,
)
from cards.data.cub_parts import keypoint_patch_mask, load_cub_segmentation, load_images_txt, load_keypoints  # noqa: E402
from cards.models.posthoc_cbm import cub_preprocess  # noqa: E402
from cards.validation.broden_faithfulness import compute_faithfulness  # noqa: E402

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
ATTRIBUTE_NAMES_PATH = CUB_ROOT / "attributes" / "new_attributes.txt"
CLASS_ATTR_DIR = CUB_ROOT / "class_attr_data_10"
RESULTS_DIR = Path("results")
SEED = 42
DEVICE = "cpu"
N_PER_ATTRIBUTE = 15
N_RANDOM_DRAWS = 5

# CUB keypoint name -> part_id, from parts/parts.txt (see cards.data.cub_attributes.CUB_PART_NAMES)
PART_NAME_TO_ID = {
    "back": 1, "beak": 2, "belly": 3, "breast": 4, "crown": 5, "forehead": 6,
    "left_eye": 7, "left_leg": 8, "left_wing": 9, "nape": 10, "right_eye": 11,
    "right_leg": 12, "right_wing": 13, "tail": 14, "throat": 15,
}


class Resnet18CubAdapter:
    def __init__(self, device: str):
        from pytorchcv.model_provider import get_model as ptcv_get_model

        self.model = ptcv_get_model("resnet18_cub", pretrained=True).to(device).eval()
        self._preprocess = cub_preprocess()
        self.device = device

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch.to(self.device))


def representative_part(part_names: list[str]) -> str:
    """Left side when a left/right pair is given, else the single
    unsided part name -- a scope-control simplification, see module
    docstring."""
    for name in part_names:
        if name.startswith("left_"):
            return name
    return part_names[0]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    model = Resnet18CubAdapter(DEVICE)
    rng_py = random.Random(SEED)

    print("Loading CUB metadata + official attribute labels...", flush=True)
    image_paths = load_images_txt(CUB_ROOT)
    keypoints = load_keypoints(CUB_ROOT)
    class_labels = {}
    for line in (CUB_ROOT / "image_class_labels.txt").read_text().splitlines():
        image_id, class_id = line.split()
        class_labels[image_id] = int(class_id)
    test_ids = [
        line.split()[0]
        for line in (CUB_ROOT / "train_test_split.txt").read_text().splitlines()
        if line.split()[1] == "0"
    ]
    ids_by_class: dict[int, list[str]] = {}
    for i in test_ids:
        ids_by_class.setdefault(class_labels[i], []).append(i)

    attribute_names = load_attribute_names(ATTRIBUTE_NAMES_PATH)
    groundable = groundable_attributes(attribute_names)
    class_attributes = load_class_attributes(CLASS_ATTR_DIR)
    print(f"{len(groundable)}/{len(attribute_names)} official attributes have a single-keypoint mapping.", flush=True)

    results = []
    for attr_idx, (prefix, part_names) in groundable.items():
        part_name = representative_part(part_names)
        part_id = PART_NAME_TO_ID[part_name]
        calibrated = part_name in CALIBRATED_PARTS
        attr_name = attribute_names[attr_idx]

        positive_classes = [cid for cid, vec in class_attributes.items() if vec[attr_idx]]
        candidate_ids = [i for cid in positive_classes for i in ids_by_class.get(cid, [])]
        rng_py.shuffle(candidate_ids)

        n_done = 0
        for image_id in candidate_ids:
            if n_done >= N_PER_ATTRIBUTE:
                break
            kp = keypoints.get(image_id, {}).get(part_id)
            if kp is None or not kp[2]:
                continue
            x, y, _visible = kp
            image_path = image_paths[image_id]
            try:
                silhouette = load_cub_segmentation(CUB_ROOT, image_id, image_paths)
            except FileNotFoundError:
                continue
            image = Image.open(image_path).convert("RGB")
            if silhouette.shape != (image.height, image.width) or silhouette.sum() == 0:
                continue
            target_area = PART_AREA_RATIO[part_name] * silhouette.sum()
            patch_mask = keypoint_patch_mask(x, y, target_area, (image.height, image.width))
            if not patch_mask.any():
                continue

            rng_np = np.random.default_rng(SEED + attr_idx * 1000 + n_done)
            result = compute_faithfulness(
                image=image, image_path=str(image_path), concept_number=attr_idx,
                category="calibrated" if calibrated else "heuristic",
                mask=patch_mask, model=model, rng=rng_np, n_random_draws=N_RANDOM_DRAWS,
                fill_strategy="blur", device=DEVICE, target_class=class_labels[image_id] - 1,
            )
            results.append((result, attr_name, prefix, part_name))
            n_done += 1

        tag = "calibrated" if calibrated else "HEURISTIC"
        print(f"[{tag:>10s}] {attr_name:<40s} ({part_name}): {n_done}/{N_PER_ATTRIBUTE} instances "
              f"({len(candidate_ids)} candidates from {len(positive_classes)} positive species)", flush=True)

    with open(RESULTS_DIR / "cub_attribute_faithfulness.csv", "w", newline="") as f:
        base_fields = list(vars(results[0][0]).keys())
        writer = csv.DictWriter(f, fieldnames=base_fields + ["attribute_name", "attribute_prefix", "part_name"])
        writer.writeheader()
        for result, attr_name, prefix, part_name in results:
            row = vars(result)
            row.update(attribute_name=attr_name, attribute_prefix=prefix, part_name=part_name)
            writer.writerow(row)

    print(f"\n{len(results)} total attribute-faithfulness records saved to results/cub_attribute_faithfulness.csv")

    print("\nMean delta_p by prefix (calibrated parts only vs. heuristic parts only):")
    from collections import defaultdict

    by_prefix_cal: dict[str, list[float]] = defaultdict(list)
    by_prefix_heur: dict[str, list[float]] = defaultdict(list)
    for result, attr_name, prefix, part_name in results:
        (by_prefix_cal if part_name in CALIBRATED_PARTS else by_prefix_heur)[prefix].append(result.delta_p)
    print("  calibrated:")
    for prefix, deltas in sorted(by_prefix_cal.items()):
        print(f"    {prefix:<25s} mean_delta_p={np.mean(deltas):.4f} (n={len(deltas)})")
    print("  HEURISTIC (unvalidated area):")
    for prefix, deltas in sorted(by_prefix_heur.items()):
        print(f"    {prefix:<25s} mean_delta_p={np.mean(deltas):.4f} (n={len(deltas)})")


if __name__ == "__main__":
    main()
