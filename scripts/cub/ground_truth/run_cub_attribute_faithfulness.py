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

Sampling is CLASS-STRATIFIED (v49), not flat-random across each
attribute's whole positive-species pool -- prompted directly ("Shouldn't
we stratify by class? and take more samples?") after finding that flat
sampling wasted 96% of its own draws: with only 15 images/attribute drawn
uniformly across dozens of positive species, 74% of the resulting
(attribute, class) groups got exactly 1 sample and only 5% ever reached
the >=3-sample threshold score_method_agreement/score_sign_agreement
need, leaving just 46 usable pairs system-wide. Now, for each attribute,
draw N_PER_CLASS (default 5) samples from EACH of up to
N_TARGET_SPECIES_PER_ATTRIBUTE of that attribute's own positive species
(species shuffled, tried in order, skipped if fewer than N_PER_CLASS
valid instances are available -- moving to the next candidate species
rather than padding with an under-sampled group) -- every species that
ends up contributing to the output is GUARANTEED to clear the downstream
3-sample threshold by construction, not by chance.

N_TARGET_SPECIES_PER_ATTRIBUTE=6 (v53) hit 6/6 for every single one of
the 87 attributes, including the rarest (only 10 candidate positive
species total) -- a 100% qualification rate, meaning the cap, not species
availability or validity-check dropout, was the binding constraint.
Raised to 15 (v56, prompted directly: "Can we further increase the
number of pairs?") after confirming real headroom exists (min positive-
species count across all 87 groundable attributes is 10, mean 40.6,
median 29 -- checked directly via load_class_attributes before raising
the target, not assumed). Raised again to effectively uncapped (999,
prompted directly: "Can we get the result on the complete list of
pairs?" -- following up on the same feasibility check, which found the
true uncapped total is 3537 pairs, computed as sum(n_positive_species)
across all 87 attributes before this run) -- every attribute now draws
from its ENTIRE positive-species pool, not a fixed-size subset.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

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
N_TARGET_SPECIES_PER_ATTRIBUTE = 999  # effectively uncapped -- max positive-species count across all 87 attributes is 192 (has_eye_color::black), so this exhausts every attribute's own full positive-species pool rather than truncating any of them
N_PER_CLASS = 5
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
        rng_py.shuffle(positive_classes)

        n_draws = 0
        qualifying_species = 0
        species_tried = 0
        for cid in positive_classes:
            if qualifying_species >= N_TARGET_SPECIES_PER_ATTRIBUTE:
                break
            species_tried += 1
            species_candidate_ids = list(ids_by_class.get(cid, []))
            rng_py.shuffle(species_candidate_ids)

            species_results = []
            for image_id in species_candidate_ids:
                if len(species_results) >= N_PER_CLASS:
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

                rng_np = np.random.default_rng(SEED + attr_idx * 1000 + n_draws)
                n_draws += 1
                result = compute_faithfulness(
                    image=image, image_path=str(image_path), concept_number=attr_idx,
                    category="calibrated" if calibrated else "heuristic",
                    mask=patch_mask, model=model, rng=rng_np, n_random_draws=N_RANDOM_DRAWS,
                    fill_strategy="blur", device=DEVICE, target_class=class_labels[image_id] - 1,
                )
                species_results.append(result)

            if len(species_results) >= 3:  # clears score_method_agreement's own min_samples_per_pair
                results.extend((r, attr_name, prefix, part_name) for r in species_results)
                qualifying_species += 1
            # else: this species couldn't supply enough valid instances -- skip it, try the next positive species

        tag = "calibrated" if calibrated else "HEURISTIC"
        print(f"[{tag:>10s}] {attr_name:<40s} ({part_name}): {qualifying_species}/{N_TARGET_SPECIES_PER_ATTRIBUTE} "
              f"qualifying species ({species_tried} tried, {len(positive_classes)} positive species total)", flush=True)

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
