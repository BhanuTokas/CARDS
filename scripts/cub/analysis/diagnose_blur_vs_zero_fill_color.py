"""Diagnostic: does the color-vs-pattern delta_p gap (v61) shrink under
fill strategies that actually erase color, unlike blur which mostly
preserves it (confirmed directly in v61)? Prompted directly ("Can we
mask it with a black mask instead?" then "Can we also look at hue
shifting?").

Recomputes delta_p with fill_strategy in {"zero_fill", "hue_shift"} for
a sample of real color-attribute and textured-pattern-attribute
instances (same sampling criteria as v61's mechanism check: striped/
spotted/multi-colored only for pattern, "solid" excluded since it isn't
a real texture case), reusing the exact same masks/images/target classes
already in results/cub_attribute_faithfulness.csv so the comparison is
apples-to-apples against each record's own existing blur-based delta_p
-- not a fresh, differently-sampled run.

zero_fill's own known confound (an out-of-distribution "hole" that can
be salient for reasons unrelated to concept content) and hue_shift's own
narrower scope (only erases color, leaves the region's real spatial
structure/luminance intact -- see mask_region's own docstring for why
that's the point) mean this is a diagnostic cross-check across three
strategies, not a proposal to switch the ground truth's default fill
strategy wholesale.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cards.data.cub_attributes import PART_AREA_RATIO
from cards.data.cub_parts import (
    keypoint_patch_mask,
    load_cub_segmentation,
    load_images_txt,
    load_keypoints,
)
from cards.models.posthoc_cbm import cub_preprocess
from cards.validation.broden_faithfulness import compute_faithfulness

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
RESULTS_DIR = Path("results")
SEED = 42
DEVICE = "cpu"
N_RANDOM_DRAWS = 5
N_SAMPLE = 60

PART_NAME_TO_ID = {
    "back": 1, "beak": 2, "belly": 3, "breast": 4, "crown": 5, "forehead": 6,
    "left_eye": 7, "left_leg": 8, "left_wing": 9, "nape": 10, "right_eye": 11,
    "right_leg": 12, "right_wing": 13, "tail": 14, "throat": 15,
}
TEXTURED_VALUES = {"striped", "spotted", "multi-colored", "speckled"}


class Resnet18CubAdapter:
    def __init__(self, device: str):
        from pytorchcv.model_provider import get_model as ptcv_get_model

        self.model = ptcv_get_model("resnet18_cub", pretrained=True).to(device).eval()
        self._preprocess = cub_preprocess()
        self.device = device

    def preprocess(self, image: Image.Image):
        return self._preprocess(image)

    def __call__(self, batch):
        return self.model(batch.to(self.device))


def attr_type(prefix: str) -> str:
    if "color" in prefix:
        return "color"
    if "pattern" in prefix:
        return "pattern"
    if "shape" in prefix:
        return "shape"
    return "other"


def main():
    rng_py = random.Random(SEED)
    model = Resnet18CubAdapter(DEVICE)
    image_paths = load_images_txt(CUB_ROOT)
    keypoints = load_keypoints(CUB_ROOT)
    class_labels = {}
    for line in (CUB_ROOT / "image_class_labels.txt").read_text().splitlines():
        image_id, class_id = line.split()
        class_labels[image_id] = int(class_id)

    # Build a path-basename -> image_id lookup once (avoid an O(N) scan per row).
    basename_to_id = {Path(p).name: iid for iid, p in image_paths.items()}

    rows_by_type = {"color": [], "pattern_textured": []}
    with open(RESULTS_DIR / "cub_attribute_faithfulness.csv", newline="") as f:
        for row in csv.DictReader(f):
            t = attr_type(row["attribute_prefix"])
            if t == "color":
                rows_by_type["color"].append(row)
            elif t == "pattern" and row["attribute_name"].split("::", 1)[1] in TEXTURED_VALUES:
                rows_by_type["pattern_textured"].append(row)

    seg_cache: dict[str, np.ndarray] = {}
    STRATEGIES = ["zero_fill", "hue_shift"]

    def recompute(rows, label):
        sample = rng_py.sample(rows, min(N_SAMPLE, len(rows)))
        deltas: dict[str, list[float]] = {"blur": [], **{s: [] for s in STRATEGIES}}
        n_draws = 0
        for row in sample:
            image_id = basename_to_id.get(Path(row["image"]).name)
            if image_id is None:
                continue
            part_name = row["part_name"]
            part_id = PART_NAME_TO_ID[part_name]
            kp = keypoints.get(image_id, {}).get(part_id)
            if kp is None or not kp[2]:
                continue
            x, y, _ = kp
            image_path = image_paths[image_id]
            try:
                if image_id not in seg_cache:
                    seg_cache[image_id] = load_cub_segmentation(CUB_ROOT, image_id, image_paths)
                silhouette = seg_cache[image_id]
            except FileNotFoundError:
                continue
            image = Image.open(image_path).convert("RGB")
            if silhouette.shape != (image.height, image.width) or silhouette.sum() == 0:
                continue
            target_area = PART_AREA_RATIO[part_name] * silhouette.sum()
            mask = keypoint_patch_mask(x, y, target_area, (image.height, image.width))
            if not mask.any():
                continue

            attr_idx = int(row["concept_number"])
            target_class = class_labels[image_id] - 1

            deltas["blur"].append(float(row["delta_p"]))  # the SAME instance's already-computed blur delta_p
            for strategy in STRATEGIES:
                rng_np = np.random.default_rng(SEED + attr_idx * 1000 + n_draws)
                n_draws += 1
                result = compute_faithfulness(
                    image=image, image_path=str(image_path), concept_number=attr_idx,
                    category=row["category"], mask=mask, model=model, rng=rng_np,
                    n_random_draws=N_RANDOM_DRAWS, fill_strategy=strategy, device=DEVICE,
                    target_class=target_class,
                )
                deltas[strategy].append(result.delta_p)

        print(f"[{label}] n={len(deltas['blur'])}")
        for strategy in ["blur"] + STRATEGIES:
            vals = deltas[strategy]
            print(f"  mean delta_p {strategy.upper():<10s}: {np.mean(vals):.4f}  (median {np.median(vals):.4f})")
        print(flush=True)
        return deltas

    color_deltas = recompute(rows_by_type["color"], "COLOR attributes")
    pattern_deltas = recompute(rows_by_type["pattern_textured"], "TEXTURED PATTERN attributes")

    print("=== summary: color/pattern delta_p ratio by fill strategy (1.0 = parity) ===")
    for strategy in ["blur"] + STRATEGIES:
        ratio = np.mean(color_deltas[strategy]) / max(np.mean(pattern_deltas[strategy]), 1e-8)
        print(f"{strategy.upper():<10s}: {ratio:.3f}")


if __name__ == "__main__":
    main()
