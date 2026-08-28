"""Direct test of the fallback/redundancy hypothesis ("could masking an
individual concept not be meaningfully misleading the model because it
can fall back on other concepts, so ground-truth delta_p is under-
reported?"): for real images with >=2 valid, DIFFERENT-part groundable
concepts simultaneously present, compares delta_p from masking region A
alone, region B alone, and BOTH regions at once (their mask union).

If the model relies on genuinely redundant/interchangeable cues (the
fallback story), losing BOTH regions should hurt MUCH more than losing
either alone predicts additively -- a super-additive interaction
(delta_AB > delta_A + delta_B). If the two regions are independent,
non-redundant cues, the combined effect should be close to additive
(delta_AB ~= delta_A + delta_B). Sub-additive (delta_AB < delta_A +
delta_B) would suggest some other interference effect.

Reuses the exact same masking/scoring machinery as the official ground
truth (compute_faithfulness, blur fill by default) for a fair, directly
comparable measurement -- only the region composition (single vs. union)
differs.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
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
N_RANDOM_DRAWS = 5
N_PAIRS_TARGET = 80  # number of (image, region-A, region-B) triples to test

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

    def preprocess(self, image: Image.Image):
        return self._preprocess(image)

    def __call__(self, batch):
        return self.model(batch.to(self.device))


def representative_part(part_names: list[str]) -> str:
    for name in part_names:
        if name.startswith("left_"):
            return name
    return part_names[0]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    model = Resnet18CubAdapter(DEVICE)
    rng_py = random.Random(SEED)

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
    rng_py.shuffle(test_ids)

    attribute_names = load_attribute_names(ATTRIBUTE_NAMES_PATH)
    groundable = groundable_attributes(attribute_names)
    class_attributes = load_class_attributes(CLASS_ATTR_DIR)

    def valid_mask_for(image, silhouette, part_name, kp):
        if kp is None or not kp[2]:
            return None
        x, y, _ = kp
        target_area = PART_AREA_RATIO[part_name] * silhouette.sum()
        mask = keypoint_patch_mask(x, y, target_area, (image.height, image.width))
        return mask if mask.any() else None

    results = []
    n_draws = 0
    for image_id in test_ids:
        if len(results) >= N_PAIRS_TARGET:
            break
        cid = class_labels[image_id]
        species_attrs = class_attributes.get(cid)
        if species_attrs is None:
            continue

        # candidate (attr_idx, part_name) pairs TRUE for this species and groundable
        candidates = []
        for attr_idx, (prefix, part_names) in groundable.items():
            if not species_attrs[attr_idx]:
                continue
            candidates.append((attr_idx, representative_part(part_names)))
        if len(candidates) < 2:
            continue
        rng_py.shuffle(candidates)

        image_path = image_paths[image_id]
        try:
            image = Image.open(image_path).convert("RGB")
            silhouette = load_cub_segmentation(CUB_ROOT, image_id, image_paths)
        except FileNotFoundError:
            continue
        if silhouette.shape != (image.height, image.width) or silhouette.sum() == 0:
            continue

        # find two candidates from DIFFERENT parts, both with valid masks
        chosen = []
        seen_parts = set()
        for attr_idx, part_name in candidates:
            if part_name in seen_parts:
                continue
            kp = keypoints.get(image_id, {}).get(PART_NAME_TO_ID[part_name])
            mask = valid_mask_for(image, silhouette, part_name, kp)
            if mask is None:
                continue
            chosen.append((attr_idx, part_name, mask))
            seen_parts.add(part_name)
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            continue

        (attr_a, part_a, mask_a), (attr_b, part_b, mask_b) = chosen
        mask_union = mask_a | mask_b
        target_class = cid - 1

        def run(mask, tag):
            nonlocal n_draws
            rng_np = np.random.default_rng(SEED + 900_000 + n_draws)
            n_draws += 1
            return compute_faithfulness(
                image=image, image_path=str(image_path), concept_number=-1,
                category="multiconcept", mask=mask, model=model, rng=rng_np,
                n_random_draws=N_RANDOM_DRAWS, fill_strategy="blur", device=DEVICE,
                target_class=target_class,
            )

        result_a = run(mask_a, "A")
        result_b = run(mask_b, "B")
        result_ab = run(mask_union, "AB")

        results.append({
            "image_id": image_id, "class": cid,
            "attr_a": attribute_names[attr_a], "part_a": part_a,
            "attr_b": attribute_names[attr_b], "part_b": part_b,
            "delta_a": result_a.delta_p, "delta_b": result_b.delta_p, "delta_ab": result_ab.delta_p,
        })
        if len(results) % 10 == 0:
            print(f"[{len(results)}/{N_PAIRS_TARGET}] {attribute_names[attr_a]} + {attribute_names[attr_b]}: "
                  f"A={result_a.delta_p:.4f} B={result_b.delta_p:.4f} AB={result_ab.delta_p:.4f} "
                  f"(additive pred={result_a.delta_p + result_b.delta_p:.4f})", flush=True)

    import csv

    with open(RESULTS_DIR / "multiconcept_masking_additivity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    deltas_a = np.array([r["delta_a"] for r in results])
    deltas_b = np.array([r["delta_b"] for r in results])
    deltas_ab = np.array([r["delta_ab"] for r in results])
    additive = deltas_a + deltas_b
    interaction = deltas_ab - additive

    print(f"\n{len(results)} (image, region-A, region-B) triples tested.")
    print(f"mean delta_p(A alone):      {deltas_a.mean():.4f}")
    print(f"mean delta_p(B alone):      {deltas_b.mean():.4f}")
    print(f"mean additive prediction:   {additive.mean():.4f}  (A + B)")
    print(f"mean delta_p(A+B combined): {deltas_ab.mean():.4f}")
    print(f"mean interaction (AB - (A+B)): {interaction.mean():+.4f}  "
          f"(positive = super-additive/redundancy signature, ~0 = independent, negative = interference)")

    from scipy.stats import wilcoxon
    stat, p = wilcoxon(deltas_ab, additive)
    print(f"Wilcoxon signed-rank test (AB vs. additive prediction): p={p:.4g}")
    print(f"fraction where combined > additive (super-additive): {(deltas_ab > additive).mean():.1%}")
    print(f"fraction where combined < additive (sub-additive):   {(deltas_ab < additive).mean():.1%}")


if __name__ == "__main__":
    main()
