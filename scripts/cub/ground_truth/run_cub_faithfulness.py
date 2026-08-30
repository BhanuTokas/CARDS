"""CUB-track analogue of run_broden_faithfulness.py: masking-based
faithfulness ground truth for the 8 CUB part concepts, across BOTH:

- the 67 classes with real CUB70-PartSegmentationDataset masks (real
  per-pixel masking, no approximation needed), and
- a sample of the ~133 classes with no CUB70 coverage, using the
  v33-validated keypoint-patch approximation, with patch area scaled to
  each image's own visible bird size via CUB's own whole-bird
  `segmentations/` silhouette (present for all 200 classes) -- v33 only
  ever matched patch area to a real mask's own area (since a real mask
  was always available there for comparison); here there is none, so the
  patch area is instead `ratio_part * this image's own silhouette area`,
  where `ratio_part = median(real_mask_area / silhouette_area)` estimated
  from the 67 CUB70-covered classes.

Both sources restricted to CUB's own TEST split (5,794 images) -- the
same held-out population the reproduction-accuracy checks (notes v34)
used, consistent with not measuring faithfulness on images any of the
CAVs/PCBM surrogate were fit against.

Note: CUB70's own class-directory numbers (1-70) are NOT CUB's own
class ids (1-200) -- coverage is determined empirically per-image (via
stem-matching against images.txt, then that image's own
image_class_labels.txt entry), never by trusting CUB70's directory name
as a species id.

Uses each image's own GROUND-TRUTH species label as `compute_faithfulness`'s
`target_class`, not the model's own top-1 prediction (the Broden/ImageNet
track's default) -- unlike Broden, CUB images genuinely have reliable
species labels, so there's no need to fall back to the model's own guess,
and using ground truth means every record's class field lines up exactly
with CARDS/TCAV/PCBM's own (concept, species) score tables (which were
always keyed by real species identity, never by what the model predicts).
See notes v41.
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

from cards.data.cub_parts import (
    CUB70_TO_CUB_PART_ID,
    keypoint_patch_mask,
    load_cub70_mask,
    load_cub_segmentation,
    load_images_txt,
    load_keypoints,
)
from cards.models.posthoc_cbm import cub_preprocess
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    compute_faithfulness,
)

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
CUB70_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB70_part_segmentation")
RESULTS_DIR = Path("results")
SEED = 42
DEVICE = "cpu"
N_REAL_PER_PART = 30
N_APPROX_PER_PART = 30
N_RANDOM_DRAWS = 5
N_RATIO_CALIBRATION_SAMPLES = 150  # per part, for estimating ratio_part


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


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    model = Resnet18CubAdapter(DEVICE)
    rng_py = random.Random(SEED)

    print("Loading CUB metadata...", flush=True)
    image_paths = load_images_txt(CUB_ROOT)
    keypoints = load_keypoints(CUB_ROOT)
    stem_to_id = {p.stem: image_id for image_id, p in image_paths.items()}
    class_labels = {}
    for line in (CUB_ROOT / "image_class_labels.txt").read_text().splitlines():
        image_id, class_id = line.split()
        class_labels[image_id] = int(class_id)
    test_ids = {
        line.split()[0]
        for line in (CUB_ROOT / "train_test_split.txt").read_text().splitlines()
        if line.split()[1] == "0"
    }

    # (part_name, image_id, cub70_class_dir, cub70_stem) for every real
    # CUB70 mask file, restricted to CUB's own TEST split.
    real_instances: dict[str, list[tuple[str, str, str]]] = {p: [] for p in CUB70_TO_CUB_PART_ID}
    covered_species: set[int] = set()
    for class_dir in sorted((CUB70_ROOT / "AnnotationMasksPerclass").iterdir(), key=lambda p: int(p.name)):
        for mask_file in class_dir.glob("*.png"):
            name = mask_file.stem
            for part_name in CUB70_TO_CUB_PART_ID:
                suffix = f"_{part_name}"
                if name.endswith(suffix):
                    stem = name[: -len(suffix)]
                    image_id = stem_to_id.get(stem)
                    if image_id is not None and image_id in test_ids:
                        real_instances[part_name].append((image_id, class_dir.name, stem))
                        covered_species.add(class_labels[image_id])
                    break

    print(f"{len(covered_species)}/200 CUB species have >=1 real CUB70 mask "
          f"(on the test split, across any of the 8 parts).", flush=True)

    # Calibrate ratio_part = median(real_mask_area / silhouette_area),
    # using CUB70-covered instances (any split -- this is a scale
    # calibration, not a model measurement, so no held-out concern).
    print("\nCalibrating patch-area scale per part against real masks + silhouettes...", flush=True)
    ratio_part: dict[str, float] = {}
    for part_name, part_id in CUB70_TO_CUB_PART_ID.items():
        candidates = real_instances[part_name][:]
        rng_py.shuffle(candidates)
        ratios = []
        for image_id, class_dir, stem in candidates[:N_RATIO_CALIBRATION_SAMPLES]:
            real_mask = load_cub70_mask(CUB70_ROOT, class_dir, stem, part_name)
            if real_mask is None or not real_mask.any():
                continue
            try:
                silhouette = load_cub_segmentation(CUB_ROOT, image_id, image_paths)
            except FileNotFoundError:
                continue
            if silhouette.shape != real_mask.shape or silhouette.sum() == 0:
                continue
            ratios.append(real_mask.sum() / silhouette.sum())
        ratio_part[part_name] = float(np.median(ratios)) if ratios else 0.01
        print(f"  {part_name}: ratio={ratio_part[part_name]:.4f} (n={len(ratios)})", flush=True)

    results: list[FaithfulnessResult] = []

    # --- Real-mask source: sample from CUB70-covered classes directly ---
    print("\n=== Real-mask source ===", flush=True)
    for part_name, part_id in CUB70_TO_CUB_PART_ID.items():
        candidates = real_instances[part_name][:]
        rng_py.shuffle(candidates)
        n_done = 0
        for image_id, class_dir, stem in candidates:
            if n_done >= N_REAL_PER_PART:
                break
            real_mask = load_cub70_mask(CUB70_ROOT, class_dir, stem, part_name)
            if real_mask is None or not real_mask.any():
                continue
            image_path = image_paths[image_id]
            image = Image.open(image_path).convert("RGB")
            if real_mask.shape != (image.height, image.width):
                continue
            rng_np = np.random.default_rng(SEED + n_done)
            result = compute_faithfulness(
                image=image, image_path=str(image_path), concept_number=part_id, category="cub_part_real",
                mask=real_mask, model=model, rng=rng_np, n_random_draws=N_RANDOM_DRAWS,
                fill_strategy="blur", device=DEVICE, target_class=class_labels[image_id] - 1,
            )
            results.append(result)
            n_done += 1
        print(f"{part_name}: {n_done}/{N_REAL_PER_PART} real-mask instances", flush=True)

    # --- Approximation source: species with NO CUB70 coverage at all ---
    print("\n=== Keypoint-patch-approximation source (uncovered species) ===", flush=True)
    uncovered_ids = [i for i in test_ids if class_labels[i] not in covered_species]
    rng_py.shuffle(uncovered_ids)
    for part_name, part_id in CUB70_TO_CUB_PART_ID.items():
        n_done = 0
        for image_id in uncovered_ids:
            if n_done >= N_APPROX_PER_PART:
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
            target_area = ratio_part[part_name] * silhouette.sum()
            patch_mask = keypoint_patch_mask(x, y, target_area, (image.height, image.width))
            if not patch_mask.any():
                continue
            rng_np = np.random.default_rng(SEED + 100000 + n_done)
            result = compute_faithfulness(
                image=image, image_path=str(image_path), concept_number=part_id, category="cub_part_approx",
                mask=patch_mask, model=model, rng=rng_np, n_random_draws=N_RANDOM_DRAWS,
                fill_strategy="blur", device=DEVICE, target_class=class_labels[image_id] - 1,
            )
            results.append(result)
            n_done += 1
        print(f"{part_name}: {n_done}/{N_APPROX_PER_PART} approximation instances "
              f"(from {len(uncovered_ids)} uncovered-species test images)", flush=True)

    with open(RESULTS_DIR / "cub_faithfulness.csv", "w", newline="") as f:
        fieldnames = list(vars(results[0]).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(vars(r))

    print(f"\n{len(results)} total faithfulness records saved to results/cub_faithfulness.csv")
    print(f"  real: {sum(1 for r in results if r.category == 'cub_part_real')}")
    print(f"  approx: {sum(1 for r in results if r.category == 'cub_part_approx')}")

    print("\nMean delta_p per part (real vs. approx):")
    for part_name, part_id in CUB70_TO_CUB_PART_ID.items():
        real_deltas = [r.delta_p for r in results if r.concept_number == part_id and r.category == "cub_part_real"]
        approx_deltas = [r.delta_p for r in results if r.concept_number == part_id and r.category == "cub_part_approx"]
        real_mean = np.mean(real_deltas) if real_deltas else float("nan")
        approx_mean = np.mean(approx_deltas) if approx_deltas else float("nan")
        print(f"  {part_name}: real={real_mean:.4f} (n={len(real_deltas)}), approx={approx_mean:.4f} (n={len(approx_deltas)})")


if __name__ == "__main__":
    main()
