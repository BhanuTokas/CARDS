"""Validates the cheap "keypoint + fixed-area patch" faithfulness
approximation against CUB70-PartSegmentationDataset's real per-part
masks, on the 67 CUB classes where both exist -- if they track closely,
the same approximation is trustworthy enough to extend faithfulness
testing to the other ~130 CUB classes that only have keypoints, not real
masks (notes/pcbm_correlation_investigation.md).

For each (image, part) instance where both a real CUB70 mask AND a
visible CUB keypoint exist: build a patch mask centered on the keypoint,
sized to that *same instance's own real mask area* (a same-area,
different-shape/location comparison -- the fairest test of "does the
cheap proxy behave like the real thing", not "does it happen to be the
same size"). Runs compute_faithfulness once per condition, both against
the same native resnet18_cub model.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr, ttest_rel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cards.data.cub_parts import (  # noqa: E402
    CUB70_TO_CUB_PART_ID,
    keypoint_patch_mask,
    load_cub70_mask,
    load_images_txt,
    load_keypoints,
)
from cards.models.posthoc_cbm import cub_preprocess  # noqa: E402
from cards.validation.broden_faithfulness import compute_faithfulness  # noqa: E402

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
CUB70_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB70_part_segmentation")
RESULTS_DIR = Path("results")
SEED = 42
DEVICE = "cpu"
N_INSTANCES_PER_PART = 40
N_RANDOM_DRAWS = 5


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

    print("Loading CUB images.txt and keypoints...", flush=True)
    image_paths = load_images_txt(CUB_ROOT)
    keypoints = load_keypoints(CUB_ROOT)

    # Build the list of CUB70-covered (class_dir, image_stem, image_id) --
    # infer image_id by matching filename stems against images.txt, since
    # CUB70's files are named by image stem, not CUB's own numeric id.
    stem_to_id = {Path(p).stem: image_id for image_id, p in image_paths.items()}

    rng_py = random.Random(SEED)
    all_pairs = []  # (part_name, image_id, image_path, real_mask_path_class_dir, image_stem)
    for class_dir in sorted((CUB70_ROOT / "AnnotationMasksPerclass").iterdir(), key=lambda p: int(p.name)):
        for mask_file in class_dir.glob("*.png"):
            # filename: "<...stem..>_<part_name>.png" -- part names can contain
            # underscores (e.g. "left_wing"), so match against known part names.
            name = mask_file.stem
            for part_name in CUB70_TO_CUB_PART_ID:
                suffix = f"_{part_name}"
                if name.endswith(suffix):
                    stem = name[: -len(suffix)]
                    image_id = stem_to_id.get(stem)
                    if image_id is not None:
                        all_pairs.append((part_name, image_id, class_dir.name, stem))
                    break

    print(f"{len(all_pairs)} total (part, image) mask files found across {len(list(CUB70_ROOT.glob('AnnotationMasksPerclass/*')))} classes.", flush=True)

    by_part: dict[str, list] = {}
    for part_name, image_id, class_dir, stem in all_pairs:
        by_part.setdefault(part_name, []).append((image_id, class_dir, stem))

    results = []
    for part_name, part_id in CUB70_TO_CUB_PART_ID.items():
        candidates = by_part.get(part_name, [])
        rng_py.shuffle(candidates)
        print(f"\n=== {part_name} (CUB part_id={part_id}): {len(candidates)} candidates ===", flush=True)

        n_done = 0
        for image_id, class_dir, stem in candidates:
            if n_done >= N_INSTANCES_PER_PART:
                break
            kp = keypoints.get(image_id, {}).get(part_id)
            if kp is None or not kp[2]:  # not visible
                continue
            x, y, _visible = kp

            real_mask = load_cub70_mask(CUB70_ROOT, class_dir, stem, part_name)
            if real_mask is None or not real_mask.any():
                continue

            image_path = image_paths[image_id]
            image = Image.open(image_path).convert("RGB")
            if real_mask.shape != (image.height, image.width):
                continue

            area = float(real_mask.sum())
            patch_mask = keypoint_patch_mask(x, y, area, (image.height, image.width))
            if not patch_mask.any():
                continue

            rng_real = np.random.default_rng(SEED + n_done)
            rng_patch = np.random.default_rng(SEED + 10000 + n_done)

            real_result = compute_faithfulness(
                image=image, image_path=str(image_path), concept_number=part_id, category="cub_part",
                mask=real_mask, model=model, rng=rng_real, n_random_draws=N_RANDOM_DRAWS,
                fill_strategy="blur", device=DEVICE,
            )
            patch_result = compute_faithfulness(
                image=image, image_path=str(image_path), concept_number=part_id, category="cub_part_approx",
                mask=patch_mask, model=model, rng=rng_patch, n_random_draws=N_RANDOM_DRAWS,
                fill_strategy="blur", device=DEVICE,
            )

            results.append(
                {
                    "part": part_name,
                    "image": str(image_path),
                    "real_area": area,
                    "patch_area": float(patch_mask.sum()),
                    "predicted_class_real": real_result.predicted_class,
                    "predicted_class_patch": patch_result.predicted_class,
                    "real_delta_p": real_result.delta_p,
                    "patch_delta_p": patch_result.delta_p,
                    "real_random_delta_p": real_result.random_delta_p_mean,
                    "patch_random_delta_p": patch_result.random_delta_p_mean,
                }
            )
            n_done += 1
            if n_done % 10 == 0:
                print(f"  [{n_done}/{N_INSTANCES_PER_PART}] processed", flush=True)

        print(f"{part_name}: {n_done} paired instances completed.", flush=True)

    with open(RESULTS_DIR / "cub_mask_approximation_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n\n########## COMPARISON: real mask vs. keypoint-patch approximation ##########")
    real_deltas = [r["real_delta_p"] for r in results]
    patch_deltas = [r["patch_delta_p"] for r in results]
    pear_r, pear_p = pearsonr(real_deltas, patch_deltas)
    spear_r, spear_p = spearmanr(real_deltas, patch_deltas)
    t_stat, t_p = ttest_rel(real_deltas, patch_deltas)
    agree_class = np.mean([r["predicted_class_real"] == r["predicted_class_patch"] for r in results])
    print(f"n={len(results)} paired instances across {len(CUB70_TO_CUB_PART_ID)} parts")
    print(f"Pearson r(real, patch) = {pear_r:.4f} (p={pear_p:.4g})")
    print(f"Spearman rho(real, patch) = {spear_r:.4f} (p={spear_p:.4g})")
    print(f"paired t-test (real vs patch delta_p): t={t_stat:.4f}, p={t_p:.4g}")
    print(f"mean real delta_p={np.mean(real_deltas):.4f}, mean patch delta_p={np.mean(patch_deltas):.4f}")
    print(f"fraction with identical unmasked predicted_class (real vs patch conditions): {agree_class:.4f}")

    print("\nPer-part breakdown:")
    for part_name in CUB70_TO_CUB_PART_ID:
        part_rows = [r for r in results if r["part"] == part_name]
        if len(part_rows) < 3:
            print(f"  {part_name}: n={len(part_rows)}, too few for stats")
            continue
        pr, _ = pearsonr([r["real_delta_p"] for r in part_rows], [r["patch_delta_p"] for r in part_rows])
        print(f"  {part_name}: n={len(part_rows)}, Pearson r={pr:.4f}, "
              f"mean real={np.mean([r['real_delta_p'] for r in part_rows]):.4f}, "
              f"mean patch={np.mean([r['patch_delta_p'] for r in part_rows]):.4f}")

    print("\nSaved results/cub_mask_approximation_check.csv")


if __name__ == "__main__":
    main()
