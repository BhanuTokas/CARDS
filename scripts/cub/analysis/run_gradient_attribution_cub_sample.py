"""CUB-track analogue of the CelebA gradient-attribution baseline
(scripts/celeba/analysis/run_gradient_attribution_celeba_full.py, v74)
-- prompted directly ("Can we run the same thing on CUB as well?"),
checking whether CUB shows the same pattern CelebA did: all 3 concept-
VECTOR methods (CARDS, TCAV, PCBM) at chance (CUB's own v62 conclusion),
but Integrated Gradients -- a structurally different, pixel-level
attribution family -- significantly beats chance (CelebA v74).

**Scope, chosen directly by the user after being asked**: CUB's own
ground truth (results/cub_attribute_faithfulness.csv) has 3,526
(attribute, species) pairs -- 68x CelebA's 52. Running IG at CelebA's
same depth (25 samples/pair, n_steps=20) over all of them would mean
~1.76M forward/backward evaluations, many hours. Instead: a random
sample of N_PAIRS=350 pairs (~10% of the full ground truth, seeded for
reproducibility), at REDUCED depth (N_SAMPLES_PER_PAIR=10, N_STEPS=10)
so total compute (350*10*10=35,000 evaluations) stays in the same order
of magnitude as CelebA's own run (26,000) rather than scaling up with
pair count -- comparable wall-clock budget, broader but shallower
coverage.

**Mask reconstruction, not stored in the ground truth CSV directly**:
CUB's own masking ground truth (v41/v56) used keypoint-patch
approximations (`cards.data.cub_parts.keypoint_patch_mask`), not a
simple lookup like CelebA's `load_celebamask_hq_mask` -- reconstructed
here EXACTLY as the original ground truth run computed it (same
keypoint, same real-silhouette-scaled target_area via `PART_AREA_RATIO`)
so the region masked here is identical to what the ground truth's own
delta_p was computed against, not a fresh/different approximation. The
ground truth CSV's own saved `part_name` column (added specifically for
this kind of downstream reuse) is what makes this reconstruction
possible without re-deriving the attribute->part mapping.

Present-vs-absent contrast score (mean(attribution inside mask) -
mean(outside)) and target=predicted_class convention are otherwise
identical to the CelebA script.
"""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from captum.attr import IntegratedGradients
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cards.data.cub_attributes import PART_AREA_RATIO
from cards.data.cub_parts import (
    keypoint_patch_mask,
    load_cub_segmentation,
    load_images_txt,
    load_keypoints,
)
from cards.models.backbones import BACKBONES
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_PAIRS = 350
N_SAMPLES_PER_PAIR = 10
N_STEPS = 10

# same mapping run_cub_attribute_faithfulness.py itself used
PART_NAME_TO_ID = {
    "back": 1, "beak": 2, "belly": 3, "breast": 4, "crown": 5, "forehead": 6,
    "left_eye": 7, "left_leg": 8, "left_wing": 9, "nape": 10, "right_eye": 11,
    "right_leg": 12, "right_wing": 13, "tail": 14, "throat": 15,
}


def load_ground_truth() -> list[FaithfulnessResult]:
    records = []
    with open(RESULTS_DIR / "cub_attribute_faithfulness.csv", newline="") as f:
        for row in csv.DictReader(f):
            records.append(FaithfulnessResult(
                image=row["image"], concept_number=int(row["concept_number"]), category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return records


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    print("Loading CUB ground truth + metadata...", flush=True)
    records = load_ground_truth()
    print(f"{len(records)} ground-truth records loaded.", flush=True)

    # (concept_number, predicted_class) -> [(image_path, part_name), ...],
    # and concept_number -> part_name (constant per concept)
    pair_to_images: dict[tuple[int, int], list[str]] = defaultdict(list)
    concept_to_part: dict[int, str] = {}
    with open(RESULTS_DIR / "cub_attribute_faithfulness.csv", newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["concept_number"]), int(row["predicted_class"]))
            pair_to_images[key].append(row["image"])
            concept_to_part[int(row["concept_number"])] = row["part_name"]

    all_pairs = list(pair_to_images.keys())
    rng.shuffle(all_pairs)
    sampled_pairs = all_pairs[:N_PAIRS]
    print(f"{len(all_pairs)} total pairs available; sampling {len(sampled_pairs)} for this check.", flush=True)

    image_paths = load_images_txt(CUB_ROOT)
    path_to_id = {str(p): iid for iid, p in image_paths.items()}
    keypoints = load_keypoints(CUB_ROOT)

    spec = BACKBONES["resnet18_cub"]
    native_model = spec.load_native().to(DEVICE).eval()
    preprocess = spec.preprocess

    def forward_func(batch: torch.Tensor) -> torch.Tensor:
        return native_model(batch)

    ig = IntegratedGradients(forward_func)

    rows = []
    for i, (concept_idx, predicted_class) in enumerate(sampled_pairs):
        part_name = concept_to_part[concept_idx]
        part_id = PART_NAME_TO_ID[part_name]
        candidates = list(pair_to_images[(concept_idx, predicted_class)])
        rng.shuffle(candidates)

        contrasts = []
        for image_path_str in candidates:
            if len(contrasts) >= N_SAMPLES_PER_PAIR:
                break
            image_id = path_to_id.get(image_path_str)
            if image_id is None:
                continue
            kp = keypoints.get(image_id, {}).get(part_id)
            if kp is None or not kp[2]:
                continue
            x, y, _visible = kp

            image = Image.open(image_path_str).convert("RGB")
            try:
                silhouette = load_cub_segmentation(CUB_ROOT, image_id, image_paths)
            except FileNotFoundError:
                continue
            if silhouette.shape != (image.height, image.width) or silhouette.sum() == 0:
                continue
            target_area = PART_AREA_RATIO[part_name] * silhouette.sum()
            mask = keypoint_patch_mask(x, y, target_area, (image.height, image.width))
            if not mask.any() or mask.all():
                continue

            inp = preprocess(image).unsqueeze(0).to(DEVICE)
            inp.requires_grad_(True)
            attr = ig.attribute(inp, target=predicted_class, n_steps=N_STEPS)  # (1, 3, H, W)
            attr_map = attr[0].sum(dim=0).detach().cpu().numpy()

            mask_resized = mask
            if mask_resized.shape != attr_map.shape:
                mask_img = Image.fromarray(mask_resized.astype(np.uint8) * 255).resize(
                    (attr_map.shape[1], attr_map.shape[0]), Image.NEAREST
                )
                mask_resized = np.array(mask_img) > 0

            inside = attr_map[mask_resized].mean()
            outside = attr_map[~mask_resized].mean()
            contrasts.append(float(inside - outside))

        raw_score = float(np.mean(contrasts)) if contrasts else None
        if raw_score is not None:
            rows.append((concept_idx, predicted_class, part_name, raw_score, len(contrasts)))

        if (i + 1) % 25 == 0 or i == len(sampled_pairs) - 1:
            print(f"[{i + 1}/{len(sampled_pairs)}] pairs processed, {len(rows)} scored so far", flush=True)

    with open(RESULTS_DIR / "gradient_attribution_cub_sample_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_number", "predicted_class", "part_name", "raw_score", "n_samples"])
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} (concept, class) gradient-attribution scores to "
          f"results/gradient_attribution_cub_sample_scores.csv", flush=True)

    print("\n=== scoring against the masking-based faithfulness ground truth ===", flush=True)
    scores = {(concept_idx, predicted_class): s for concept_idx, predicted_class, _p, s, _n in rows}
    agree = score_method_agreement(records, scores)
    sign = score_sign_agreement(records, scores)
    if agree is not None:
        print(f"n={agree.n_pairs} rho={agree.spearman_rho:+.4f} (p={agree.spearman_p:.4g})  "
              f"sign={sign.agreement_frac:.1%} ({sign.n_agree}/{sign.n_pairs}, p={sign.binom_p:.4g})", flush=True)
    else:
        print("too few pairs to score", flush=True)


if __name__ == "__main__":
    main()
