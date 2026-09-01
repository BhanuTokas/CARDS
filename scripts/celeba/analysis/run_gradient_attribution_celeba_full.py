"""Alternate baseline for the CelebA full 26-concept comparison: gradient-
based PIXEL attribution (Integrated Gradients), scored against the SAME
masking-based faithfulness ground truth CARDS/TCAV/PCBM are scored
against -- prompted directly ("Can we use the Gradient based pixel
attribution as baseline?"), following a recommendation made when v72/v73
showed all 3 existing methods (CARDS, TCAV, PCBM) collapsing to chance:
this is a genuinely different methodological family (none of the other
3 look at per-pixel gradients, they all reason about concept VECTORS),
so it answers a different question -- does even basic gradient
attribution correlate with the masking ground truth, or does the null
result generalize beyond concept-vector methods specifically?

Design: for each (concept, task) pair, reuse the SAME sample images the
blur-based ground truth (celeba_full_faithfulness.csv) already drew for
that pair (first N_SAMPLES_PER_PAIR of them, in that file's own saved
order) -- no new candidate-selection logic, and the exact same
population every method in this comparison is scored against. For each
image, run Integrated Gradients w.r.t. the task's own positive-class
logit (target=1 within a 2-way task slice, matching every other script
in this track), sum attribution across color channels, and reduce to a
single per-(concept, task) importance score:
  mean(attribution inside the concept's mask) - mean(attribution outside it)
-- a present-vs-absent contrast, mirroring CARDS' own raw_score
convention and controlling for per-image attribution-scale differences
across images, the same way random_placements controls for that in the
masking metric itself.

n_steps=20 (not captum's default 50) and N_SAMPLES_PER_PAIR=25 (not the
full ~100 the masking ground truth itself used) are both deliberately
reduced from what a definitive IG study would use -- this is a baseline
sanity check, not the primary result, and the total compute (26 concepts
x 2 tasks x 25 images x 20 steps = 26,000 forward+backward evaluations)
is calibrated to roughly match the full ground-truth run's own cost
(35,350 forward passes), not to be authoritative on its own.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from captum.attr import IntegratedGradients
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cards.data.celeba import load_celebamask_hq_mask
from cards.data.celeba_attributes import ATTRIBUTE_TO_REGIONS, GROUNDABLE_CONCEPTS, TARGET_CLASSES
from cards.models.backbones import BACKBONES
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES_PER_PAIR = 25
N_STEPS = 20

TASK_SLICES: dict[str, slice] = {"Attractive": slice(0, 2), "Young": slice(2, 4)}


def load_sample_paths_by_pair() -> dict[tuple[str, str], list[Path]]:
    """(concept_name, target_task) -> first N_SAMPLES_PER_PAIR image paths
    the blur-based ground truth already drew for that pair, in that
    file's own saved order -- reusing the exact population every other
    method here is scored against, not a fresh draw."""
    by_pair: dict[tuple[str, str], list[Path]] = {}
    with open(RESULTS_DIR / "celeba_full_faithfulness.csv", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["concept_name"], row["target_task"])
            paths = by_pair.setdefault(key, [])
            if len(paths) < N_SAMPLES_PER_PAIR:
                paths.append(Path(row["image"]))
    return by_pair


def load_records_by_task() -> dict[str, list[FaithfulnessResult]]:
    concept_to_idx = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
    by_task: dict[str, list[FaithfulnessResult]] = {t: [] for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "celeba_full_faithfulness.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]].append(FaithfulnessResult(
                image=row["image"], concept_number=concept_to_idx[row["concept_name"]], category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return by_task


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()
    preprocess = spec.preprocess

    sample_paths_by_pair = load_sample_paths_by_pair()
    print(f"Loaded sample image paths for {len(sample_paths_by_pair)} (concept, task) pairs.", flush=True)

    rows = []
    for task_name in TARGET_CLASSES:
        task_slice = TASK_SLICES[task_name]

        def forward_func(batch: torch.Tensor, task_slice=task_slice) -> torch.Tensor:
            return native_model(batch)[:, task_slice]

        ig = IntegratedGradients(forward_func)

        for concept_name in GROUNDABLE_CONCEPTS:
            region_names = ATTRIBUTE_TO_REGIONS[concept_name]
            paths = sample_paths_by_pair.get((concept_name, task_name), [])
            contrasts = []
            for path in paths:
                hq_idx = int(path.stem)
                image = Image.open(path).convert("RGB")
                mask = load_celebamask_hq_mask(
                    CELEBA_HQ_ROOT, hq_idx, region_names, target_hw=(image.height, image.width)
                )
                if not mask.any() or mask.all():
                    continue

                x = preprocess(image).unsqueeze(0).to(DEVICE)
                x.requires_grad_(True)
                attr = ig.attribute(x, target=1, n_steps=N_STEPS)  # (1, 3, H, W)
                attr_map = attr[0].sum(dim=0).detach().cpu().numpy()  # (H, W), sum over channels

                mask_resized = mask
                if mask_resized.shape != attr_map.shape:
                    mask_img = Image.fromarray(mask_resized.astype(np.uint8) * 255).resize(
                        (attr_map.shape[1], attr_map.shape[0]), Image.NEAREST
                    )
                    mask_resized = np.array(mask_img) > 0

                inside = attr_map[mask_resized].mean()
                outside = attr_map[~mask_resized].mean()
                contrasts.append(float(inside - outside))

            raw_score = float(np.mean(contrasts)) if contrasts else 0.0
            rows.append((concept_name, task_name, raw_score, len(contrasts)))
            print(f"{concept_name:<20s} / {task_name:<12s}: n={len(contrasts):>3d}  "
                  f"gradient_score={raw_score:+.5f}", flush=True)

    with open(RESULTS_DIR / "gradient_attribution_celeba_full_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "target_task", "raw_score", "n_samples"])
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} (concept, task) gradient-attribution scores to "
          f"results/gradient_attribution_celeba_full_scores.csv", flush=True)

    print("\n=== scoring against the masking-based faithfulness ground truth ===", flush=True)
    concept_to_idx = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
    records_by_task = load_records_by_task()
    for task_name in TARGET_CLASSES:
        scores = {
            (concept_to_idx[c], 1): s for c, t, s, _n in rows if t == task_name
        }
        agree = score_method_agreement(records_by_task[task_name], scores)
        sign = score_sign_agreement(records_by_task[task_name], scores)
        if agree is not None:
            print(f"{task_name}: rho={agree.spearman_rho:+.3f} (p={agree.spearman_p:.3f}, n={agree.n_pairs})  "
                  f"sign={sign.agreement_frac:.1%} ({sign.n_agree}/{sign.n_pairs}, p={sign.binom_p:.3f})", flush=True)
        else:
            print(f"{task_name}: too few pairs to score", flush=True)


if __name__ == "__main__":
    main()
