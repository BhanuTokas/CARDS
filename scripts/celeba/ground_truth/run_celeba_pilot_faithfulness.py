"""Phase 4 of the CelebA plan: masking-based faithfulness ground truth for
the 8-concept pilot, against BOTH target classes (Attractive, Young).

Reuses cards.validation.broden_faithfulness (mask_region, random_placements,
compute_faithfulness) completely unchanged -- confirmed dataset-agnostic in
Phase 0/2. No keypoint-patch approximation anywhere in this track (unlike
CUB): every pilot concept maps to a REAL CelebAMask-HQ per-pixel region.

A real structural difference from CUB, not CUB's stratification machinery
reused blindly: CUB needed class-stratified sampling (v49) because scoring
groups by (concept, species) with 200 species meant flat sampling spread
thin across hundreds of groups. Here there are only 2 target classes, so
each concept only ever needs to populate 2 groups: (concept, Attractive)
and (concept, Young). Flat sampling from a generously large candidate pool
clears min_samples_per_pair=3 trivially by construction -- candidate-pool
sizes for every (concept, task) pair were checked directly before this run
(a background sweep over all 4,500 val images), and the per-pair sample
counts this script itself prints at the end are the authoritative record
of how close each pair actually got to the N_PER_ATTRIBUTE target (the two
rarest concepts, Eyeglasses and Wearing_Hat, were expected going in not to
reach it, still clearing the threshold by a wide margin).

Ground truth uses each image's own REAL attribute label as target_class
(CUB's v41 convention) -- specifically, candidates are drawn only from
images where the target task's own label is already True, and every
compute_faithfulness call scores target_class=1 (that task's positive
class) within a task-specific 2-way adapter (CelebaTaskAdapter, slicing
Phase 1's 4-way head). This is the same "always score the ground-truth-
positive class" design CUB's attribute run used (target_class=class_labels
[image_id]-1, always a positive species for that attribute) -- generalized
here to a fixed target_class=1 per task rather than a per-image varying
species id, since there are only 2 possible target classes total.

Because target_class is always 1 regardless of which task is running, the
resulting FaithfulnessResult.predicted_class field is constant (=1) across
BOTH tasks -- it does NOT by itself disambiguate Attractive from Young the
way CUB's species-id-valued predicted_class naturally did. The extra
`target_task` CSV column carries that distinction instead; any downstream
scoring (Phase 7) must filter records by target_task BEFORE calling
score_method_agreement/score_sign_agreement, which then sees exactly 8
(concept_number, predicted_class=1) pairs per call -- unchanged, zero-
modification reuse of the existing aggregation, just invoked once per task.

Candidate images are pre-filtered to {task attribute == True} intersected
with {concept mask non-empty} -- checked directly for all 8x2 pairs before
writing this script (a Monitor-tracked background check, not assumed) to
confirm every pair has real headroom above min_samples_per_pair=3.
"""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cards.data.celeba import (
    load_celebamask_hq_image_paths,
    load_celebamask_hq_mask,
    split_celebamask_hq,
)
from cards.data.celeba_attributes import (
    ATTRIBUTE_TO_REGIONS,
    PILOT_CONCEPTS,
    TARGET_CLASSES,
    load_attribute_labels,
    load_attribute_names,
)
from cards.models.backbones import BACKBONES
from cards.validation.broden_faithfulness import compute_faithfulness

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PER_ATTRIBUTE = 100
N_RANDOM_DRAWS = 5
FILL_STRATEGY = "blur"
MIN_SAMPLES_PER_PAIR = 3  # matches score_method_agreement/score_sign_agreement's own default


class CelebaTaskAdapter:
    """One of the 2 independent 2-way softmax blocks in Phase 1's 4-way
    head, sliced to present as a clean (N,2)-logit MultiClassModel --
    compute_faithfulness's own `target_class` override (its docstring's
    "e.g. CUB's own species labels" case) generalizes cleanly to "which
    binary value of THIS task", not just "which of N species"."""

    def __init__(self, device: str, task_slice: slice):
        spec = BACKBONES["celeba_attractive_young"]
        self.model = spec.load_native().to(device).eval()
        self._preprocess = spec.preprocess
        self.task_slice = task_slice
        self.device = device

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch.to(self.device))[:, self.task_slice]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rng_py = random.Random(SEED)

    print("Loading CelebAMask-HQ metadata...", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]

    _, val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices)
    print(f"{len(val_hq)} held-out val images (never seen by the classifier).", flush=True)

    results = []  # (FaithfulnessResult, concept_name, target_task)
    n_draws = 0

    for task_idx, task_name in enumerate(TARGET_CLASSES):
        print(f"\n=== target task: {task_name} ===", flush=True)
        adapter = CelebaTaskAdapter(DEVICE, task_slice=slice(task_idx * 2, task_idx * 2 + 2))
        task_attr_idx = attr_names.index(task_name)
        task_positive_hq = [i for i in val_hq if attr_labels_by_file[f"{i}.jpg"][task_attr_idx]]
        print(f"{len(task_positive_hq)}/{len(val_hq)} val images positive for {task_name}.", flush=True)

        for concept_idx, concept_name in enumerate(PILOT_CONCEPTS):
            region_names = ATTRIBUTE_TO_REGIONS[concept_name]
            candidates = list(task_positive_hq)
            rng_py.shuffle(candidates)

            concept_results = []
            for hq_idx in candidates:
                if len(concept_results) >= N_PER_ATTRIBUTE:
                    break
                image_path = image_paths_by_idx[hq_idx]
                image = Image.open(image_path).convert("RGB")
                mask = load_celebamask_hq_mask(
                    CELEBA_HQ_ROOT, hq_idx, region_names, target_hw=(image.height, image.width)
                )
                if not mask.any():
                    continue

                rng_np = np.random.default_rng(SEED + task_idx * 1_000_000 + concept_idx * 10_000 + n_draws)
                n_draws += 1
                result = compute_faithfulness(
                    image=image, image_path=str(image_path), concept_number=concept_idx,
                    category="pilot", mask=mask, model=adapter, rng=rng_np,
                    n_random_draws=N_RANDOM_DRAWS, fill_strategy=FILL_STRATEGY, device=DEVICE,
                    target_class=1,
                )
                concept_results.append(result)

            results.extend((r, concept_name, task_name) for r in concept_results)
            print(
                f"  {concept_name:<20s} ({'+'.join(region_names)}): {len(concept_results)}/{N_PER_ATTRIBUTE} "
                f"samples ({len(candidates)} task-positive candidates total)", flush=True,
            )

    with open(RESULTS_DIR / "celeba_pilot_faithfulness.csv", "w", newline="") as f:
        base_fields = list(vars(results[0][0]).keys())
        writer = csv.DictWriter(f, fieldnames=base_fields + ["concept_name", "target_task"])
        writer.writeheader()
        for result, concept_name, task_name in results:
            row = vars(result)
            row.update(concept_name=concept_name, target_task=task_name)
            writer.writerow(row)

    print(f"\n{len(results)} total faithfulness records saved to results/celeba_pilot_faithfulness.csv", flush=True)

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for _r, concept_name, task_name in results:
        counts[(concept_name, task_name)] += 1

    print("\nSamples per (concept, task) pair:")
    n_below_threshold = 0
    for (concept_name, task_name), n in sorted(counts.items()):
        below = n < MIN_SAMPLES_PER_PAIR
        n_below_threshold += int(below)
        flag = f"  <-- BELOW min_samples_per_pair={MIN_SAMPLES_PER_PAIR}" if below else ""
        print(f"  {concept_name:<20s} / {task_name:<12s}: n={n}{flag}")
    print(
        f"\n{len(counts)}/{2 * len(PILOT_CONCEPTS)} (concept, task) pairs populated, "
        f"{n_below_threshold} below threshold.", flush=True,
    )


if __name__ == "__main__":
    main()
