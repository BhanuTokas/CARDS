"""Phase 7 scale-up of the CelebA plan: masking-based faithfulness ground
truth for ALL 26 groundable concepts (cards.data.celeba_attributes.
GROUNDABLE_CONCEPTS), against BOTH target classes (Attractive, Young) --
extends run_celeba_pilot_faithfulness.py's own 8-concept pilot (v65-v71)
the same way CUB's 8-part pilot (v33-v37) scaled to its 87-attribute bank
(v38+). Logic is otherwise IDENTICAL to the pilot script -- see that
script's own docstring for the design rationale (target_class=1 always,
target_task CSV column disambiguates tasks, candidate pre-filtering).

Candidate-pool sizes for the 18 NEW concepts (beyond the pilot's own 8)
were checked directly before this run, same discipline as the pilot's
own pre-check: the tightest is Wearing_Necklace/Attractive at 150
candidates (vs. the pilot's own tightest, Eyeglasses/Attractive at 19) --
comfortably above min_samples_per_pair=3 for every one of the 52
(concept, task) pairs.
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
    GROUNDABLE_CONCEPTS,
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
    head, sliced to present as a clean (N,2)-logit MultiClassModel."""

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
    print(f"{len(GROUNDABLE_CONCEPTS)} groundable concepts to score.", flush=True)

    results = []  # (FaithfulnessResult, concept_name, target_task)
    n_draws = 0

    for task_idx, task_name in enumerate(TARGET_CLASSES):
        print(f"\n=== target task: {task_name} ===", flush=True)
        adapter = CelebaTaskAdapter(DEVICE, task_slice=slice(task_idx * 2, task_idx * 2 + 2))
        task_attr_idx = attr_names.index(task_name)
        task_positive_hq = [i for i in val_hq if attr_labels_by_file[f"{i}.jpg"][task_attr_idx]]
        print(f"{len(task_positive_hq)}/{len(val_hq)} val images positive for {task_name}.", flush=True)

        for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
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
                    category="full", mask=mask, model=adapter, rng=rng_np,
                    n_random_draws=N_RANDOM_DRAWS, fill_strategy=FILL_STRATEGY, device=DEVICE,
                    target_class=1,
                )
                concept_results.append(result)

            results.extend((r, concept_name, task_name) for r in concept_results)
            print(
                f"  {concept_name:<20s} ({'+'.join(region_names)}): {len(concept_results)}/{N_PER_ATTRIBUTE} "
                f"samples ({len(candidates)} task-positive candidates total)", flush=True,
            )

    with open(RESULTS_DIR / "celeba_full_faithfulness.csv", "w", newline="") as f:
        base_fields = list(vars(results[0][0]).keys())
        writer = csv.DictWriter(f, fieldnames=base_fields + ["concept_name", "target_task"])
        writer.writeheader()
        for result, concept_name, task_name in results:
            row = vars(result)
            row.update(concept_name=concept_name, target_task=task_name)
            writer.writerow(row)

    print(f"\n{len(results)} total faithfulness records saved to results/celeba_full_faithfulness.csv", flush=True)

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
        f"\n{len(counts)}/{2 * len(GROUNDABLE_CONCEPTS)} (concept, task) pairs populated, "
        f"{n_below_threshold} below threshold.", flush=True,
    )


if __name__ == "__main__":
    main()
