"""zero_fill variant of run_celeba_full_faithfulness.py -- identical in
every respect (same 26 concepts, same candidate pre-filtering, same
target_class=1 convention) except FILL_STRATEGY, and writes to a
separate output file so the original blur-based ground truth
(results/celeba_full_faithfulness.csv) is never overwritten -- both are
kept side by side to compare directly.

Prompted directly ("Can you redo the experiments with zero_fill?"),
following CUB's own established masking-strategy-bias lesson (notes/
cub_correlation_investigation.md v61): blur preserves a masked region's
mean color (a low-pass filter, by construction) and so under-erases
color-defined concepts specifically, while zero_fill destroys color
completely but also erases spatial structure and creates an
out-of-distribution "hole" -- a different confound of its own. Re-running
the full 26-concept comparison under zero_fill checks whether v72's
"every method collapses to chance" finding is itself an artifact of the
blur strategy specifically, or holds regardless of which masking
strategy generates the ground truth.

CARDS/TCAV/PCBM scores are NOT re-run here -- none of the three methods'
own scoring depends on the masking strategy at all (masking only touches
how the ground truth itself is generated), so the existing
`cards_celeba_full_scores.csv`/`tcav_celeba_full_scores.csv`/PCBM
checkpoints are reused unchanged when this new ground truth is scored.
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
FILL_STRATEGY = "zero_fill"
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

    with open(RESULTS_DIR / "celeba_full_faithfulness_zerofill.csv", "w", newline="") as f:
        base_fields = list(vars(results[0][0]).keys())
        writer = csv.DictWriter(f, fieldnames=base_fields + ["concept_name", "target_task"])
        writer.writeheader()
        for result, concept_name, task_name in results:
            row = vars(result)
            row.update(concept_name=concept_name, target_task=task_name)
            writer.writerow(row)

    print(f"\n{len(results)} total faithfulness records saved to results/celeba_full_faithfulness_zerofill.csv", flush=True)

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
