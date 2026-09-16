"""Male-task extension of `run_celeba_full_faithfulness_attribute_
conditioned.py`, prompted directly ("Can we extend to add Male to our
analysis?" -> "Full 3-method comparison"). Generates the SAME kind of
masking-based faithfulness ground truth (both fixes from that script
applied identically: concept-attribute-value-conditioned candidates,
both label directions pooled) but for the NEW `Male` task instead of
Attractive/Young.

Deliberately a SEPARATE script/output file, not an extension of
`TARGET_CLASSES` (=["Attractive","Young"]) itself -- that constant is
read by dozens of existing scripts throughout this track that all
assume exactly 2 target classes; changing it would have a huge, hard-
to-audit blast radius. Male stays a parallel, independently-scored
task, the same pattern `celeba_attractive_young_lowres` already used
for a parallel classifier variant.

Uses the NEW `celeba_attractive_young_male` backbone (6-way head,
[4:6]=Male) and the SAME held-out val_hq split as every other ground-
truth script in this track (computed from the ORIGINAL 2-class
target_indices, matching what the classifier itself trained/held out
on -- Male's own labels were never part of the stratification key,
per train_attractive_young_male_classifier.py's own design).
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
N_PER_ATTRIBUTE = 90
N_RANDOM_DRAWS = 5
FILL_STRATEGY = "blur"
MIN_SAMPLES_PER_PAIR = 3
TASK_NAME = "Male"
TASK_IDX_OFFSET = 2  # distinct from Attractive(0)/Young(1) in the original script's own seed formula


class CelebaMaleTaskAdapter:
    """The 3rd 2-way softmax block ([4:6]) of the extended 6-way head,
    sliced to present as a clean (N,2)-logit MultiClassModel -- same
    adapter shape convention as the original script's CelebaTaskAdapter."""

    def __init__(self, device: str):
        spec = BACKBONES["celeba_attractive_young_male"]
        self.model = spec.load_native().to(device).eval()
        self._preprocess = spec.preprocess
        self.device = device

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch.to(self.device))[:, 4:6]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rng_py = random.Random(SEED)

    print("Loading CelebAMask-HQ metadata...", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    split_target_indices = [attr_names.index(t) for t in TARGET_CLASSES]  # Attractive, Young ONLY -- split reproducibility

    _, val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, split_target_indices)
    print(f"{len(val_hq)} held-out val images (never seen by the classifier, SAME split as Attractive/Young).", flush=True)
    print(f"{len(GROUNDABLE_CONCEPTS)} groundable concepts to score against {TASK_NAME}.", flush=True)

    adapter = CelebaMaleTaskAdapter(DEVICE)
    task_attr_idx = attr_names.index(TASK_NAME)
    n_task_positive = sum(1 for i in val_hq if attr_labels_by_file[f"{i}.jpg"][task_attr_idx])
    print(f"{n_task_positive} positive / {len(val_hq) - n_task_positive} negative "
          f"for {TASK_NAME} (both pooled as candidates).", flush=True)

    results = []  # (FaithfulnessResult, concept_name, target_task)
    n_draws = 0

    for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
        region_names = ATTRIBUTE_TO_REGIONS[concept_name]
        concept_attr_idx = attr_names.index(concept_name)
        candidates = [i for i in val_hq if attr_labels_by_file[f"{i}.jpg"][concept_attr_idx]]
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

            rng_np = np.random.default_rng(SEED + TASK_IDX_OFFSET * 1_000_000 + concept_idx * 10_000 + n_draws)
            n_draws += 1
            result = compute_faithfulness(
                image=image, image_path=str(image_path), concept_number=concept_idx,
                category="full", mask=mask, model=adapter, rng=rng_np,
                n_random_draws=N_RANDOM_DRAWS, fill_strategy=FILL_STRATEGY, device=DEVICE,
                target_class=1,
            )
            concept_results.append(result)

        results.extend((r, concept_name, TASK_NAME) for r in concept_results)
        print(f"  {concept_name:<20s} ({'+'.join(region_names)}): {len(concept_results)}/{N_PER_ATTRIBUTE} "
              f"samples ({len(candidates)} concept-positive candidates total)", flush=True)

    out_path = RESULTS_DIR / "celeba_male_faithfulness_attribute_conditioned.csv"
    with open(out_path, "w", newline="") as f:
        base_fields = list(vars(results[0][0]).keys())
        writer = csv.DictWriter(f, fieldnames=base_fields + ["concept_name", "target_task"])
        writer.writeheader()
        for result, concept_name, task_name in results:
            row = vars(result)
            row.update(concept_name=concept_name, target_task=task_name)
            writer.writerow(row)

    print(f"\n{len(results)} total faithfulness records saved to {out_path}", flush=True)

    counts: dict[str, int] = defaultdict(int)
    for _r, concept_name, _task_name in results:
        counts[concept_name] += 1

    print("\nSamples per concept:")
    n_below_threshold = 0
    n_below_target = 0
    for concept_name, n in sorted(counts.items()):
        below_min = n < MIN_SAMPLES_PER_PAIR
        below_target = n < N_PER_ATTRIBUTE
        n_below_threshold += int(below_min)
        n_below_target += int(below_target)
        flag = f"  <-- BELOW min_samples_per_pair={MIN_SAMPLES_PER_PAIR}" if below_min else (
            "  (below target)" if below_target else ""
        )
        print(f"  {concept_name:<20s}: n={n}{flag}")
    print(f"\n{len(counts)}/{len(GROUNDABLE_CONCEPTS)} concepts populated, "
          f"{n_below_threshold} below min_samples_per_pair, {n_below_target} below the "
          f"N_PER_ATTRIBUTE={N_PER_ATTRIBUTE} target.", flush=True)


if __name__ == "__main__":
    main()
