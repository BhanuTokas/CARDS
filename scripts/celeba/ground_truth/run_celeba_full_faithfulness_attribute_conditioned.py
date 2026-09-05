"""A corrected masking-based faithfulness ground truth, addressing two
real limitations of `run_celeba_full_faithfulness.py` found while
building the LOCAL attribution experiment (v112) and discussed at length
afterward -- NOT a replacement for that file (still the ground truth
every other result in this track was scored against), a separate,
explicitly-labeled variant.

**Limitation 1 (found via `local_attribution_comparison_celeba.py`'s own
top-10 grids, verified against real labels): candidates were filtered
only by target-task positivity and region-mask presence, NEVER by the
concept's own attribute value.** Ground truth for `Pale_Skin` had ZERO
of its top-10-by-delta_p images actually labeled `Pale_Skin=True`.
Masking the `skin` region measures "does this region matter," which
conflates the specific attribute value with everything else co-located
in that region (texture, evenness, blemishes, makeup) -- a real,
irreducible limitation given CelebAMask-HQ never segments finer than
whole regions (discussed at length; NOT fixed here, since no annotation
exists to isolate a value's own pixel footprint within its region).
**Fixed here**: candidates additionally require
`attr_labels_by_file[...][concept_attr_idx] == True` -- the sampled
images at least genuinely have the concept, even though the mask itself
still can't isolate just that value's own contribution.

**Limitation 2 (prompted directly, "What if we assigned a negative
value where the person is not attractive? ... how much did masking
baldness increase attractiveness?"): candidates were restricted to
target-task-POSITIVE images only** (`task_positive_hq` in the original
script), which starves any concept anti-correlated with the target task
of samples -- `Bald`+`Attractive=True` had exactly 1 candidate in all of
HQ-val. **No sign-flip is needed to fix this**: `compute_faithfulness`
already computes `delta_p = p(target_class=1) - p_masked(target_class=1)`
against a FIXED class index (the task's own positive class), regardless
of the image's own actual label for that task -- confirmed directly
against `compute_faithfulness`'s own implementation. So a Bald,
NOT-Attractive person's delta_p is already directly poolable with a
Bald, Attractive person's delta_p under the exact same sign convention:
positive delta_p means the concept was helping p(Attractive) (masking
it away reduced p(Attractive)); negative means it was hurting (masking
it away increased p(Attractive)) -- true regardless of which side of
the Attractive/Young label the sampled person started on. **Fixed
here**: candidates are drawn from ALL of val_hq (both target_class=True
and target_class=False), not just the positive side.

**Sample-size verification, done directly on real data before running**
(both fixes combined, on the full 4,500-image HQ-val pool -- the 477-
image overlap with the official CelebA val pool used elsewhere in this
track was checked and found NOT to require exclusion here: that
protection against circularity already exists on the OFFICIAL-val side,
`build_clean_official_val_paths()` already excludes this same overlap
from ITS OWN pool, so official-val never touches these images
regardless of what HQ-val's own ground truth does): the tightest pair,
`Bald`/Attractive, has 95 candidates pooled across both label
directions (1 target-positive + 94 target-negative) -- comfortably
above `N_PER_ATTRIBUTE=90` (lowered from the original 100 specifically
so every single (concept,task) pair, including this tightest one, can
reach the FULL target depth uniformly, not just clear the
`min_samples_per_pair=3` viability floor).

Logic is otherwise IDENTICAL to `run_celeba_full_faithfulness.py` --
same RNG seeding scheme, same fill strategy, same `target_class=1`
convention, same output schema (plus this file's own output path).
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
        # Both label directions pooled (fix for limitation 2) -- delta_p
        # is always w.r.t. the SAME fixed target_class=1 below, so a
        # target-negative image's delta_p is already directly comparable
        # to a target-positive image's, no sign adjustment needed.
        n_task_positive = sum(1 for i in val_hq if attr_labels_by_file[f"{i}.jpg"][task_attr_idx])
        print(f"{n_task_positive} positive / {len(val_hq) - n_task_positive} negative "
              f"for {task_name} (both pooled as candidates).", flush=True)

        for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
            region_names = ATTRIBUTE_TO_REGIONS[concept_name]
            concept_attr_idx = attr_names.index(concept_name)
            # Concept attribute value required (fix for limitation 1) --
            # on top of the region-mask-presence check below.
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
                f"samples ({len(candidates)} concept-positive candidates total)", flush=True,
            )

    out_path = RESULTS_DIR / "celeba_full_faithfulness_attribute_conditioned.csv"
    with open(out_path, "w", newline="") as f:
        base_fields = list(vars(results[0][0]).keys())
        writer = csv.DictWriter(f, fieldnames=base_fields + ["concept_name", "target_task"])
        writer.writeheader()
        for result, concept_name, task_name in results:
            row = vars(result)
            row.update(concept_name=concept_name, target_task=task_name)
            writer.writerow(row)

    print(f"\n{len(results)} total faithfulness records saved to {out_path}", flush=True)

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for _r, concept_name, task_name in results:
        counts[(concept_name, task_name)] += 1

    print("\nSamples per (concept, task) pair:")
    n_below_threshold = 0
    n_below_target = 0
    for (concept_name, task_name), n in sorted(counts.items()):
        below_min = n < MIN_SAMPLES_PER_PAIR
        below_target = n < N_PER_ATTRIBUTE
        n_below_threshold += int(below_min)
        n_below_target += int(below_target)
        flag = f"  <-- BELOW min_samples_per_pair={MIN_SAMPLES_PER_PAIR}" if below_min else (
            "  (below target)" if below_target else ""
        )
        print(f"  {concept_name:<20s} / {task_name:<12s}: n={n}{flag}")
    print(
        f"\n{len(counts)}/{2 * len(GROUNDABLE_CONCEPTS)} (concept, task) pairs populated, "
        f"{n_below_threshold} below min_samples_per_pair, {n_below_target} below the "
        f"N_PER_ATTRIBUTE={N_PER_ATTRIBUTE} target.", flush=True,
    )


if __name__ == "__main__":
    main()
