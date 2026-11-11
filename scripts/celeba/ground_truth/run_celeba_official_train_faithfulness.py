"""Masking-based faithfulness ground truth for the official-train
classifier (`train_official_celeba_classifier.py`), prompted directly
as part of "Can we train a classifier on the CelebA official train set
and see if it holds the same pattern?" then "Can you create two ground
truth versions with the non-overlapping and the previously used set."

**Generalized to any official-train backbone** (`CARDS_BACKBONE_NAME`
env var, default `celeba_official_train_attractive_male` -- the
original ResNet18), prompted directly ("The full pipeline with PCBM
variants" for the newly-trained ViT-B/16 and ConvNeXt-Tiny official-
train classifiers). Previously hardcoded a ResNet18 construction
directly (`load_official_train_native`/`official_train_preprocess`)
instead of going through the `BACKBONES` registry the rest of this
track's scripts already use -- refactored to `BACKBONES[BACKBONE_NAME]`
so this works for any registered official-train backbone unchanged;
the [0:2]=Attractive/[2:4]=Male 4-way-logit-head slicing convention is
identical across all three architectures, only the backbone underneath
differs. Output filenames get a model-name suffix derived from
`BACKBONE_NAME` (empty for the original ResNet18, preserving the
existing filenames exactly; `_vit`/`_convnext` for the new ones).

Produces TWO versions, both scored through the SAME new classifier:

  - "non_overlapping": the 5,817 CelebAMask-HQ images that fall in
    OFFICIAL CelebA's val (2,993) or test (2,824) partitions --
    verified directly to be disjoint from official train, where this
    classifier's gradient updates came from. Genuinely clean.
  - "previously_used": the ORIGINAL CelebAMask-HQ val split (4,500
    images, `split_celebamask_hq`'s own held-out set) -- the SAME
    image source every other ground-truth CSV in this track was built
    from. **Confirmed directly, not assumed: 3,587/4,500 (79.7%) of
    these images are now inside the new classifier's OWN official-
    train set** -- this version carries known, substantial leakage for
    THIS classifier specifically. Included anyway, deliberately, so
    the two versions can be compared directly against each other and
    against every prior result in this track that used this same
    image source (apples-to-apples comparability vs. genuine held-out
    cleanliness are two different, both useful, things to know).

CelebAMask-HQ is the only source of real segmentation masks in this
track, so both versions draw from it. Same corrected methodology as
`run_celeba_full_faithfulness_attribute_conditioned.py` /
`run_celeba_male_faithfulness_attribute_conditioned.py` (concept-
attribute-value-conditioned candidates, both label directions pooled).
Both Attractive and Male are scored (Young dropped per direct
instruction).
"""

from __future__ import annotations

import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

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

CELEBA_HQ_ROOT = Path(os.environ.get("CELEBA_HQ_ROOT", r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ"))
CELEBA_ROOT = Path(os.environ.get("CELEBA_ROOT", r"C:\Users\btokas\Projects\Datasets\CelebA\celeba"))
RESULTS_DIR = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
BACKBONE_NAME = os.environ.get("CARDS_BACKBONE_NAME", "celeba_official_train_attractive_male")
MODEL_SUFFIX = BACKBONE_NAME.replace("celeba_official_train_attractive_male", "")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PER_ATTRIBUTE = 90
N_RANDOM_DRAWS = 5
FILL_STRATEGY = "blur"
MIN_SAMPLES_PER_PAIR = 3
TASKS = ["Attractive", "Male"]
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}


class OfficialTrainTaskAdapter:
    def __init__(self, native_model: nn.Module, device: str, task_slice: slice, preprocess):
        self.model = native_model.to(device).eval()
        self._preprocess = preprocess
        self.task_slice = task_slice
        self.device = device

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch.to(self.device))[:, self.task_slice]


def generate_ground_truth(
    version_name: str, candidate_hq_indices: list[int], image_paths_by_idx, attr_names, attr_labels_by_file,
    native_model, preprocess, seed_offset: int,
) -> list[tuple]:
    rng_py = random.Random(SEED + seed_offset)
    results = []  # (FaithfulnessResult, concept_name, target_task)
    n_draws = 0

    for task_idx, task_name in enumerate(TASKS):
        print(f"\n=== [{version_name}] target task: {task_name} ===", flush=True)
        adapter = OfficialTrainTaskAdapter(native_model, DEVICE, slice(task_idx * 2, task_idx * 2 + 2), preprocess)
        task_attr_idx = attr_names.index(task_name)
        n_task_positive = sum(1 for i in candidate_hq_indices if attr_labels_by_file[f"{i}.jpg"][task_attr_idx])
        print(f"{n_task_positive} positive / {len(candidate_hq_indices) - n_task_positive} negative "
              f"for {task_name} (both pooled as candidates).", flush=True)

        for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
            region_names = ATTRIBUTE_TO_REGIONS[concept_name]
            concept_attr_idx = attr_names.index(concept_name)
            candidates = [i for i in candidate_hq_indices if attr_labels_by_file[f"{i}.jpg"][concept_attr_idx]]
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

                rng_np = np.random.default_rng(SEED + seed_offset + task_idx * 1_000_000 + concept_idx * 10_000 + n_draws)
                n_draws += 1
                result = compute_faithfulness(
                    image=image, image_path=str(image_path), concept_number=concept_idx,
                    category="full", mask=mask, model=adapter, rng=rng_np,
                    n_random_draws=N_RANDOM_DRAWS, fill_strategy=FILL_STRATEGY, device=DEVICE,
                    target_class=1,
                )
                concept_results.append(result)

            results.extend((r, concept_name, task_name) for r in concept_results)
            print(f"  {concept_name:<20s} ({'+'.join(region_names)}): {len(concept_results)}/{N_PER_ATTRIBUTE} "
                  f"samples ({len(candidates)} concept-positive candidates total)", flush=True)

    return results


def save_and_report(version_name: str, out_filename: str, results: list[tuple]):
    out_path = RESULTS_DIR / out_filename
    with open(out_path, "w", newline="") as f:
        base_fields = list(vars(results[0][0]).keys())
        writer = csv.DictWriter(f, fieldnames=base_fields + ["concept_name", "target_task"])
        writer.writeheader()
        for result, concept_name, task_name in results:
            row = vars(result)
            row.update(concept_name=concept_name, target_task=task_name)
            writer.writerow(row)
    print(f"\n[{version_name}] {len(results)} total faithfulness records saved to {out_path}", flush=True)

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for _r, concept_name, task_name in results:
        counts[(concept_name, task_name)] += 1
    n_below_threshold = sum(1 for n in counts.values() if n < MIN_SAMPLES_PER_PAIR)
    n_below_target = sum(1 for n in counts.values() if n < N_PER_ATTRIBUTE)
    print(f"[{version_name}] {len(counts)}/{len(TASKS) * len(GROUNDABLE_CONCEPTS)} (concept, task) pairs populated, "
          f"{n_below_threshold} below min_samples_per_pair, {n_below_target} below the "
          f"N_PER_ATTRIBUTE={N_PER_ATTRIBUTE} target.", flush=True)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading CelebAMask-HQ <-> official-CelebA mapping + attribute labels...", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")

    hq_to_orig = {}
    with open(CELEBA_HQ_ROOT / "CelebA-HQ-to-CelebA-mapping.txt") as f:
        next(f)
        for line in f:
            parts = line.split()
            hq_to_orig[int(parts[0])] = parts[2]

    partition = {}
    with open(CELEBA_ROOT / "list_eval_partition.txt") as f:
        for line in f:
            fname, part = line.split()
            partition[fname] = part

    # Version 1: genuinely non-overlapping with official train (5,817 images).
    non_overlapping = [i for i in image_paths_by_idx if partition.get(hq_to_orig[i]) in ("1", "2")]
    print(f"non_overlapping candidate set: {len(non_overlapping)} HQ images "
          f"(in official val+test, never in official train)", flush=True)

    # Version 2: the ORIGINAL HQ-val split every other ground-truth CSV in
    # this track used (4,500 images) -- confirmed 79.7% now leaked into
    # this classifier's own official-train set, kept anyway for direct
    # comparability with every prior result.
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]
    _, previously_used = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices)
    n_leaked = sum(1 for i in previously_used if partition.get(hq_to_orig[i]) == "0")
    print(f"previously_used candidate set: {len(previously_used)} HQ images "
          f"(the ORIGINAL HQ-val split -- {n_leaked}/{len(previously_used)} "
          f"({100 * n_leaked / len(previously_used):.1f}%) are inside this classifier's OWN "
          f"official-train set -- known leakage, kept for comparability)", flush=True)

    print(f"{len(GROUNDABLE_CONCEPTS)} groundable concepts to score against {TASKS}.", flush=True)
    print(f"BACKBONE_NAME={BACKBONE_NAME}  MODEL_SUFFIX={MODEL_SUFFIX!r}", flush=True)
    spec = BACKBONES[BACKBONE_NAME]
    native_model = spec.load_native()
    preprocess = spec.preprocess

    results_non_overlap = generate_ground_truth(
        "non_overlapping", non_overlapping, image_paths_by_idx, attr_names, attr_labels_by_file,
        native_model, preprocess, seed_offset=0,
    )
    save_and_report("non_overlapping", f"celeba_official_train{MODEL_SUFFIX}_faithfulness_non_overlapping.csv", results_non_overlap)

    results_previously_used = generate_ground_truth(
        "previously_used", previously_used, image_paths_by_idx, attr_names, attr_labels_by_file,
        native_model, preprocess, seed_offset=500_000_000,
    )
    save_and_report("previously_used", f"celeba_official_train{MODEL_SUFFIX}_faithfulness_previously_used.csv", results_previously_used)


if __name__ == "__main__":
    main()
