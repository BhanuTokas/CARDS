"""Phase 6 (TCAV half) of the CelebA plan: TCAV against the native
celeba_attractive_young model, for the 8-concept pilot against BOTH
target tasks (Attractive, Young).

Structural template: scripts/cub/run/run_tcav_cub_official_attributes.py
(the whole-image, real-attribute-label concept definition, NOT the
cropped-region version scripts/cub/run/run_tcav_cub.py's 14-pair
validation slice used) -- CelebA's attributes are genuinely per-IMAGE
binary labels already (unlike CUB, which only had species-level
attribute labels and had to define "positive" at the species level), so
concept positives here are simply whole training images where the
concept's own attribute is literally True. This is a cleaner match to
CARDS' own attribute-specific text queries than the crop-based PCBM bank
is (build_celeba_pilot_concept_bank.py's crops are region-generic, not
attribute-specific -- see that script's own docstring); TCAV's target
concepts don't inherit that asymmetry here.

Only 2 possible target classes total (not 200 species), so there's no
CUB-style "N_TARGETS_PER_ATTRIBUTE species sampled from the positive
pool" step -- every concept is tested against BOTH tasks, always. The
task-positive VAL population Phase 4/5 also drew from (2,582 Attractive-
positive, 3,505 Young-positive) is far larger than a typical CUB
species' own test split (~29 images/species on average) and would make
each interpret() call trivially heavy at full size, so a fixed
N_VAL_SAMPLES-image random subsample per task stands in for "that
species' own test split" -- same order of magnitude as CUB's typical
per-species group size, not the full population.

target=1 (Attractive) or target=3 (Young) is passed straight to
tcav.interpret() as a raw output-index into the native model's 4-way
head -- no task-specific adapter class needed here (unlike Phase 4's
CelebaTaskAdapter): captum only needs a valid index into whatever the
model returns, exactly the same way CUB's target=species_idx indexed
straight into its own 200-way output.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from captum.concept import TCAV, Concept
from concepts.concept_utils import ListDataset

from cards.data.celeba import load_celebamask_hq_image_paths, split_celebamask_hq
from cards.data.celeba_attributes import (
    PILOT_CONCEPTS,
    TARGET_CLASSES,
    load_attribute_labels,
    load_attribute_names,
)
from cards.models.backbones import BACKBONES

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
SEED = 42
DEVICE = "cpu"  # captum's Concept.data_iter never moves batches to CUDA -- same rationale as every prior TCAV script here
N_RANDOM = 6
N_CONTROL = 6
N_PER_RANDOM_SET = 25
N_CONCEPT_EXEMPLARS = 40
N_VAL_SAMPLES = 40  # stands in for "that species' own test split" -- same order of magnitude as CUB's ~29/species average

# task name -> index of that task's positive-class logit in the native
# model's 4-way head, passed straight to tcav.interpret()'s `target`.
TASK_POSITIVE_LOGIT_INDEX: dict[str, int] = {"Attractive": 1, "Young": 3}


def make_concept(concept_id: int, name: str, image_paths: list[Path], preprocess, batch_size: int = 32) -> Concept:
    ds = ListDataset([str(p) for p in image_paths], preprocess=preprocess)
    return Concept(id=concept_id, name=name, data_iter=DataLoader(ds, batch_size=batch_size, shuffle=False))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()
    preprocess = spec.preprocess

    tcav = TCAV(model=native_model, layers=[spec.hook_layer], save_path=str(RESULTS_DIR / "tcav_celeba_pilot_cav_cache"))

    print("Loading CelebAMask-HQ metadata...", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]
    train_hq, val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices)

    rng_py = random.Random(SEED)

    print("Building random/control concept pools from CelebAMask-HQ's own train split...", flush=True)
    all_train_paths = [image_paths_by_idx[i] for i in train_hq]
    rng_py.shuffle(all_train_paths)
    concept_id = 0
    idx = 0
    random_concepts = []
    for i in range(N_RANDOM):
        chunk = all_train_paths[idx : idx + N_PER_RANDOM_SET]
        idx += N_PER_RANDOM_SET
        random_concepts.append(make_concept(concept_id, f"random_{i}", chunk, preprocess))
        concept_id += 1
    control_concepts = []
    for i in range(N_CONTROL):
        chunk = all_train_paths[idx : idx + N_PER_RANDOM_SET]
        idx += N_PER_RANDOM_SET
        control_concepts.append(make_concept(concept_id, f"control_{i}", chunk, preprocess))
        concept_id += 1
    control_sets = [[control_concepts[2 * i], control_concepts[2 * i + 1]] for i in range(N_CONTROL // 2)]
    print(f"{N_RANDOM} random + {N_CONTROL} control pools built, {idx} train images consumed.", flush=True)

    val_by_task: dict[str, list[int]] = {}
    for task_name in TARGET_CLASSES:
        task_attr_idx = attr_names.index(task_name)
        positive_val = [i for i in val_hq if attr_labels_by_file[f"{i}.jpg"][task_attr_idx]]
        rng_py.shuffle(positive_val)
        val_by_task[task_name] = positive_val[:N_VAL_SAMPLES]

    results = []
    for concept_name in PILOT_CONCEPTS:
        concept_attr_idx = attr_names.index(concept_name)
        pos_train_paths = [
            image_paths_by_idx[i] for i in train_hq if attr_labels_by_file[f"{i}.jpg"][concept_attr_idx]
        ]
        rng_py.shuffle(pos_train_paths)
        target_concept = make_concept(concept_id, concept_name, pos_train_paths[:N_CONCEPT_EXEMPLARS], preprocess)
        concept_id += 1

        for task_name in TARGET_CLASSES:
            target_idx = TASK_POSITIVE_LOGIT_INDEX[task_name]
            val_paths = [image_paths_by_idx[i] for i in val_by_task[task_name]]
            inputs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in val_paths]).to(DEVICE)

            experimental_sets = [[target_concept, rc] for rc in random_concepts]
            scores = tcav.interpret(inputs=inputs, experimental_sets=experimental_sets, target=target_idx)
            target_sign_counts = [
                scores[f"{target_concept.id}-{rc.id}"][spec.hook_layer]["sign_count"][0].item() for rc in random_concepts
            ]
            target_magnitudes = [
                scores[f"{target_concept.id}-{rc.id}"][spec.hook_layer]["magnitude"][0].item() for rc in random_concepts
            ]

            control_scores = tcav.interpret(inputs=inputs, experimental_sets=control_sets, target=target_idx)
            control_sign_counts = [
                control_scores[f"{a.id}-{b.id}"][spec.hook_layer]["sign_count"][0].item() for a, b in control_sets
            ]

            t_stat, p_value = ttest_ind(target_sign_counts, control_sign_counts)
            mean_sign_count = float(np.mean(target_sign_counts))
            mean_magnitude = float(np.mean(target_magnitudes))
            mean_control = float(np.mean(control_sign_counts))

            results.append(
                {
                    "concept_name": concept_name,
                    "target_task": task_name,
                    "n_val_images": len(val_paths),
                    "mean_sign_count": mean_sign_count,
                    "mean_control_sign_count": mean_control,
                    "mean_magnitude": mean_magnitude,
                    "t_stat": t_stat,
                    "p_value": p_value,
                }
            )
            print(f"{concept_name:<20s} / {task_name:<12s}: sign_count={mean_sign_count:.3f} "
                  f"(null={mean_control:.3f})  magnitude={mean_magnitude:.4f}  p={p_value:.4g}", flush=True)

    with open(RESULTS_DIR / "tcav_celeba_pilot_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{len(results)} total (concept, task) TCAV pairs saved to results/tcav_celeba_pilot_scores.csv")


if __name__ == "__main__":
    main()
