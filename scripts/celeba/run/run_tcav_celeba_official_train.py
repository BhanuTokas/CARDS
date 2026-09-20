"""TCAV on the official-train classifier, both tasks (Attractive,
Male -- Young dropped), prompted directly ("Do we also have results of
PCBM and TCAV on this version of ground truth?"). Mirrors `run_tcav_
celeba_male.py`'s own design (whole-image real-attribute-label concept
definition, random/control pool construction, N_VAL_SAMPLES-image
subsample) but sources ALL of its own images (random/control pools,
per-concept exemplars, val samples) from standard CelebA's OFFICIAL
TRAIN partition (162,770 images, `list_attr_celeba.txt` labels) instead
of CelebAMask-HQ's train_hq -- matching what this specific classifier
actually trained on, and avoiding any CelebAMask-HQ dependency this
official-train arc otherwise doesn't need for TCAV specifically (the
ground truth is the only piece that needs CelebAMask-HQ's masks; TCAV's
own concept definition here is whole-image, no masks required).

**Generalized to any official-train backbone** (`CARDS_BACKBONE_NAME`
env var, default `celeba_official_train_attractive_male`), prompted
directly ("The full pipeline with PCBM variants" for ViT-B/16 and
ConvNeXt-Tiny) -- already hooked `spec.hook_layer` dynamically rather
than a hardcoded "layer4", so this only needed the backbone-name lookup
itself parameterized, plus an output-filename suffix.
"""

from __future__ import annotations

import csv
import os
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

from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.models.backbones import BACKBONES

CELEBA_ROOT = Path(os.environ.get("CELEBA_ROOT", r"C:\Users\btokas\Projects\Datasets\CelebA\celeba"))
RESULTS_DIR = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
BACKBONE_NAME = os.environ.get("CARDS_BACKBONE_NAME", "celeba_official_train_attractive_male")
MODEL_SUFFIX = BACKBONE_NAME.replace("celeba_official_train_attractive_male", "")
SEED = 42
DEVICE = "cpu"  # captum's Concept.data_iter never moves batches to CUDA
N_RANDOM = 6
N_CONTROL = 6
N_PER_RANDOM_SET = 25
N_CONCEPT_EXEMPLARS = 40
N_VAL_SAMPLES = 40
TASKS = ["Attractive", "Male"]
TASK_POSITIVE_LOGIT_INDEX = {"Attractive": 1, "Male": 3}


def load_attr_names(path: Path) -> list[str]:
    return Path(path).read_text().splitlines()[1].split()


def load_attr_labels(path: Path) -> dict[str, np.ndarray]:
    lines = Path(path).read_text().splitlines()
    result: dict[str, np.ndarray] = {}
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split()
        result[parts[0]] = np.array([v == "1" for v in parts[1:]], dtype=bool)
    return result


def make_concept(concept_id: int, name: str, filenames: list[str], preprocess, batch_size: int = 32) -> Concept:
    paths = [str(CELEBA_ROOT / "img_align_celeba" / f) for f in filenames]
    ds = ListDataset(paths, preprocess=preprocess)
    return Concept(id=concept_id, name=name, data_iter=DataLoader(ds, batch_size=batch_size, shuffle=False))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"BACKBONE_NAME={BACKBONE_NAME}  MODEL_SUFFIX={MODEL_SUFFIX!r}", flush=True)
    spec = BACKBONES[BACKBONE_NAME]
    native_model = spec.load_native().to(DEVICE).eval()
    preprocess = spec.preprocess

    tcav = TCAV(model=native_model, layers=[spec.hook_layer],
                save_path=str(RESULTS_DIR / f"tcav_celeba_official_train{MODEL_SUFFIX}_cav_cache"))

    print("Loading official CelebA metadata...", flush=True)
    attr_names = load_attr_names(CELEBA_ROOT / "list_attr_celeba.txt")
    attr_labels = load_attr_labels(CELEBA_ROOT / "list_attr_celeba.txt")
    partition = {}
    with open(CELEBA_ROOT / "list_eval_partition.txt") as f:
        for line in f:
            fname, part = line.split()
            partition[fname] = part
    train_files = sorted(f for f, p in partition.items() if p == "0")
    print(f"{len(train_files)} official-train images, {len(GROUNDABLE_CONCEPTS)} groundable concepts.", flush=True)

    rng_py = random.Random(SEED)

    print("Building random/control concept pools from official train...", flush=True)
    shuffled = train_files.copy()
    rng_py.shuffle(shuffled)
    concept_id = 0
    idx = 0
    random_concepts = []
    for i in range(N_RANDOM):
        chunk = shuffled[idx : idx + N_PER_RANDOM_SET]
        idx += N_PER_RANDOM_SET
        random_concepts.append(make_concept(concept_id, f"random_{i}", chunk, preprocess))
        concept_id += 1
    control_concepts = []
    for i in range(N_CONTROL):
        chunk = shuffled[idx : idx + N_PER_RANDOM_SET]
        idx += N_PER_RANDOM_SET
        control_concepts.append(make_concept(concept_id, f"control_{i}", chunk, preprocess))
        concept_id += 1
    control_sets = [[control_concepts[2 * i], control_concepts[2 * i + 1]] for i in range(N_CONTROL // 2)]
    print(f"{N_RANDOM} random + {N_CONTROL} control pools built, {idx} train images consumed.", flush=True)

    val_by_task: dict[str, list[str]] = {}
    for task_name in TASKS:
        task_attr_idx = attr_names.index(task_name)
        positive_train = [f for f in train_files if attr_labels[f][task_attr_idx]]
        rng_py.shuffle(positive_train)
        val_by_task[task_name] = positive_train[:N_VAL_SAMPLES]

    results = []
    for concept_name in GROUNDABLE_CONCEPTS:
        concept_attr_idx = attr_names.index(concept_name)
        pos_files = [f for f in train_files if attr_labels[f][concept_attr_idx]]
        rng_py.shuffle(pos_files)
        target_concept = make_concept(concept_id, concept_name, pos_files[:N_CONCEPT_EXEMPLARS], preprocess)
        concept_id += 1

        for task_name in TASKS:
            target_idx = TASK_POSITIVE_LOGIT_INDEX[task_name]
            val_files = val_by_task[task_name]
            inputs = torch.stack([
                preprocess(Image.open(CELEBA_ROOT / "img_align_celeba" / f).convert("RGB")) for f in val_files
            ]).to(DEVICE)

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

            results.append({
                "concept_name": concept_name, "target_task": task_name, "n_val_images": len(val_files),
                "mean_sign_count": mean_sign_count, "mean_control_sign_count": mean_control,
                "mean_magnitude": mean_magnitude, "t_stat": t_stat, "p_value": p_value,
            })
            print(f"{concept_name:<20s} / {task_name:<12s}: sign_count={mean_sign_count:.3f} "
                  f"(null={mean_control:.3f})  magnitude={mean_magnitude:.4f}  p={p_value:.4g}", flush=True)

    out_path = RESULTS_DIR / f"tcav_celeba_official_train{MODEL_SUFFIX}_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{len(results)} total (concept, task) TCAV rows saved to {out_path}")


if __name__ == "__main__":
    main()
