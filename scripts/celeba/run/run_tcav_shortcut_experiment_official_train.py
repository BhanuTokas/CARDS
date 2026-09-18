"""Official-train counterpart of `run_tcav_shortcut_experiment.py` --
"conventional" TCAV (native ResNet18 `layer4` activation-space
directional derivatives, matching this track's original TCAV
convention throughout) against the 22 official-train shortcut
checkpoints (2 tasks x 11 rates) instead of the original 4 HQ-trained
ones. Confirmed directly ("Wait, TCAV doesn't need those [CLIP-RN50/
SigLIP] variants. Only PCBM has these variants.") -- TCAV stays this
one conventional form only; see `run_pcbm_shortcut_experiment_official_
train.py` / `run_pcbm_clip_concepts_shortcut_experiment_official_train.py`
for the 3-variant PCBM comparison.

Uses ONLY official CelebA (list_attr_celeba.txt, list_eval_partition.txt)
-- no CelebA-HQ dependency, unlike the original HQ-trained shortcut
experiment's own TCAV script. Confirmed directly ("I do not have the
CELEBA_HQ downloaded there"): random/control pools and each concept's
target exemplars are now static, classifier- AND task-independent
official CelebA TRAIN (partition==0) images, attribute-filtered the same
way HQ-train images were -- built ONCE, reused across both tasks and all
11 rates. The official-val SCORING sample is plain partition==1 (no HQ-
leakage exclusion needed: these classifiers train on partition==0 only,
already disjoint from partition==1 by construction), task-specific (real
Attractive=1 images for the Attractive checkpoints, real Male=1 images
for the Male checkpoints) -- built once per task, reused across that
task's 11 rates.

target=1 (each single 2-way head's positive-class logit).
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
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from captum.concept import TCAV, Concept
from concepts.concept_utils import ListDataset

from cards.data.celeba_attributes import (
    GROUNDABLE_CONCEPTS,
    load_attribute_labels,
    load_attribute_names,
)
from cards.models.backbones import BACKBONES

CELEBA_ROOT = Path(os.environ.get("CELEBA_ROOT", r"C:\Users\btokas\Projects\Datasets\CelebA\celeba"))
RESULTS_DIR = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
CKPT_DIR = Path(os.environ.get("CARDS_CKPT_DIR", "trained_models_new/celeba"))
SEED = 42
DEVICE = "cpu"  # captum's Concept.data_iter never moves batches to CUDA
N_RANDOM = 6
N_CONTROL = 6
N_PER_RANDOM_SET = 25
N_CONCEPT_EXEMPLARS = 40
N_VAL_SAMPLES = 40
RATES_PCT = list(range(0, 101, 10))
HOOK_LAYER = "layer4"
TARGET_IDX = 1
TASKS = ["Attractive", "Male"]


def build_model(task: str, rate_pct: int) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    ckpt_path = CKPT_DIR / f"resnet18_official_train_{task.lower()}_shortcut_{rate_pct}.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    return model.eval()


def load_official_partition_paths() -> tuple[list[Path], list[Path]]:
    """(train_paths, val_paths) from list_eval_partition.txt (0=train, 1=val)."""
    train_paths, val_paths = [], []
    with open(CELEBA_ROOT / "list_eval_partition.txt") as f:
        for line in f:
            fname, part = line.split()
            if part == "0":
                train_paths.append(CELEBA_ROOT / "img_align_celeba" / fname)
            elif part == "1":
                val_paths.append(CELEBA_ROOT / "img_align_celeba" / fname)
    return train_paths, val_paths


def make_concept(concept_id: int, name: str, image_paths: list, preprocess, batch_size: int = 32) -> Concept:
    ds = ListDataset([str(p) for p in image_paths], preprocess=preprocess)
    return Concept(id=concept_id, name=name, data_iter=DataLoader(ds, batch_size=batch_size, shuffle=False))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    spec = BACKBONES["celeba_attractive_young"]  # preprocess only -- architecture-generic
    preprocess = spec.preprocess

    print("Loading official CelebA metadata (concept-bank fitting resources)...", flush=True)
    train_paths, val_paths_all = load_official_partition_paths()
    attr_names = load_attribute_names(CELEBA_ROOT / "list_attr_celeba.txt")
    attr_labels = load_attribute_labels(CELEBA_ROOT / "list_attr_celeba.txt")
    print(f"{len(train_paths)} official train images, {len(val_paths_all)} official val images.", flush=True)
    print(f"{len(GROUNDABLE_CONCEPTS)} groundable concepts to score.", flush=True)

    rng_py = random.Random(SEED)

    print("Building random/control concept pools (shared across both tasks, all rates)...", flush=True)
    all_train_paths = list(train_paths)
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

    target_concepts = {}
    for concept_name in GROUNDABLE_CONCEPTS:
        concept_attr_idx = attr_names.index(concept_name)
        pos_train_paths = [p for p in train_paths if attr_labels[p.name][concept_attr_idx]]
        rng_py.shuffle(pos_train_paths)
        target_concepts[concept_name] = make_concept(
            concept_id, concept_name, pos_train_paths[:N_CONCEPT_EXEMPLARS], preprocess
        )
        concept_id += 1

    val_inputs_by_task: dict[str, torch.Tensor] = {}
    for task in TASKS:
        print(f"\nBuilding official-val scoring sample for {task} (real {task}=1 images)...", flush=True)
        task_idx = attr_names.index(task)
        positive_official = [p for p in val_paths_all if attr_labels[p.name][task_idx]]
        rng_py.shuffle(positive_official)
        val_paths = positive_official[:N_VAL_SAMPLES]
        print(f"{len(positive_official)} official-val images positive for {task}; using {len(val_paths)}.", flush=True)
        val_inputs_by_task[task] = torch.stack(
            [preprocess(Image.open(p).convert("RGB")) for p in val_paths]
        ).to(DEVICE)

    all_rows = []  # (task, rate_pct, concept_name, mean_sign_count, mean_control_sign_count, mean_magnitude, t_stat, p_value)
    scores_by_task_rate: dict[str, dict[int, dict[str, float]]] = {t: {} for t in TASKS}

    for task in TASKS:
        inputs = val_inputs_by_task[task]
        for rate_pct in RATES_PCT:
            print(f"\n=== task={task} rate={rate_pct}% ===", flush=True)
            model = build_model(task, rate_pct).to(DEVICE)
            tcav = TCAV(model=model, layers=[HOOK_LAYER],
                        save_path=str(RESULTS_DIR / f"tcav_official_train_shortcut_cav_cache_{task.lower()}_{rate_pct}"))

            scores_by_task_rate[task][rate_pct] = {}
            for concept_name in GROUNDABLE_CONCEPTS:
                target_concept = target_concepts[concept_name]

                experimental_sets = [[target_concept, rc] for rc in random_concepts]
                scores = tcav.interpret(inputs=inputs, experimental_sets=experimental_sets, target=TARGET_IDX)
                target_sign_counts = [
                    scores[f"{target_concept.id}-{rc.id}"][HOOK_LAYER]["sign_count"][0].item() for rc in random_concepts
                ]
                target_magnitudes = [
                    scores[f"{target_concept.id}-{rc.id}"][HOOK_LAYER]["magnitude"][0].item() for rc in random_concepts
                ]

                control_scores = tcav.interpret(inputs=inputs, experimental_sets=control_sets, target=TARGET_IDX)
                control_sign_counts = [
                    control_scores[f"{a.id}-{b.id}"][HOOK_LAYER]["sign_count"][0].item() for a, b in control_sets
                ]

                t_stat, p_value = ttest_ind(target_sign_counts, control_sign_counts)
                mean_sign_count = float(np.mean(target_sign_counts))
                mean_magnitude = float(np.mean(target_magnitudes))
                mean_control = float(np.mean(control_sign_counts))

                scores_by_task_rate[task][rate_pct][concept_name] = mean_magnitude
                all_rows.append((task, rate_pct, concept_name, mean_sign_count, mean_control, mean_magnitude, t_stat, p_value))
                print(f"  {concept_name:<20s} sign_count={mean_sign_count:.3f} (null={mean_control:.3f}) "
                      f"magnitude={mean_magnitude:.4f} p={p_value:.4g}", flush=True)

            ranked = sorted(scores_by_task_rate[task][rate_pct].items(), key=lambda kv: -abs(kv[1]))
            print(f"  top-5 by |magnitude|: {[(c, round(s, 4)) for c, s in ranked[:5]]}", flush=True)
            print(f"  mean |magnitude| across all {len(GROUNDABLE_CONCEPTS)} concepts: "
                  f"{np.mean([abs(s) for s in scores_by_task_rate[task][rate_pct].values()]):.4f}", flush=True)

    out_path = RESULTS_DIR / "tcav_celeba_official_train_shortcut_experiment.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "rate_pct", "concept_name", "mean_sign_count", "mean_control_sign_count",
                          "mean_magnitude", "t_stat", "p_value"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")

    print("\n=== mean |magnitude| across all concepts, per task, per rate (the headline decline check) ===")
    for task in TASKS:
        for rate_pct in RATES_PCT:
            mean_abs = np.mean([abs(s) for s in scores_by_task_rate[task][rate_pct].values()])
            print(f"  task={task:<10s} rate={rate_pct:>3d}%: mean |magnitude| = {mean_abs:.4f}")


if __name__ == "__main__":
    main()
