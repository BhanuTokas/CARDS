"""Instrumented timing benchmark for TCAV, reusing run_tcav_celeba_
official_train.py's exact pipeline (whole-image concept definition,
official-train random/control/concept pools, captum) at full production
scale. Reconstructed as a saved, HPC-runnable script, matching the
original ad hoc benchmark's phase breakdown: pool_build (random/control
pool construction) and cav_fit_and_interpret_all_concepts_x_tasks
(captum's CAV fit + interpret, called fresh per concept x task since
captum caches nothing across calls -- this is TCAV's entire cost, 100%
black-box-model-dependent, no separate encoder step at all).

DEVICE is forced to "cpu" throughout, matching every other TCAV script in
this codebase: captum's Concept.data_iter never moves batches to CUDA,
so running on GPU here would just desync device placement, not speed
anything up.
"""

from __future__ import annotations

import os
import random
import sys
import time
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
DEVICE = "cpu"
SEED = 42
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
    result: dict[str, np.ndarray] = {}
    for line in Path(path).read_text().splitlines()[2:]:
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
    print(f"BACKBONE_NAME={BACKBONE_NAME}  DEVICE={DEVICE}  CELEBA_ROOT={CELEBA_ROOT}", flush=True)
    timings: dict[str, float] = {}

    spec = BACKBONES[BACKBONE_NAME]
    native_model = spec.load_native().to(DEVICE).eval()
    preprocess = spec.preprocess
    tcav = TCAV(model=native_model, layers=[spec.hook_layer],
                save_path=str(RESULTS_DIR / "tcav_cost_bench_cav_cache"))

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

    t0 = time.perf_counter()
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
    timings["pool_build"] = time.perf_counter() - t0
    print(f"{N_RANDOM} random + {N_CONTROL} control pools built (pool_build: {timings['pool_build']:.4f}s)", flush=True)

    val_by_task: dict[str, list[str]] = {}
    for task_name in TASKS:
        task_attr_idx = attr_names.index(task_name)
        positive_train = [f for f in train_files if attr_labels[f][task_attr_idx]]
        rng_py.shuffle(positive_train)
        val_by_task[task_name] = positive_train[:N_VAL_SAMPLES]

    t0 = time.perf_counter()
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

            control_scores = tcav.interpret(inputs=inputs, experimental_sets=control_sets, target=target_idx)
            control_sign_counts = [
                control_scores[f"{a.id}-{b.id}"][spec.hook_layer]["sign_count"][0].item() for a, b in control_sets
            ]

            t_stat, p_value = ttest_ind(target_sign_counts, control_sign_counts)
            results.append((concept_name, task_name, t_stat, p_value))
        print(f"  {concept_name}: done both tasks", flush=True)
    timings["cav_fit_and_interpret_all_concepts_x_tasks"] = time.perf_counter() - t0
    print(f"cav_fit_and_interpret_all_concepts_x_tasks: {timings['cav_fit_and_interpret_all_concepts_x_tasks']:.2f}s", flush=True)

    timings["TOTAL"] = sum(timings.values())

    print("\n=== Summary: TCAV ===", flush=True)
    for phase, secs in timings.items():
        print(f"  {phase:<40s} {secs:>10.2f}s")

    out_path = RESULTS_DIR / "computational_cost_benchmark_tcav_full_scale.csv"
    with open(out_path, "w") as f:
        f.write("method,phase,seconds\n")
        for phase, secs in timings.items():
            f.write(f"TCAV,{phase},{secs}\n")
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
