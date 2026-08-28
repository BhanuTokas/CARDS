"""Phase 2 (notes/pcbm_correlation_investigation.md, CARDS vs. TCAV vs.
PCBM plan): TCAV against the native resnet18 model directly (not PCBM's
own surrogate stack -- see the plan's surrogate-modeling correction),
hooking `BACKBONES["resnet18"].hook_layer` ("layer4", the last nonlinear
block before the native classification head).

First validation slice: 6 (concept, class) pairs mixing obviously-strong
and obviously-irrelevant relationships, using Broden concepts confirmed
present on disk in the existing (unmodified, v20-flawed-baseline)
Datasets/broden_concepts/ bank, and ImageNet classes from the Phase 1
slice. Not the full concept x class matrix -- a real discriminating
test before scaling up.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from build_imagenet_slice import TARGET_CLASSES  # noqa: E402
from cards.models.backbones import BACKBONES  # noqa: E402
from concepts.concept_utils import ListDataset  # noqa: E402

from captum.concept import TCAV, Concept  # noqa: E402

BRODEN_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\broden_concepts")
IMAGENET_SLICE_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\imagenet_slice")
RESULTS_DIR = Path("results")
SEED = 42
# CPU, not "cuda": captum pulls concept exemplar batches straight from each
# Concept's own data_iter (ListDataset, post_hoc_cbm's external class) with
# no device hook available -- those never get moved to CUDA, so a CUDA
# model would hit a device mismatch against them. CPU keeps everything
# consistent without touching post_hoc_cbm's own Dataset class; the small
# batches here (~50 images/concept) make this a fine tradeoff for this
# validation-scale run.
DEVICE = "cpu"
N_RANDOM = 10
N_CONTROL = 10
N_PER_RANDOM_SET = 50
N_CONCEPT_EXEMPLARS = 50

CLASS_NAMES = [name for _, _, name in TARGET_CLASSES]
NATIVE_LABEL_IDX = {name: idx for idx, _, name in TARGET_CLASSES}  # class_name -> index into the native model's 1000-way output

# (broden_concept, target_class, expected_relationship) -- confirmed
# present on disk in Datasets/broden_concepts/ (checked directly; several
# NetDissect concepts like "table"/"vase"/"clock" are either not in this
# repackaged 170-concept subset or on the drop list, so this list is
# scoped to what's actually available, not the full original taxonomy).
TEST_PAIRS = [
    ("car", "sports_car", "positive"),
    ("cat", "tabby_cat", "positive"),
    ("dog", "golden_retriever", "positive"),
    ("chair", "rocking_chair", "positive"),
    ("car", "siamese_cat", "negative"),
    ("bottle", "golden_retriever", "negative"),
    # Added to cover the full set of each concept's own matching class(es)
    # (CONCEPT_MATCHING_CLASSES in run_broden_faithfulness.py / v30-v31),
    # so TCAV can finally be scored against the same faithfulness ground
    # truth CARDS/PCBM were (v31) -- reuses the CAV cache these first 6
    # pairs already built (same 5 concepts, same random/control pools),
    # so only new target-class directional derivatives get computed, not
    # a refit from scratch.
    ("car", "convertible", "matching"),
    ("cat", "siamese_cat", "matching"),
    ("cat", "egyptian_cat", "matching"),
    ("dog", "labrador_retriever", "matching"),
    ("chair", "folding_chair", "matching"),
    ("bottle", "water_bottle", "matching"),
]


def make_concept(concept_id: int, name: str, image_paths: list[Path], preprocess, batch_size: int = 32) -> Concept:
    ds = ListDataset([str(p) for p in image_paths], preprocess=preprocess)
    return Concept(id=concept_id, name=name, data_iter=DataLoader(ds, batch_size=batch_size, shuffle=False))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    spec = BACKBONES["resnet18"]
    native_model = spec.load_native().to(DEVICE).eval()
    preprocess = spec.preprocess

    tcav = TCAV(model=native_model, layers=[spec.hook_layer], save_path=str(RESULTS_DIR / "tcav_cav_cache"))

    print("Building random/control concept pools from the ImageNet train slice...", flush=True)
    all_train_paths: list[Path] = []
    for name in CLASS_NAMES:
        all_train_paths.extend(sorted((IMAGENET_SLICE_ROOT / "train" / name).glob("*.jpg")))
    rng = random.Random(SEED)
    rng.shuffle(all_train_paths)

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
    print(f"{N_RANDOM} random + {N_CONTROL} control concept pools built, {idx} train images consumed "
          f"(of {len(all_train_paths)} available).", flush=True)

    control_sets = [[control_concepts[2 * i], control_concepts[2 * i + 1]] for i in range(N_CONTROL // 2)]

    results = []
    target_concept_cache: dict[str, Concept] = {}  # broden_concept name -> Concept, reused across pairs so
    # captum's CAV cache actually hits for a repeated concept (e.g. "car"
    # tested against both sports_car and convertible) instead of
    # re-extracting activations under a fresh id each time.
    for broden_concept, target_class, expected in TEST_PAIRS:
        print(f"\n=== {broden_concept} -> {target_class} (expected: {expected}) ===", flush=True)
        if broden_concept not in target_concept_cache:
            pos_paths = sorted((BRODEN_ROOT / broden_concept / "positives").glob("*"))[:N_CONCEPT_EXEMPLARS]
            target_concept_cache[broden_concept] = make_concept(concept_id, broden_concept, pos_paths, preprocess)
            concept_id += 1
        target_concept = target_concept_cache[broden_concept]

        val_paths = sorted((IMAGENET_SLICE_ROOT / "val" / target_class).glob("*.jpg"))
        inputs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in val_paths]).to(DEVICE)
        target_idx = NATIVE_LABEL_IDX[target_class]

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

        print(f"target sign_count: mean={mean_sign_count:.4f} (values: {[round(v, 3) for v in target_sign_counts]})")
        print(f"control (null) sign_count: mean={mean_control:.4f} (values: {[round(v, 3) for v in control_sign_counts]})")
        print(f"magnitude: mean={mean_magnitude:.4f}")
        print(f"t={t_stat:.4f}, p={p_value:.4g}")

        results.append(
            {
                "broden_concept": broden_concept,
                "target_class": target_class,
                "expected": expected,
                "mean_sign_count": mean_sign_count,
                "mean_control_sign_count": mean_control,
                "mean_magnitude": mean_magnitude,
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )

    import csv

    with open(RESULTS_DIR / "tcav_matching_pairs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n=== summary ===")
    for r in results:
        print(
            f"{r['broden_concept']:>10s} -> {r['target_class']:<20s} [{r['expected']:>8s}]  "
            f"sign_count={r['mean_sign_count']:.3f} (null={r['mean_control_sign_count']:.3f})  "
            f"p={r['p_value']:.4g}"
        )
    print("\nSaved results/tcav_matching_pairs.csv")


if __name__ == "__main__":
    main()
