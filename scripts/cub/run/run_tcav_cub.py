"""CUB-track analogue of run_tcav.py: TCAV against the native resnet18_cub
model directly, hooking BACKBONES["resnet18_cub"].hook_layer
("features.stage4"), using the new real-part-crop concept bank
(scripts/build_cub_part_concept_bank.py's output) as concept exemplars.

First validation slice: (part, species) pairs chosen from CUB class names
that are LITERALLY named for a body part (e.g. "Red_eyed_Vireo",
"Scissor_tailed_Flycatcher") as clean "matching" positives, crossed with
the same species against an unrelated part as negative controls -- a real
discriminating test (does TCAV find beak important for a species named
for its eye? it shouldn't) before scaling to the full concept x class
matrix, mirroring the ImageNet track's Phase 2 approach.
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

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from cards.data.cub_parts import load_images_txt  # noqa: E402
from cards.models.backbones import BACKBONES  # noqa: E402
from concepts.concept_utils import ListDataset  # noqa: E402

from captum.concept import TCAV, Concept  # noqa: E402

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
CONCEPT_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\cub_part_concepts")
RESULTS_DIR = Path("results")
SEED = 42
DEVICE = "cpu"  # same rationale as run_tcav.py: captum's Concept.data_iter never moves batches to CUDA
N_RANDOM = 10
N_CONTROL = 10
N_PER_RANDOM_SET = 30  # smaller than the ImageNet track's 50 -- CUB train images are already tightly cropped-to-bird, less need for large pools
N_CONCEPT_EXEMPLARS = 50

# (part, species, expected) -- species chosen because their common name is
# literally about that body part (a real, checkable ground truth, not a
# guess), confirmed present in classes.txt.
TEST_PAIRS = [
    ("beak", "Groove_billed_Ani", "matching"),
    ("beak", "Pied_billed_Grebe", "matching"),
    ("left_wing", "Red_winged_Blackbird", "matching"),
    ("right_wing", "Red_winged_Blackbird", "matching"),
    ("tail", "Scissor_tailed_Flycatcher", "matching"),
    ("left_eye", "Red_eyed_Vireo", "matching"),
    ("right_eye", "Red_eyed_Vireo", "matching"),
    ("left_leg", "Black_footed_Albatross", "matching"),
    ("right_leg", "Red_legged_Kittiwake", "matching"),
    # negative controls: same concepts, species named for a DIFFERENT part
    ("beak", "Scissor_tailed_Flycatcher", "negative"),
    ("tail", "Red_eyed_Vireo", "negative"),
    ("left_eye", "Red_winged_Blackbird", "negative"),
    ("left_wing", "Black_footed_Albatross", "negative"),
    ("left_leg", "Groove_billed_Ani", "negative"),
]


def make_concept(concept_id: int, name: str, image_paths: list[Path], preprocess, batch_size: int = 32) -> Concept:
    ds = ListDataset([str(p) for p in image_paths], preprocess=preprocess)
    return Concept(id=concept_id, name=name, data_iter=DataLoader(ds, batch_size=batch_size, shuffle=False))


def load_classes(cub_root: Path) -> dict[str, int]:
    """species name (e.g. "Red_eyed_Vireo") -> 0-indexed native output idx."""
    result = {}
    for line in (cub_root / "classes.txt").read_text().splitlines():
        class_id, raw_name = line.split(maxsplit=1)
        name = raw_name.split(".", 1)[1] if "." in raw_name else raw_name
        result[name] = int(class_id) - 1
    return result


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    spec = BACKBONES["resnet18_cub"]
    native_model = spec.load_native().to(DEVICE).eval()
    preprocess = spec.preprocess

    tcav = TCAV(model=native_model, layers=[spec.hook_layer], save_path=str(RESULTS_DIR / "tcav_cub_cav_cache"))

    image_paths = load_images_txt(CUB_ROOT)
    class_to_idx = load_classes(CUB_ROOT)
    class_labels = {}
    for line in (CUB_ROOT / "image_class_labels.txt").read_text().splitlines():
        image_id, class_id = line.split()
        class_labels[image_id] = int(class_id)
    train_ids = {
        line.split()[0]
        for line in (CUB_ROOT / "train_test_split.txt").read_text().splitlines()
        if line.split()[1] == "1"
    }
    test_ids = {
        line.split()[0]
        for line in (CUB_ROOT / "train_test_split.txt").read_text().splitlines()
        if line.split()[1] == "0"
    }

    print("Building random/control concept pools from CUB's own train split...", flush=True)
    all_train_paths = [image_paths[i] for i in train_ids]
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

    # target_class val images, indexed by species name (test-split only)
    ids_by_class: dict[str, list[str]] = {}
    for image_id in test_ids:
        cid = class_labels[image_id]
        ids_by_class.setdefault(cid, []).append(image_id)

    results = []
    target_concept_cache: dict[str, Concept] = {}
    for part_name, species, expected in TEST_PAIRS:
        print(f"\n=== {part_name} -> {species} (expected: {expected}) ===", flush=True)
        if part_name not in target_concept_cache:
            pos_paths = sorted((CONCEPT_ROOT / part_name / "positives").glob("*.jpg"))[:N_CONCEPT_EXEMPLARS]
            target_concept_cache[part_name] = make_concept(concept_id, part_name, pos_paths, preprocess)
            concept_id += 1
        target_concept = target_concept_cache[part_name]

        target_idx = class_to_idx[species]
        val_image_ids = ids_by_class[target_idx + 1]  # class_labels are 1-indexed class_id
        val_paths = [image_paths[i] for i in val_image_ids]
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

        print(f"n_val_images={len(val_paths)}")
        print(f"target sign_count: mean={mean_sign_count:.4f} (values: {[round(v, 3) for v in target_sign_counts]})")
        print(f"control (null) sign_count: mean={mean_control:.4f} (values: {[round(v, 3) for v in control_sign_counts]})")
        print(f"magnitude: mean={mean_magnitude:.4f}")
        print(f"t={t_stat:.4f}, p={p_value:.4g}")

        results.append(
            {
                "part": part_name,
                "species": species,
                "expected": expected,
                "n_val_images": len(val_paths),
                "mean_sign_count": mean_sign_count,
                "mean_control_sign_count": mean_control,
                "mean_magnitude": mean_magnitude,
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )

    with open(RESULTS_DIR / "tcav_cub_matching_pairs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n=== summary ===")
    for r in results:
        print(
            f"{r['part']:>10s} -> {r['species']:<25s} [{r['expected']:>8s}]  "
            f"sign_count={r['mean_sign_count']:.3f} (null={r['mean_control_sign_count']:.3f})  "
            f"p={r['p_value']:.4g}"
        )
    print("\nSaved results/tcav_cub_matching_pairs.csv")


if __name__ == "__main__":
    main()
