"""Runs TCAV against the FULL v49 class-stratified ground truth's own
(attribute, class) pairs (522, up from the earlier 46-pair targeted run's
coverage) -- superseding run_tcav_cub_targeted_46.py now that results/
cub_attribute_faithfulness.csv has been rebuilt with class-stratified
sampling (every sampled species guaranteed >=3 valid instances by
construction, see run_cub_attribute_faithfulness.py's own docstring).

Same rationale as the 46-pair script: TCAV's own broad/independent run
(run_tcav_cub_official_attributes.py, 435 pairs chosen by TRUE label, not
by what the faithfulness ground truth's own PREDICTED-class distribution
lands on) only coincidentally overlaps the ground truth's pairs (125/522
this time -- confirmed by scoring it directly), and that partial-coverage
subsample already showed a materially different, inflated sign agreement
(81.6%) versus what full targeted coverage is expected to reveal, exactly
the same coverage-bias pattern diagnosed for the original 16/46 overlap.
This run targets all 522 pairs directly instead of hoping for overlap.

Mechanically identical to run_tcav_cub_targeted_46.py (same concept-
exemplar construction, same random/control pools, same SEED/N_RANDOM/
N_CONTROL) -- only the pair-loading source (now naturally 522 pairs,
since load_targeted_pairs() just reads whatever is currently in
cub_attribute_faithfulness.csv) and the output filename differ.
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
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from captum.concept import TCAV, Concept
from concepts.concept_utils import ListDataset

from cards.data.cub_attributes import (
    groundable_attributes,
    load_attribute_names,
    load_class_attributes,
)
from cards.data.cub_parts import load_images_txt
from cards.models.backbones import BACKBONES

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
ATTRIBUTE_NAMES_PATH = CUB_ROOT / "attributes" / "new_attributes.txt"
CLASS_ATTR_DIR = CUB_ROOT / "class_attr_data_10"
RESULTS_DIR = Path("results")
SEED = 42
DEVICE = "cpu"  # same rationale as every prior TCAV script: captum's Concept.data_iter never moves batches to CUDA
N_RANDOM = 6
N_CONTROL = 6
N_PER_RANDOM_SET = 25
N_CONCEPT_EXEMPLARS = 40


def make_concept(concept_id: int, name: str, image_paths: list[Path], preprocess, batch_size: int = 32) -> Concept:
    ds = ListDataset([str(p) for p in image_paths], preprocess=preprocess)
    return Concept(id=concept_id, name=name, data_iter=DataLoader(ds, batch_size=batch_size, shuffle=False))


def load_targeted_pairs() -> list[tuple[int, int]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    with open(RESULTS_DIR / "cub_attribute_faithfulness.csv", newline="") as f:
        for row in csv.DictReader(f):
            grouped[(int(row["concept_number"]), int(row["predicted_class"]))].append(float(row["delta_p"]))
    return sorted(pair for pair, deltas in grouped.items() if len(deltas) >= 3)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    spec = BACKBONES["resnet18_cub"]
    native_model = spec.load_native().to(DEVICE).eval()
    preprocess = spec.preprocess

    tcav = TCAV(model=native_model, layers=[spec.hook_layer], save_path=str(RESULTS_DIR / "tcav_cub_attr_cav_cache"))

    pairs = load_targeted_pairs()
    pairs_by_attr: dict[int, list[int]] = defaultdict(list)
    for attr_idx, class_idx in pairs:
        pairs_by_attr[attr_idx].append(class_idx)
    print(f"{len(pairs)} targeted (attribute, class) pairs, across {len(pairs_by_attr)} distinct attributes.", flush=True)

    print("Loading CUB metadata + official attribute labels...", flush=True)
    image_paths = load_images_txt(CUB_ROOT)
    class_labels = {}
    for line in (CUB_ROOT / "image_class_labels.txt").read_text().splitlines():
        image_id, class_id = line.split()
        class_labels[image_id] = int(class_id)
    train_ids = [
        line.split()[0]
        for line in (CUB_ROOT / "train_test_split.txt").read_text().splitlines()
        if line.split()[1] == "1"
    ]
    test_ids = [
        line.split()[0]
        for line in (CUB_ROOT / "train_test_split.txt").read_text().splitlines()
        if line.split()[1] == "0"
    ]
    train_ids_by_class: dict[int, list[str]] = defaultdict(list)
    for i in train_ids:
        train_ids_by_class[class_labels[i]].append(i)
    test_ids_by_class: dict[int, list[str]] = defaultdict(list)
    for i in test_ids:
        test_ids_by_class[class_labels[i]].append(i)

    attribute_names = load_attribute_names(ATTRIBUTE_NAMES_PATH)
    groundable = groundable_attributes(attribute_names)
    class_attributes = load_class_attributes(CLASS_ATTR_DIR)

    rng_py = random.Random(SEED)

    print("Building random/control concept pools from CUB's own train split...", flush=True)
    all_train_paths = [image_paths[i] for i in train_ids]
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

    results = []
    for attr_idx, class_indices in pairs_by_attr.items():
        attr_name = attribute_names[attr_idx]
        _prefix, _part_names = groundable[attr_idx]
        positive_classes = [cid for cid, vec in class_attributes.items() if vec[attr_idx]]
        pos_train_paths = [image_paths[i] for cid in positive_classes for i in train_ids_by_class.get(cid, [])]
        if not pos_train_paths:
            print(f"[SKIP] {attr_name}: no positive train images found", flush=True)
            continue
        rng_py.shuffle(pos_train_paths)
        safe_name = attr_name.replace("::", "__")
        target_concept = make_concept(concept_id, safe_name, pos_train_paths[:N_CONCEPT_EXEMPLARS], preprocess)
        concept_id += 1

        for class_idx in class_indices:
            cid_1indexed = class_idx + 1  # faithfulness's predicted_class is 0-indexed; class_labels is 1-indexed
            val_ids = test_ids_by_class.get(cid_1indexed)
            if not val_ids:
                print(f"[SKIP] {attr_name} -> class {class_idx}: no test images for this class", flush=True)
                continue
            val_paths = [image_paths[i] for i in val_ids]
            inputs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in val_paths]).to(DEVICE)

            experimental_sets = [[target_concept, rc] for rc in random_concepts]
            scores = tcav.interpret(inputs=inputs, experimental_sets=experimental_sets, target=class_idx)
            target_sign_counts = [
                scores[f"{target_concept.id}-{rc.id}"][spec.hook_layer]["sign_count"][0].item() for rc in random_concepts
            ]
            target_magnitudes = [
                scores[f"{target_concept.id}-{rc.id}"][spec.hook_layer]["magnitude"][0].item() for rc in random_concepts
            ]

            control_scores = tcav.interpret(inputs=inputs, experimental_sets=control_sets, target=class_idx)
            control_sign_counts = [
                control_scores[f"{a.id}-{b.id}"][spec.hook_layer]["sign_count"][0].item() for a, b in control_sets
            ]

            t_stat, p_value = ttest_ind(target_sign_counts, control_sign_counts)
            mean_sign_count = float(np.mean(target_sign_counts))
            mean_magnitude = float(np.mean(target_magnitudes))
            mean_control = float(np.mean(control_sign_counts))

            results.append(
                {
                    "attribute_index": attr_idx,
                    "attribute_name": attr_name,
                    "native_class_idx": class_idx,
                    "n_val_images": len(val_paths),
                    "mean_sign_count": mean_sign_count,
                    "mean_control_sign_count": mean_control,
                    "mean_magnitude": mean_magnitude,
                    "t_stat": t_stat,
                    "p_value": p_value,
                }
            )
            print(f"[{len(results):>4d}/{len(pairs)}] {attr_name:<35s} -> class {class_idx:>3d}: "
                  f"sign_count={mean_sign_count:.3f} (null={mean_control:.3f})", flush=True)

    with open(RESULTS_DIR / "tcav_cub_targeted_v49.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{len(results)}/{len(pairs)} targeted pairs scored, saved to results/tcav_cub_targeted_v49.csv")


if __name__ == "__main__":
    main()
