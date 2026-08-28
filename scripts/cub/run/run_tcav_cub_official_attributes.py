"""Runs TCAV against CUB's official 112-concept bank's own concept
definition (whole, uncropped images, split into positive/negative purely
by each SPECIES' own class-level attribute label -- post_hoc_cbm's own
convention, confirmed in notes v34, no cropping involved at all), for
the 87 attributes that also have a faithfulness ground truth (v38).

Unlike v35's 14-pair hand-picked validation slice, this aims for real
coverage across many (attribute, species) pairs so the results can
actually be scored against the faithfulness ground truth via
score_method_agreement (min_samples_per_pair=3 needs real pair overlap,
which a 14-pair slice mostly can't provide -- see notes v39). Target
species per attribute are the SAME "positive species" population v38's
faithfulness run itself sampled from, to maximize the chance of hitting
a predicted_class the ground truth actually has samples for.

CAV fitting is the expensive part (once per (attribute, random/control-
concept) pair); a target species' own interpret() call after that first
fit is cheap (captum's own CAV cache, keyed by concept ids + layer, not
by the interpreted inputs) -- so testing several target species per
attribute is far cheaper than the initial N_RANDOM+N_CONTROL CAV fits,
which is why this scales targets-per-attribute up while keeping
random/control set counts modest to bound the dominant cost.
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from cards.data.cub_attributes import groundable_attributes, load_attribute_names, load_class_attributes  # noqa: E402
from cards.data.cub_parts import load_images_txt  # noqa: E402
from cards.models.backbones import BACKBONES  # noqa: E402
from concepts.concept_utils import ListDataset  # noqa: E402

from captum.concept import TCAV, Concept  # noqa: E402

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
ATTRIBUTE_NAMES_PATH = CUB_ROOT / "attributes" / "new_attributes.txt"
CLASS_ATTR_DIR = CUB_ROOT / "class_attr_data_10"
RESULTS_DIR = Path("results")
SEED = 42
DEVICE = "cpu"  # same rationale as run_tcav.py/run_tcav_cub.py -- captum's Concept.data_iter never moves batches to CUDA
N_RANDOM = 6
N_CONTROL = 6
N_PER_RANDOM_SET = 25
N_CONCEPT_EXEMPLARS = 40
N_TARGETS_PER_ATTRIBUTE = 5


def make_concept(concept_id: int, name: str, image_paths: list[Path], preprocess, batch_size: int = 32) -> Concept:
    ds = ListDataset([str(p) for p in image_paths], preprocess=preprocess)
    return Concept(id=concept_id, name=name, data_iter=DataLoader(ds, batch_size=batch_size, shuffle=False))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    spec = BACKBONES["resnet18_cub"]
    native_model = spec.load_native().to(DEVICE).eval()
    preprocess = spec.preprocess

    tcav = TCAV(model=native_model, layers=[spec.hook_layer], save_path=str(RESULTS_DIR / "tcav_cub_attr_cav_cache"))

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
    train_ids_by_class: dict[int, list[str]] = {}
    for i in train_ids:
        train_ids_by_class.setdefault(class_labels[i], []).append(i)
    test_ids_by_class: dict[int, list[str]] = {}
    for i in test_ids:
        test_ids_by_class.setdefault(class_labels[i], []).append(i)

    attribute_names = load_attribute_names(ATTRIBUTE_NAMES_PATH)
    groundable = groundable_attributes(attribute_names)
    class_attributes = load_class_attributes(CLASS_ATTR_DIR)
    print(f"{len(groundable)}/{len(attribute_names)} official attributes have a faithfulness ground truth to score against.", flush=True)

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
    for attr_idx, (prefix, _part_names) in groundable.items():
        attr_name = attribute_names[attr_idx]
        positive_classes = [cid for cid, vec in class_attributes.items() if vec[attr_idx]]
        if not positive_classes:
            continue

        pos_train_paths = [image_paths[i] for cid in positive_classes for i in train_ids_by_class.get(cid, [])]
        rng_py.shuffle(pos_train_paths)
        # captum's own CAV cache uses Concept.name directly to build a
        # filesystem directory -- "::" (as in "has_bill_shape::dagger") is
        # invalid in a Windows path, so sanitize the name used for the
        # Concept object; attr_name (with "::" intact) stays the one saved
        # to the results CSV.
        safe_name = attr_name.replace("::", "__")
        target_concept = make_concept(concept_id, safe_name, pos_train_paths[:N_CONCEPT_EXEMPLARS], preprocess)
        concept_id += 1

        target_species = positive_classes[:]
        rng_py.shuffle(target_species)
        target_species = [cid for cid in target_species if test_ids_by_class.get(cid)][:N_TARGETS_PER_ATTRIBUTE]

        for target_cid in target_species:
            val_paths = [image_paths[i] for i in test_ids_by_class[target_cid]]
            inputs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in val_paths]).to(DEVICE)
            target_idx = target_cid - 1  # native model's 0-indexed class output

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
                    "attribute_index": attr_idx,
                    "attribute_name": attr_name,
                    "attribute_prefix": prefix,
                    "native_class_idx": target_idx,
                    "n_val_images": len(val_paths),
                    "mean_sign_count": mean_sign_count,
                    "mean_control_sign_count": mean_control,
                    "mean_magnitude": mean_magnitude,
                    "t_stat": t_stat,
                    "p_value": p_value,
                }
            )

        print(f"[{len(results):>5d} pairs so far] {attr_name:<40s}: {len(target_species)} target species scored", flush=True)

    with open(RESULTS_DIR / "tcav_cub_attribute_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{len(results)} total (attribute, species) TCAV pairs saved to results/tcav_cub_attribute_scores.csv")


if __name__ == "__main__":
    main()
