"""TCAV against all 4 shortcut-injected classifiers (0%/33%/67%/100%,
see scripts/celeba/build/train_attractive_shortcut_classifiers.py) --
a masking-INDEPENDENT attribution method run on the same controlled
shortcut experiment as run_attribution_shortcut_experiment.py, prompted
directly ("Can we also check the attribution for TCAV and PCBM?" ->
"I am planning on adding this experiment to the paper, as a way to
remove circularity concerns due to masking."). TCAV never masks an
image at all (it probes activation-space directional derivatives), so
a matching decline here would show the masking hybrid's own result
isn't an artifact of the masking mechanism specifically.

Matches run_tcav_celeba_full.py's own already-reported settings exactly
(N_RANDOM=6, N_CONTROL=6, N_PER_RANDOM_SET=25, N_CONCEPT_EXEMPLARS=40,
N_VAL_SAMPLES=40, layer4 hook) -- paper-bound, so no reduced/approximate
version, full parity with this track's existing TCAV convention.

Concept exemplars and random/control pools are whole real-attribute-
positive CelebA-HQ TRAIN images (static, model-independent) -- built
ONCE, reused across all 4 rate checkpoints, same as
run_attribution_shortcut_experiment.py's own shared-localization-cache
design. The scoring/validation sample (which images TCAV's own sign_
count/magnitude get computed FROM) is drawn from the official CelebA
val pool instead of CelebA-HQ's own val split -- the standing default
for CelebA evaluation pools ("Can you please default to that for all
future experiments unless explicitly told otherwise"), cross-referenced
against standard CelebA's own list_attr_celeba.txt for real Attractive
labels (build_clean_official_val_paths() only returns paths, not
labels). Built ONCE, reused across all 4 rates for a fair comparison.

target=1 (the single 2-way head's positive-class logit), NOT the
original 4-way head's TASK_POSITIVE_LOGIT_INDEX -- these are freshly
trained single-task classifiers, not the joint Attractive/Young model.
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
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from captum.concept import TCAV, Concept
from concepts.concept_utils import ListDataset
from run_cards_celeba_masking_hybrid_official_val_zscore import build_clean_official_val_paths

from cards.data.celeba import load_celebamask_hq_image_paths, split_celebamask_hq
from cards.data.celeba_attributes import (
    GROUNDABLE_CONCEPTS,
    TARGET_CLASSES,
    load_attribute_labels,
    load_attribute_names,
)
from cards.models.backbones import BACKBONES

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
CELEBA_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebA\celeba")
RESULTS_DIR = Path("results")
CKPT_DIR = Path("trained_models_new/celeba")
SEED = 42
DEVICE = "cpu"  # captum's Concept.data_iter never moves batches to CUDA, matching run_tcav_celeba_full.py
N_RANDOM = 6
N_CONTROL = 6
N_PER_RANDOM_SET = 25
N_CONCEPT_EXEMPLARS = 40
N_VAL_SAMPLES = 40
RATES_PCT = [0, 33, 67, 100]
HOOK_LAYER = "layer4"
TARGET_IDX = 1  # single 2-way head's positive-class logit


def build_model(rate_pct: int) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(CKPT_DIR / f"resnet18_attractive_shortcut_{rate_pct}.pt", map_location="cpu")
    model.load_state_dict(state)
    return model.eval()


def make_concept(concept_id: int, name: str, image_paths: list, preprocess, batch_size: int = 32) -> Concept:
    ds = ListDataset([str(p) for p in image_paths], preprocess=preprocess)
    return Concept(id=concept_id, name=name, data_iter=DataLoader(ds, batch_size=batch_size, shuffle=False))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    spec = BACKBONES["celeba_attractive_young"]  # preprocess only -- architecture-generic, not this experiment's own checkpoint
    preprocess = spec.preprocess

    print("Loading CelebAMask-HQ metadata (concept-bank fitting resources)...", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]
    train_hq, _val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices)
    print(f"{len(GROUNDABLE_CONCEPTS)} groundable concepts to score.", flush=True)

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

    target_concepts = {}
    for concept_name in GROUNDABLE_CONCEPTS:
        concept_attr_idx = attr_names.index(concept_name)
        pos_train_paths = [
            image_paths_by_idx[i] for i in train_hq if attr_labels_by_file[f"{i}.jpg"][concept_attr_idx]
        ]
        rng_py.shuffle(pos_train_paths)
        target_concepts[concept_name] = make_concept(
            concept_id, concept_name, pos_train_paths[:N_CONCEPT_EXEMPLARS], preprocess
        )
        concept_id += 1

    print("\nBuilding official-val scoring sample (real Attractive=1 images)...", flush=True)
    official_paths = build_clean_official_val_paths()
    official_attr_names = load_attribute_names(CELEBA_ROOT / "list_attr_celeba.txt")
    official_attr_labels = load_attribute_labels(CELEBA_ROOT / "list_attr_celeba.txt")
    official_attractive_idx = official_attr_names.index("Attractive")
    positive_official = [
        p for p in official_paths if official_attr_labels[p.name][official_attractive_idx]
    ]
    rng_py.shuffle(positive_official)
    val_paths = positive_official[:N_VAL_SAMPLES]
    print(f"{len(positive_official)} official-val images positive for Attractive; using {len(val_paths)}.", flush=True)
    inputs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in val_paths]).to(DEVICE)

    all_rows = []  # (rate_pct, concept_name, mean_sign_count, mean_control_sign_count, mean_magnitude, t_stat, p_value)
    scores_by_rate: dict[int, dict[str, float]] = {}

    for rate_pct in RATES_PCT:
        print(f"\n=== rate={rate_pct}% ===", flush=True)
        model = build_model(rate_pct).to(DEVICE)
        tcav = TCAV(model=model, layers=[HOOK_LAYER], save_path=str(RESULTS_DIR / f"tcav_shortcut_cav_cache_{rate_pct}"))

        scores_by_rate[rate_pct] = {}
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

            scores_by_rate[rate_pct][concept_name] = mean_magnitude
            all_rows.append((rate_pct, concept_name, mean_sign_count, mean_control, mean_magnitude, t_stat, p_value))
            print(f"  {concept_name:<20s} sign_count={mean_sign_count:.3f} (null={mean_control:.3f}) "
                  f"magnitude={mean_magnitude:.4f} p={p_value:.4g}", flush=True)

        ranked = sorted(scores_by_rate[rate_pct].items(), key=lambda kv: -abs(kv[1]))
        print(f"  top-5 by |magnitude|: {[(c, round(s, 4)) for c, s in ranked[:5]]}", flush=True)
        print(f"  mean |magnitude| across all 26 concepts: {np.mean([abs(s) for s in scores_by_rate[rate_pct].values()]):.4f}", flush=True)

    with open(RESULTS_DIR / "tcav_celeba_shortcut_experiment.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rate_pct", "concept_name", "mean_sign_count", "mean_control_sign_count", "mean_magnitude", "t_stat", "p_value"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to results/tcav_celeba_shortcut_experiment.csv")

    print("\n=== mean |magnitude| across all 26 concepts, per rate (the headline decline check) ===")
    for rate_pct in RATES_PCT:
        mean_abs = np.mean([abs(s) for s in scores_by_rate[rate_pct].values()])
        print(f"  rate={rate_pct:>3d}%: mean |magnitude| = {mean_abs:.4f}")


if __name__ == "__main__":
    main()
