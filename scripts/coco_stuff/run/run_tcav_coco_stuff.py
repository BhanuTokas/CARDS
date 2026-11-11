"""TCAV baseline for COCO-Stuff.

Mirrors scripts/celeba/run/run_tcav_celeba_full.py but for 143 Broden
concepts × 20 COCO-Stuff co-occurrence-bias classes.

Key differences from the CelebA version:
  - Concept positives: Broden directory images (load_broden) instead of
    CelebA attribute-positive images.
  - Random/control pools: sampled from COCO train2017 (not CelebA train).
  - Validation images: random sample of COCO val2017 (multi-label dataset,
    no task-positive filtering needed for TCAV's gradient direction check).
  - 20 target classes (sigmoid heads) instead of 2 (softmax tasks).
  - DEVICE = "cpu" -- captum's Concept.data_iter constraint, same as CelebA.

Outputs:
  results/coco_stuff/tcav_coco_stuff_scores.csv
    concept_name, class_name, n_val_images, mean_sign_count,
    mean_control_sign_count, mean_magnitude, t_stat, p_value

Also prints Spearman rho vs. faithfulness at the end (all concepts +
co-occur subset), matching script 02b's evaluation format.

Prerequisites:
  scripts/coco_stuff/build/train_coco_stuff_classifier.py
  scripts/coco_stuff/build/compute_coco_stuff_faithfulness.py
"""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import torch
from PIL import Image
from scipy import stats
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader

from captum.concept import TCAV, Concept


class ListDataset:
    def __init__(self, images, preprocess=None):
        self.images = images
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = Image.open(self.images[idx]).convert("RGB")
        return self.preprocess(img) if self.preprocess else img

from cards.data.coco_stuff import CO_OCCUR_BRODEN_CONCEPTS, COCO_STUFF_20_CLASSES
from cards.data.datasets import list_broden_concepts, load_broden, load_coco_stuff
from cards.models.backbones import BACKBONES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COCO_ROOT   = Path("/scratch/rnair21/LLMConceptEditing_old/coco")
BRODEN_ROOT = Path("/scratch/rnair21/LLMConceptEditing_old/broden_concepts")
RESULTS_DIR = Path("results/coco_stuff")
FAITH_CSV   = RESULTS_DIR / "coco_stuff_faithfulness.csv"

BRODEN_DROPPED_CONCEPTS: set[str] = {
    "air_conditioner", "apron", "awning", "bathroom_s", "beak", "bedroom_s",
    "blotchy", "bumper", "can", "candlestick", "cap", "clock", "dining_room_s",
    "drinking_glass", "eye", "eyebrow", "faucet", "figurine", "handle",
    "inside_arm", "jar", "knob", "loudspeaker", "mouse", "mouth",
    "outside_arm", "street_s",
}

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SEED                = 42
DEVICE              = "cpu"   # captum's ListDataset constraint
N_RANDOM            = 6
N_CONTROL           = 6
N_PER_RANDOM_SET    = 25
N_CONCEPT_EXEMPLARS = 40
N_VAL_SAMPLES       = 40
MIN_SAMPLES         = 3       # min faithfulness records per pair for rho


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_concept(concept_id: int, name: str, image_paths: list[Path], preprocess, batch_size: int = 32) -> Concept:
    ds = ListDataset([str(p) for p in image_paths], preprocess=preprocess)
    return Concept(id=concept_id, name=name, data_iter=DataLoader(ds, batch_size=batch_size, shuffle=False))


def _load_faithfulness(path: Path) -> dict[tuple[str, str], list[float]]:
    delta_p: dict[tuple[str, str], list[float]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            dp = float(row["delta_p"])
            label = int(row["class_label"])
            delta_p[(row["concept_name"], row["class_name"])].append(dp if label == 1 else -dp)
    return dict(delta_p)


def _eval_rho(
    scores: dict[tuple[str, str], float],
    faith: dict[tuple[str, str], list[float]],
) -> tuple[float, float]:
    """Returns (mean_rho_all_concepts, mean_rho_co_occur_subset)."""
    class_rhos_all, class_rhos_co = [], []
    for cls_name in COCO_STUFF_20_CLASSES:
        for rho_list, co_only in [(class_rhos_all, False), (class_rhos_co, True)]:
            s_vals, g_vals = [], []
            for (c, cl), dp_list in faith.items():
                if cl != cls_name or len(dp_list) < MIN_SAMPLES:
                    continue
                if co_only and c not in CO_OCCUR_BRODEN_CONCEPTS:
                    continue
                if (c, cl) not in scores:
                    continue
                s_vals.append(scores[(c, cl)])
                g_vals.append(float(np.mean(dp_list)))
            if len(s_vals) >= 2:
                rho, _ = stats.spearmanr(s_vals, g_vals)
                rho_list.append(float(rho))
    return (
        float(np.mean(class_rhos_all)) if class_rhos_all else float("nan"),
        float(np.mean(class_rhos_co))  if class_rhos_co  else float("nan"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng_py = random.Random(SEED)

    # --- Backbone (DEVICE=cpu for captum) ---
    spec         = BACKBONES["coco_stuff"]
    native_model = spec.load_native().to(DEVICE).eval()
    preprocess   = spec.preprocess
    print(f"Backbone: coco_stuff (ResNet18, 20-class)  device={DEVICE}", flush=True)

    tcav = TCAV(
        model=native_model,
        layers=[spec.hook_layer],
        save_path=str(RESULTS_DIR / "tcav_coco_stuff_cav_cache"),
    )

    # --- Concepts (143 Broden) ---
    concepts = [c for c in list_broden_concepts(BRODEN_ROOT) if c not in BRODEN_DROPPED_CONCEPTS]
    print(f"Concepts: {len(concepts)}", flush=True)

    # --- Random/control pools from COCO train2017 ---
    print("Building random/control pools from COCO train2017...", flush=True)
    train_pairs = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="train")
    all_train_paths = [p for p, _ in train_pairs]
    rng_py.shuffle(all_train_paths)

    concept_id = 0
    idx = 0
    random_concepts: list[Concept] = []
    for i in range(N_RANDOM):
        chunk = all_train_paths[idx : idx + N_PER_RANDOM_SET]
        idx += N_PER_RANDOM_SET
        random_concepts.append(make_concept(concept_id, f"random_{i}", chunk, preprocess))
        concept_id += 1
    control_concepts: list[Concept] = []
    for i in range(N_CONTROL):
        chunk = all_train_paths[idx : idx + N_PER_RANDOM_SET]
        idx += N_PER_RANDOM_SET
        control_concepts.append(make_concept(concept_id, f"control_{i}", chunk, preprocess))
        concept_id += 1
    control_sets = [[control_concepts[2 * i], control_concepts[2 * i + 1]] for i in range(N_CONTROL // 2)]
    print(f"{N_RANDOM} random + {N_CONTROL} control pools built ({idx} train images used)", flush=True)

    # --- Validation images from COCO val2017 ---
    val_pairs = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="val")
    val_paths_all = [p for p, _ in val_pairs]
    rng_py.shuffle(val_paths_all)
    val_paths = val_paths_all[:N_VAL_SAMPLES]
    inputs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in val_paths]).to(DEVICE)
    print(f"Validation images: {len(val_paths)} from COCO val2017", flush=True)

    # --- Per-concept TCAV scoring ---
    results: list[dict] = []
    scores_for_rho: dict[tuple[str, str], float] = {}

    for c_i, concept_name in enumerate(concepts):
        pos_paths, _ = load_broden(BRODEN_ROOT, concept_name)
        pos_paths = list(pos_paths)
        rng_py.shuffle(pos_paths)
        pos_paths = pos_paths[:N_CONCEPT_EXEMPLARS]
        if len(pos_paths) == 0:
            print(f"[{c_i + 1:>3d}/{len(concepts)}] {concept_name:<25s} SKIP (no positives)", flush=True)
            continue

        target_concept = make_concept(concept_id, concept_name, pos_paths, preprocess)
        concept_id += 1
        experimental_sets = [[target_concept, rc] for rc in random_concepts]

        for cls_idx, cls_name in enumerate(COCO_STUFF_20_CLASSES):
            scores_captum = tcav.interpret(
                inputs=inputs,
                experimental_sets=experimental_sets,
                target=cls_idx,
            )
            target_sign_counts = [
                scores_captum[f"{target_concept.id}-{rc.id}"][spec.hook_layer]["sign_count"][0].item()
                for rc in random_concepts
            ]
            target_magnitudes = [
                scores_captum[f"{target_concept.id}-{rc.id}"][spec.hook_layer]["magnitude"][0].item()
                for rc in random_concepts
            ]

            control_scores_captum = tcav.interpret(
                inputs=inputs,
                experimental_sets=control_sets,
                target=cls_idx,
            )
            control_sign_counts = [
                control_scores_captum[f"{a.id}-{b.id}"][spec.hook_layer]["sign_count"][0].item()
                for a, b in control_sets
            ]

            t_stat, p_value = ttest_ind(target_sign_counts, control_sign_counts)
            mean_sign   = float(np.mean(target_sign_counts))
            mean_mag    = float(np.mean(target_magnitudes))
            mean_ctrl   = float(np.mean(control_sign_counts))

            results.append({
                "concept_name":             concept_name,
                "class_name":               cls_name,
                "n_val_images":             len(val_paths),
                "mean_sign_count":          mean_sign,
                "mean_control_sign_count":  mean_ctrl,
                "mean_magnitude":           mean_mag,
                "t_stat":                   float(t_stat),
                "p_value":                  float(p_value),
            })
            scores_for_rho[(concept_name, cls_name)] = mean_sign

        print(
            f"[{c_i + 1:>3d}/{len(concepts)}] {concept_name:<25s} "
            f"sign_count mean over classes={float(np.mean([r['mean_sign_count'] for r in results if r['concept_name'] == concept_name])):.3f}",
            flush=True,
        )

    # --- Save CSV ---
    out_path = RESULTS_DIR / "tcav_coco_stuff_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows → {out_path}", flush=True)

    # --- Evaluate vs. faithfulness ---
    if FAITH_CSV.exists():
        faith = _load_faithfulness(FAITH_CSV)
        mean_rho_all, mean_rho_co = _eval_rho(scores_for_rho, faith)
        print(f"\nSpearman rho vs. faithfulness:", flush=True)
        print(f"  all concepts:    {mean_rho_all:+.3f}", flush=True)
        print(f"  co-occur subset: {mean_rho_co:+.3f}", flush=True)
    else:
        print(f"\nFaithfulness CSV not found ({FAITH_CSV}) -- skipping rho evaluation", flush=True)


if __name__ == "__main__":
    main()
