"""TCAV attribution across 11 shortcut-rate COCO-Stuff models.

For each of 11 shortcut-rate checkpoints (0-100% in steps of 10%),
runs TCAV with 103 balanced Broden concepts x 20 COCO-Stuff classes.
Random/control pools are sampled from COCO val2017 (model-independent, built once).
Validation images are also from COCO val2017.

Output: results/coco_stuff/shortcut/tcav_shortcut_scores.csv
  rate_pct, concept_name, class_name, mean_sign_count, mean_magnitude
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from captum.concept import TCAV, Concept

from cards.data.coco_stuff import COCO_STUFF_20_CLASSES
from cards.data.datasets import list_broden_concepts, load_broden, load_coco_stuff
from cards.models.backbones import BACKBONES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COCO_ROOT   = Path("/scratch/rnair21/LLMConceptEditing_old/coco")
BRODEN_ROOT = Path("/scratch/rnair21/LLMConceptEditing_old/broden_concepts")
RESULTS_DIR = Path("results/coco_stuff")
OUT_DIR     = Path("results/coco_stuff/shortcut")
CKPT_DIR    = Path("trained_models_new/coco_stuff/shortcut")
COUNTS_CSV  = RESULTS_DIR / "coco_stuff_faithfulness_detection_counts.csv"

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
DEVICE              = "cpu"   # captum constraint
RATES_PCT           = list(range(0, 101, 10))
MIN_FAITH_IMGS      = 100
N_RANDOM            = 6
N_CONTROL           = 6
N_PER_RANDOM_SET    = 25
N_CONCEPT_EXEMPLARS = 40
N_VAL_SAMPLES       = 40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ListDataset:
    def __init__(self, paths: list[str], preprocess):
        self.paths     = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.preprocess(img)


def _make_concept(cid: int, name: str, paths: list[Path], preprocess) -> Concept:
    ds = _ListDataset([str(p) for p in paths], preprocess)
    return Concept(id=cid, name=name, data_iter=DataLoader(ds, batch_size=32, shuffle=False))


def _load_balanced_concepts(counts_csv: Path, min_images: int) -> set[str]:
    balanced: set[str] = set()
    with open(counts_csv, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["n_images"]) >= min_images:
                balanced.add(row["concept_name"])
    return balanced


def _load_shortcut_model(rate_pct: int, spec, device: str) -> torch.nn.Module:
    import torch.nn as nn
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(COCO_STUFF_20_CLASSES))
    ckpt = CKPT_DIR / f"resnet18_coco_stuff_shortcut_{rate_pct}.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng_py = random.Random(SEED)

    spec       = BACKBONES["coco_stuff"]
    preprocess = spec.preprocess

    # --- Balanced concepts ---
    balanced = _load_balanced_concepts(COUNTS_CSV, MIN_FAITH_IMGS)
    concepts = [
        c for c in list_broden_concepts(BRODEN_ROOT)
        if c not in BRODEN_DROPPED_CONCEPTS and c in balanced
    ]
    print(f"Concepts: {len(concepts)}", flush=True)

    # --- Random / control pools from COCO val2017 (model-independent) ---
    print("Building random/control pools from COCO val2017...", flush=True)
    val_pairs      = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="val")
    val_paths_all  = [p for p, _ in val_pairs]
    rng_py.shuffle(val_paths_all)

    concept_id = 0
    idx = 0
    random_concepts: list[Concept] = []
    for i in range(N_RANDOM):
        chunk = val_paths_all[idx: idx + N_PER_RANDOM_SET]
        idx  += N_PER_RANDOM_SET
        random_concepts.append(_make_concept(concept_id, f"random_{i}", chunk, preprocess))
        concept_id += 1
    control_concepts: list[Concept] = []
    for i in range(N_CONTROL):
        chunk = val_paths_all[idx: idx + N_PER_RANDOM_SET]
        idx  += N_PER_RANDOM_SET
        control_concepts.append(_make_concept(concept_id, f"control_{i}", chunk, preprocess))
        concept_id += 1
    control_sets = [
        [control_concepts[2 * i], control_concepts[2 * i + 1]]
        for i in range(N_CONTROL // 2)
    ]

    # --- Validation images ---
    rng_py.shuffle(val_paths_all)
    val_paths = val_paths_all[:N_VAL_SAMPLES]
    inputs = torch.stack([
        preprocess(Image.open(p).convert("RGB")) for p in val_paths
    ]).to(DEVICE)
    print(f"Val images: {len(val_paths)}", flush=True)

    # --- Pre-load Broden concept paths (model-independent) ---
    concept_pos_paths: dict[str, list[Path]] = {}
    for concept_name in concepts:
        pos_paths, _ = load_broden(BRODEN_ROOT, concept_name)
        pos_paths = list(pos_paths)
        rng_py.shuffle(pos_paths)
        concept_pos_paths[concept_name] = pos_paths[:N_CONCEPT_EXEMPLARS]

    all_rows: list[dict] = []
    scores_by_rate: dict[int, float] = {}

    for rate_pct in RATES_PCT:
        print(f"\n=== rate={rate_pct}% ===", flush=True)
        model = _load_shortcut_model(rate_pct, spec, DEVICE)

        tcav = TCAV(
            model=model,
            layers=[spec.hook_layer],
            save_path=str(OUT_DIR / f"tcav_cav_cache_shortcut_{rate_pct}"),
        )

        abs_scores: list[float] = []

        for c_i, concept_name in enumerate(concepts):
            pos_paths = concept_pos_paths[concept_name]
            if not pos_paths:
                continue

            target_concept = _make_concept(concept_id, concept_name, pos_paths, preprocess)
            concept_id += 1
            experimental_sets = [[target_concept, rc] for rc in random_concepts]

            for cls_idx, cls_name in enumerate(COCO_STUFF_20_CLASSES):
                scores_captum = tcav.interpret(
                    inputs=inputs,
                    experimental_sets=experimental_sets,
                    target=cls_idx,
                )
                sign_counts = [
                    scores_captum[f"{target_concept.id}-{rc.id}"][spec.hook_layer]["sign_count"][0].item()
                    for rc in random_concepts
                ]
                magnitudes = [
                    scores_captum[f"{target_concept.id}-{rc.id}"][spec.hook_layer]["magnitude"][0].item()
                    for rc in random_concepts
                ]
                mean_sign = float(np.mean(sign_counts))
                mean_mag  = float(np.mean(magnitudes))

                all_rows.append({
                    "rate_pct":        rate_pct,
                    "concept_name":    concept_name,
                    "class_name":      cls_name,
                    "mean_sign_count": mean_sign,
                    "mean_magnitude":  mean_mag,
                })
                abs_scores.append(abs(mean_sign))

            if (c_i + 1) % 10 == 0:
                print(f"  [{c_i + 1}/{len(concepts)}] {concept_name}", flush=True)

        mean_abs = float(np.mean(abs_scores)) if abs_scores else 0.0
        scores_by_rate[rate_pct] = mean_abs
        print(f"  mean |sign_count|: {mean_abs:.4f}", flush=True)

    # --- Save ---
    out_path = OUT_DIR / "tcav_shortcut_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rate_pct", "concept_name", "class_name",
                                               "mean_sign_count", "mean_magnitude"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows -> {out_path}", flush=True)

    print("\n=== TCAV mean |sign_count| per rate ===", flush=True)
    for rate_pct in RATES_PCT:
        print(f"  rate={rate_pct:>3d}%: {scores_by_rate[rate_pct]:.4f}", flush=True)


if __name__ == "__main__":
    main()
