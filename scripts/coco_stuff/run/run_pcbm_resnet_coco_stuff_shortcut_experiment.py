"""PCBM-resnet attribution across 11 shortcut-rate COCO-Stuff models.

For each of 11 shortcut-rate checkpoints (0-100% in steps of 10%):
  1. Extract ResNet18 features from that checkpoint on the COCO train-split.
  2. Fit LinearSVC CAVs (one per concept) from Broden pos/neg crops using those features.
  3. Project train-split images onto 103 CAV directions -> (N_train, 103).
  4. Get surrogate labels from that model's logits (logit > 0).
  5. Fit SGDClassifier probe per class -> weights are PCBM scores.

Output: results/coco_stuff/shortcut/pcbm_resnet_shortcut_weights.csv
  rate_pct, concept_name, class_name, weight
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from cards.data.coco_stuff import COCO_STUFF_20_CLASSES
from cards.data.datasets import list_broden_concepts, load_broden
from cards.models.backbones import BACKBONES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BRODEN_ROOT = Path("/scratch/rnair21/LLMConceptEditing_old/broden_concepts")
RESULTS_DIR = Path("results/coco_stuff")
OUT_DIR     = Path("results/coco_stuff/shortcut")
CKPT_DIR    = Path("trained_models_new/coco_stuff/shortcut")
SPLIT_FILE  = OUT_DIR / "train_test_split.json"
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
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
SEED            = 42
BATCH_SIZE      = 128
RATES_PCT       = list(range(0, 101, 10))
MIN_FAITH_IMGS  = 100
CAV_C           = 0.1
N_CAV_EXEMPLARS = 150
PROBE_LAM       = 0.0002


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_balanced_concepts(counts_csv: Path, min_images: int) -> set[str]:
    balanced: set[str] = set()
    with open(counts_csv, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["n_images"]) >= min_images:
                balanced.add(row["concept_name"])
    return balanced


def _load_shortcut_model(rate_pct: int):
    import torch.nn as nn
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(COCO_STUFF_20_CLASSES))
    ckpt = CKPT_DIR / f"resnet18_coco_stuff_shortcut_{rate_pct}.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(DEVICE).eval()


def _feature_extractor(model: torch.nn.Module) -> torch.nn.Module:
    """Return ResNet18 with FC replaced by identity (512-dim output)."""
    import torch.nn as nn
    fe = torch.nn.Sequential(*list(model.children())[:-1])
    return fe.to(DEVICE).eval()


def _encode_batched(fe, paths: list[Path], preprocess) -> np.ndarray:
    chunks = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch = torch.stack([
            preprocess(Image.open(p).convert("RGB")) for p in paths[start: start + BATCH_SIZE]
        ]).to(DEVICE)
        with torch.no_grad():
            feats = fe(batch).squeeze(-1).squeeze(-1).cpu().numpy()
        chunks.append(feats)
        if (start // BATCH_SIZE) % 50 == 0:
            print(f"    encoded {start + len(batch)}/{len(paths)}", flush=True)
    return np.concatenate(chunks, axis=0)


def _native_logits_batched(paths: list[Path], model, preprocess) -> np.ndarray:
    chunks = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch = torch.stack([
            preprocess(Image.open(p).convert("RGB")) for p in paths[start: start + BATCH_SIZE]
        ]).to(DEVICE)
        with torch.no_grad():
            chunks.append(model(batch).cpu().numpy())
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import json
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_paths = [Path(p) for p in split["train"]]
    print(f"Train-split: {len(train_paths)} images", flush=True)

    balanced = _load_balanced_concepts(COUNTS_CSV, MIN_FAITH_IMGS)
    concepts = [
        c for c in list_broden_concepts(BRODEN_ROOT)
        if c not in BRODEN_DROPPED_CONCEPTS and c in balanced
    ]
    print(f"Concepts: {len(concepts)}", flush=True)

    spec       = BACKBONES["coco_stuff"]
    preprocess = spec.preprocess

    all_rows: list[dict] = []
    scores_by_rate: dict[int, float] = {}

    for rate_pct in RATES_PCT:
        print(f"\n=== rate={rate_pct}% ===", flush=True)
        model = _load_shortcut_model(rate_pct)
        fe    = _feature_extractor(model)

        # Encode Broden concepts + fit CAVs using this model's features
        print("  Fitting CAVs...", flush=True)
        cav_directions: list[np.ndarray] = []
        valid_concepts: list[str] = []

        for c_i, concept_name in enumerate(concepts):
            pos_paths, neg_paths = load_broden(BRODEN_ROOT, concept_name)
            pos_paths = list(pos_paths)[:N_CAV_EXEMPLARS]
            neg_paths = list(neg_paths)[:N_CAV_EXEMPLARS]
            if len(pos_paths) < 5 or len(neg_paths) < 5:
                continue

            pos_feats = _encode_batched(fe, pos_paths, preprocess)
            neg_feats = _encode_batched(fe, neg_paths, preprocess)
            X = np.concatenate([pos_feats, neg_feats], axis=0)
            y = np.array([1] * len(pos_feats) + [0] * len(neg_feats))

            clf = LinearSVC(C=CAV_C, max_iter=2000, random_state=SEED)
            clf.fit(X, y)
            cav = clf.coef_[0]
            cav = cav / (np.linalg.norm(cav) + 1e-8)
            cav_directions.append(cav)
            valid_concepts.append(concept_name)

        cav_matrix = np.stack(cav_directions, axis=0)  # (n_concepts, 512)
        print(f"  CAVs fitted: {len(valid_concepts)}", flush=True)

        # Project train-split images
        print("  Encoding train-split...", flush=True)
        train_feats  = _encode_batched(fe, train_paths, preprocess)     # (N, 512)
        train_proj   = train_feats @ cav_matrix.T                        # (N, n_concepts)
        train_logits = _native_logits_batched(train_paths, model, preprocess)  # (N, 20)

        # Fit probe per class
        abs_weights: list[float] = []
        for cls_idx, cls_name in enumerate(COCO_STUFF_20_CLASSES):
            surr = (train_logits[:, cls_idx] > 0).astype(int)
            n_pos = surr.sum()
            if n_pos < 10 or n_pos > len(surr) - 10:
                continue
            probe = SGDClassifier(
                loss="log_loss", alpha=PROBE_LAM, max_iter=1000,
                class_weight="balanced", random_state=SEED,
            )
            probe.fit(train_proj, surr)
            coef = probe.coef_[0]
            for col, concept_name in enumerate(valid_concepts):
                w = float(coef[col])
                all_rows.append({"rate_pct": rate_pct, "concept_name": concept_name,
                                 "class_name": cls_name, "weight": w})
                abs_weights.append(abs(w))

        mean_abs = float(np.mean(abs_weights)) if abs_weights else 0.0
        scores_by_rate[rate_pct] = mean_abs
        print(f"  mean |weight|: {mean_abs:.4f}", flush=True)

    out_path = OUT_DIR / "pcbm_resnet_shortcut_weights.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rate_pct", "concept_name", "class_name", "weight"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows -> {out_path}", flush=True)

    print("\n=== PCBM-resnet mean |weight| per rate ===", flush=True)
    for rate_pct in RATES_PCT:
        print(f"  rate={rate_pct:>3d}%: {scores_by_rate.get(rate_pct, float('nan')):.4f}", flush=True)


if __name__ == "__main__":
    main()
