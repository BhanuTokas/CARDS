"""PCBM-resnet baseline for COCO-Stuff.

Fits a PCBM linear classifier as a surrogate of the ResNet18 COCO-Stuff
classifier, using Broden concept activation vectors (CAVs) fitted inline
with sklearn LinearSVC (no post_hoc_cbm dependency).

Pipeline:
  1. Load ResNet18 feature extractor (backbone without FC head).
  2. For each of 143 Broden concepts:
       - Encode Broden positives + negatives with the feature extractor.
       - Fit LinearSVC(C=0.1) on combined pos+neg features.
       - CAV direction = clf.coef_[0], L2-normalized.
  3. Encode all COCO train2017 images → project onto CAV directions
     → concept activation matrix (N_train, n_cavs).
  4. Get native model's logits on train images → binary surrogate labels
     per class (logit > 0 → positive).
  5. For each of 20 classes: fit SGDClassifier on activations → surrogate.
  6. Save weights as (concept_name, class_name, weight) CSV.
  7. Print Spearman rho vs. faithfulness.

Outputs (results/coco_stuff/):
  pcbm_resnet_concept_weights.csv  -- concept_name, class_name, weight

Prerequisites:
  scripts/coco_stuff/build/train_coco_stuff_classifier.py
  scripts/coco_stuff/build/compute_coco_stuff_faithfulness.py
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
from scipy import stats
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

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
COUNTS_CSV  = RESULTS_DIR / "coco_stuff_faithfulness_detection_counts.csv"
MIN_FAITH_IMAGES = 100

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
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEED          = 42
BATCH_SIZE    = 64
CAV_C         = 0.1    # LinearSVC regularization (matches CelebA PCBM default)
N_CAV_MAX     = 150    # max Broden exemplars per concept side
N_CAV_MIN     = 5      # skip concept if fewer pos or neg exemplars
PROBE_LAM     = 2e-4   # SGDClassifier L2 reg (matches CelebA PCBM default)
MIN_SAMPLES   = 3      # min faithfulness records per pair for rho eval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_features(
    paths: list[Path],
    feature_extractor: torch.nn.Module,
    preprocess,
    label: str = "",
) -> np.ndarray:
    feats: list[np.ndarray] = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start : start + BATCH_SIZE]
        batch = torch.stack([
            preprocess(Image.open(p).convert("RGB")) for p in batch_paths
        ]).to(DEVICE)
        with torch.no_grad():
            f = feature_extractor(batch)          # (B, 512, 1, 1)
        feats.append(torch.flatten(f, 1).cpu().numpy())
        if label and (start // BATCH_SIZE) % 50 == 0:
            print(f"  [{label}] {start + len(batch_paths)}/{len(paths)}", flush=True)
    return np.concatenate(feats, axis=0)


def _native_logits(
    paths: list[Path],
    native_model: torch.nn.Module,
    preprocess,
) -> np.ndarray:
    logits: list[np.ndarray] = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start : start + BATCH_SIZE]
        batch = torch.stack([
            preprocess(Image.open(p).convert("RGB")) for p in batch_paths
        ]).to(DEVICE)
        with torch.no_grad():
            logits.append(native_model(batch).cpu().numpy())
    return np.concatenate(logits, axis=0)   # (N, 20)


def _load_balanced_concepts(counts_csv: Path, min_images: int) -> set[str]:
    balanced: set[str] = set()
    with open(counts_csv, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["n_images"]) >= min_images:
                balanced.add(row["concept_name"])
    return balanced


def _load_faithfulness(path: Path, strategy: str = "black_mask") -> dict[tuple[str, str], list[float]]:
    col = f"delta_p_{strategy}"
    delta_p: dict[tuple[str, str], list[float]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            delta_p[(row["concept_name"], row["class_name"])].append(float(row[col]))
    return dict(delta_p)


def _eval_rho(
    scores: dict[tuple[str, str], float],
    faith: dict[tuple[str, str], list[float]],
) -> tuple[float, float]:
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
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- Backbone ---
    spec             = BACKBONES["coco_stuff"]
    native_model     = spec.load_native().to(DEVICE).eval()
    feature_extractor = spec.feature_extractor(native_model).to(DEVICE).eval()
    preprocess       = spec.preprocess
    print(f"Backbone: coco_stuff (ResNet18)  device={DEVICE}", flush=True)

    # --- Broden concepts (filtered to those with >= MIN_FAITH_IMAGES faithfulness images) ---
    balanced = _load_balanced_concepts(COUNTS_CSV, MIN_FAITH_IMAGES)
    concepts = [c for c in list_broden_concepts(BRODEN_ROOT)
                if c not in BRODEN_DROPPED_CONCEPTS and c in balanced]
    print(f"Concepts: {len(concepts)} (>= {MIN_FAITH_IMAGES} faithfulness images)", flush=True)

    # --- Step 1: Fit CAVs for each concept ---
    print("\n=== Fitting CAVs ===", flush=True)
    cav_directions: dict[str, np.ndarray] = {}
    for c_i, concept in enumerate(concepts):
        pos_paths, neg_paths = load_broden(BRODEN_ROOT, concept)
        pos_paths = list(pos_paths)[:N_CAV_MAX]
        neg_paths = list(neg_paths)[:N_CAV_MAX]
        if len(pos_paths) < N_CAV_MIN or len(neg_paths) < N_CAV_MIN:
            print(
                f"  [{c_i + 1:>3d}/{len(concepts)}] SKIP {concept}: "
                f"pos={len(pos_paths)} neg={len(neg_paths)} (< {N_CAV_MIN})",
                flush=True,
            )
            continue

        pos_feats = _extract_features(pos_paths, feature_extractor, preprocess)
        neg_feats = _extract_features(neg_paths, feature_extractor, preprocess)
        X = np.concatenate([pos_feats, neg_feats], axis=0)
        y = np.array([1] * len(pos_feats) + [0] * len(neg_feats))

        clf = LinearSVC(C=CAV_C, max_iter=2000, dual="auto", random_state=SEED)
        try:
            clf.fit(X, y)
        except Exception as e:
            print(f"  [{c_i + 1:>3d}] SKIP {concept}: LinearSVC failed ({e})", flush=True)
            continue

        w = clf.coef_[0]
        norm = np.linalg.norm(w)
        if norm < 1e-10:
            print(f"  [{c_i + 1:>3d}] SKIP {concept}: zero-norm CAV", flush=True)
            continue
        cav_directions[concept] = w / norm
        print(
            f"  [{c_i + 1:>3d}/{len(concepts)}] {concept:<25s} "
            f"CAV fitted  pos={len(pos_feats)} neg={len(neg_feats)}",
            flush=True,
        )

    concepts_with_cav = [c for c in concepts if c in cav_directions]
    print(f"\nCAVs fitted: {len(concepts_with_cav)} / {len(concepts)}", flush=True)

    cav_matrix = np.stack([cav_directions[c] for c in concepts_with_cav], axis=0)  # (n_cavs, 512)

    # --- Step 2: Encode COCO train images ---
    print("\n=== Encoding COCO train2017 ===", flush=True)
    train_pairs = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="train")
    train_paths = [p for p, _ in train_pairs]
    print(f"COCO train2017 images: {len(train_paths)}", flush=True)

    train_feats  = _extract_features(train_paths, feature_extractor, preprocess, label="train-feats")
    train_proj   = train_feats @ cav_matrix.T   # (N_train, n_cavs)

    print("\n=== Getting native model logits (train) ===", flush=True)
    train_logits = _native_logits(train_paths, native_model, preprocess)   # (N_train, 20)

    # Encode COCO val images for fidelity reporting
    print("\n=== Encoding COCO val2017 (fidelity check) ===", flush=True)
    val_pairs  = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="val")
    val_paths  = [p for p, _ in val_pairs]
    val_feats  = _extract_features(val_paths, feature_extractor, preprocess, label="val-feats")
    val_proj   = val_feats @ cav_matrix.T
    val_logits = _native_logits(val_paths, native_model, preprocess)

    # --- Step 3: Fit binary linear probe per class ---
    print("\n=== Fitting linear probes ===", flush=True)
    weight_rows: list[tuple[str, str, float]] = []
    scores_for_rho: dict[tuple[str, str], float] = {}

    for cls_idx, cls_name in enumerate(COCO_STUFF_20_CLASSES):
        train_surr = (train_logits[:, cls_idx] > 0).astype(int)
        val_surr   = (val_logits[:, cls_idx] > 0).astype(int)
        n_pos_train = train_surr.sum()

        if n_pos_train < 10 or n_pos_train > len(train_surr) - 10:
            print(f"  [{cls_name}] SKIP: degenerate distribution ({n_pos_train}/{len(train_surr)} positive)", flush=True)
            continue

        probe = SGDClassifier(
            loss="log_loss",
            alpha=PROBE_LAM,
            max_iter=1000,
            class_weight="balanced",
            random_state=SEED,
        )
        probe.fit(train_proj, train_surr)

        train_fid = 100.0 * (probe.predict(train_proj) == train_surr).mean()
        val_fid   = 100.0 * (probe.predict(val_proj)   == val_surr).mean()
        print(
            f"  [{cls_name:<15s}] train_fidelity={train_fid:.1f}%  val_fidelity={val_fid:.1f}%",
            flush=True,
        )

        coef = probe.coef_[0]   # (n_cavs,)
        for col, concept in enumerate(concepts_with_cav):
            w = float(coef[col])
            weight_rows.append((concept, cls_name, w))
            scores_for_rho[(concept, cls_name)] = w

    # --- Save weights CSV ---
    out_path = RESULTS_DIR / "pcbm_resnet_concept_weights.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "class_name", "weight"])
        writer.writerows(weight_rows)
    print(f"\nSaved {len(weight_rows)} rows → {out_path}", flush=True)

    # --- Evaluate vs. faithfulness ---
    if FAITH_CSV.exists():
        faith = _load_faithfulness(FAITH_CSV)
        mean_rho_all, mean_rho_co = _eval_rho(scores_for_rho, faith)
        print(f"\nSpearman rho vs. faithfulness (PCBM-resnet):", flush=True)
        print(f"  all concepts:    {mean_rho_all:+.3f}", flush=True)
        print(f"  co-occur subset: {mean_rho_co:+.3f}", flush=True)
    else:
        print(f"\nFaithfulness CSV not found ({FAITH_CSV}) -- skipping rho evaluation", flush=True)


if __name__ == "__main__":
    main()
