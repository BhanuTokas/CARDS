"""PCBM model-variant experiment for COCO-Stuff.

Fits a Post-hoc Concept Bottleneck Model surrogate for each of 3 backbone
architectures:
  coco_stuff               ResNet18      embed_dim=512
  coco_stuff_vit_b_16      ViT-B/16      embed_dim=768
  coco_stuff_convnext_tiny ConvNeXt-Tiny embed_dim=768

Pipeline (identical to train_pcbm_resnet_coco_stuff.py, now architecture-aware):
  1. Load backbone via spec.load_native(); feature extractor via spec.feature_extractor().
     torch.flatten(f, 1) handles all output shapes:
       ResNet18  [N, 512, 1, 1] → [N, 512]
       ViT-B/16  [N, 768, 1]    → [N, 768]   (CLS token from _ViTBody)
       ConvNeXt  [N, 768]       → [N, 768]
  2. For each of 103 balanced Broden concepts:
       Fit LinearSVC(C=0.1) on Broden pos + neg feature extractor activations.
       CAV = L2-normalised coef_[0].
  3. Encode all COCO train2017 images → project onto CAV matrix → (N_train, n_cav).
  4. Get native model logits on train images → binary surrogate labels (logit > 0).
  5. For each of 20 classes: fit SGDClassifier surrogate on concept projections.
  6. Save (concept_name, class_name, weight) as output CSV.

Architecture selected via BACKBONE_NAME env var (default: "coco_stuff").

Balanced concepts: n_images >= 100 in coco_stuff_faithfulness_detection_counts.csv
(same 103-concept filter as all other COCO-Stuff experiments).

Outputs:
  results/coco_stuff/model_variants/{backbone_name}_pcbm_concept_weights.csv
    concept_name, class_name, weight
  Prints Spearman rho vs. faithfulness (black_mask + black_bbox).

Usage:
    cd /scratch/rnair21/CARDS
    BACKBONE_NAME=coco_stuff_vit_b_16 \\
        /scratch/rnair21/envs/cards_env/bin/python \\
        scripts/coco_stuff/run/train_pcbm_coco_stuff_model_variants.py
"""

from __future__ import annotations

import csv
import os
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

from cards.data.coco_stuff import COCO_STUFF_20_CLASSES
from cards.data.datasets import list_broden_concepts, load_broden, load_coco_stuff
from cards.models.backbones import BACKBONES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COCO_ROOT   = Path("/scratch/rnair21/LLMConceptEditing_old/coco")
BRODEN_ROOT = Path("/scratch/rnair21/LLMConceptEditing_old/broden_concepts")
RESULTS_DIR = Path("results/coco_stuff")
OUT_DIR     = Path("results/coco_stuff/model_variants")
COUNTS_CSV  = RESULTS_DIR / "coco_stuff_faithfulness_detection_counts.csv"
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
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
SEED            = 42
BATCH_SIZE      = 64
CAV_C           = 0.1     # LinearSVC regularization
N_CAV_MAX       = 150     # max Broden exemplars per side
N_CAV_MIN       = 5       # skip if fewer pos or neg exemplars
PROBE_LAM       = 2e-4    # SGDClassifier L2 regularization
MIN_FAITH_IMGS  = 100
MIN_SAMPLES     = 3
STRATEGIES      = ("black_mask", "black_bbox")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_features(
    paths: list[Path],
    feature_extractor: torch.nn.Module,
    preprocess,
    label: str = "",
) -> np.ndarray:
    """Encode images with feature_extractor; flatten to [N, embed_dim]."""
    feats: list[np.ndarray] = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start : start + BATCH_SIZE]
        batch = torch.stack([
            preprocess(Image.open(p).convert("RGB")) for p in batch_paths
        ]).to(DEVICE)
        with torch.no_grad():
            f = feature_extractor(batch)
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
    return np.concatenate(logits, axis=0)   # [N, 20]


# ---------------------------------------------------------------------------
# Faithfulness + evaluation
# ---------------------------------------------------------------------------

def _load_balanced_concepts(counts_csv: Path, min_images: int) -> set[str]:
    balanced: set[str] = set()
    with open(counts_csv, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["n_images"]) >= min_images:
                balanced.add(row["concept_name"])
    return balanced


def _load_faithfulness(
    path: Path, strategy: str
) -> tuple[dict[tuple[str, str], list[float]], dict[str, int]]:
    col = f"delta_p_{strategy}"
    concept_imgs: dict[str, set[str]] = defaultdict(set)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            concept_imgs[row["concept_name"]].add(row["image_path"])
    valid_imgs = {
        c: set(sorted(imgs)[:MIN_FAITH_IMGS])
        for c, imgs in concept_imgs.items()
        if len(imgs) >= MIN_FAITH_IMGS
    }
    faith: dict[tuple[str, str], list[float]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            c, cl, img = row["concept_name"], row["class_name"], row["image_path"]
            if c in valid_imgs and img in valid_imgs[c]:
                faith[(c, cl)].append(float(row[col]))
    return dict(faith), {c: MIN_FAITH_IMGS for c in valid_imgs}


def _sign_pval(n_agree: int, n: int) -> float:
    from scipy.stats import binomtest
    return binomtest(n_agree, n, p=0.5, alternative="two-sided").pvalue


def evaluate(
    scores: dict[tuple[str, str], float],
    faith: dict[tuple[str, str], list[float]],
    image_counts: dict[str, int],
) -> tuple[list[dict], float, float, float, float, int]:
    valid = {c for c, n in image_counts.items() if n >= MIN_FAITH_IMGS}
    rows, class_rhos, class_signs = [], [], []
    pooled_s, pooled_g = [], []

    for cls_name in COCO_STUFF_20_CLASSES:
        s_vals, g_vals = [], []
        for key, dp_list in faith.items():
            if key[1] != cls_name or key[0] not in valid or len(dp_list) < MIN_SAMPLES:
                continue
            if key not in scores:
                continue
            s_vals.append(scores[key])
            g_vals.append(float(np.mean(dp_list)))
        pooled_s.extend(s_vals)
        pooled_g.extend(g_vals)

        n = len(s_vals)
        if n < 2:
            rows.append(dict(class_name=cls_name, n_pairs=n,
                             spearman_rho=float("nan"), spearman_p=float("nan"),
                             sign_agreement=float("nan"), sign_p=float("nan")))
            continue

        rho, rho_p = stats.spearmanr(s_vals, g_vals)
        n_agree  = int(np.sum(np.sign(np.array(s_vals)) == np.sign(np.array(g_vals))))
        sign_agr = n_agree / n
        rows.append(dict(class_name=cls_name, n_pairs=n,
                         spearman_rho=float(rho), spearman_p=float(rho_p),
                         sign_agreement=sign_agr, sign_p=_sign_pval(n_agree, n)))
        class_rhos.append(float(rho))
        class_signs.append(sign_agr)

    mean_rho  = float(np.mean(class_rhos))  if class_rhos  else float("nan")
    mean_sign = float(np.mean(class_signs)) if class_signs else float("nan")
    if len(pooled_s) >= 2:
        pr, pp = stats.spearmanr(pooled_s, pooled_g)
    else:
        pr, pp = float("nan"), float("nan")
    return rows, mean_rho, mean_sign, float(pr), float(pp), len(pooled_s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    backbone_name = os.environ.get("BACKBONE_NAME", "coco_stuff")
    if backbone_name not in BACKBONES:
        raise ValueError(f"Unknown BACKBONE_NAME={backbone_name!r}. Available: {list(BACKBONES)}")
    spec = BACKBONES[backbone_name]
    print(f"Backbone: {backbone_name}  embed_dim={spec.embed_dim}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- Backbone ---
    native_model      = spec.load_native().to(DEVICE).eval()
    feature_extractor = spec.feature_extractor(native_model).to(DEVICE).eval()
    preprocess        = spec.preprocess
    print(f"Feature extractor: {feature_extractor.__class__.__name__}  device={DEVICE}", flush=True)

    # --- Balanced concepts ---
    balanced = _load_balanced_concepts(COUNTS_CSV, MIN_FAITH_IMGS)
    concepts = [
        c for c in list_broden_concepts(BRODEN_ROOT)
        if c not in BRODEN_DROPPED_CONCEPTS and c in balanced
    ]
    print(f"Concepts: {len(concepts)} (>= {MIN_FAITH_IMGS} faithfulness images)", flush=True)

    # --- Step 1: Fit CAVs ---
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

        try:
            clf = LinearSVC(C=CAV_C, max_iter=2000, dual="auto", random_state=SEED)
            clf.fit(X, y)
        except Exception as e:
            print(f"  [{c_i + 1:>3d}] SKIP {concept}: LinearSVC failed ({e})", flush=True)
            continue

        w    = clf.coef_[0]
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

    cav_matrix = np.stack([cav_directions[c] for c in concepts_with_cav], axis=0)
    # shape: [n_cavs, embed_dim]

    # --- Step 2: Encode COCO train2017 ---
    print("\n=== Encoding COCO train2017 ===", flush=True)
    train_pairs  = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="train")
    train_paths  = [p for p, _ in train_pairs]
    print(f"COCO train2017 images: {len(train_paths)}", flush=True)

    train_feats  = _extract_features(train_paths, feature_extractor, preprocess, label="train-feats")
    train_proj   = train_feats @ cav_matrix.T   # [N_train, n_cavs]

    print("\n=== Getting native model logits (train) ===", flush=True)
    train_logits = _native_logits(train_paths, native_model, preprocess)   # [N_train, 20]

    # --- Encode COCO val2017 for fidelity check ---
    print("\n=== Encoding COCO val2017 (fidelity check) ===", flush=True)
    val_pairs  = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="val")
    val_paths  = [p for p, _ in val_pairs]
    val_feats  = _extract_features(val_paths, feature_extractor, preprocess, label="val-feats")
    val_proj   = val_feats @ cav_matrix.T
    val_logits = _native_logits(val_paths, native_model, preprocess)

    # --- Step 3: Fit binary linear probe per class ---
    print("\n=== Fitting linear probes ===", flush=True)
    weight_rows: list[tuple] = []
    scores_for_rho: dict[tuple[str, str], float] = {}

    for cls_idx, cls_name in enumerate(COCO_STUFF_20_CLASSES):
        train_surr = (train_logits[:, cls_idx] > 0).astype(int)
        val_surr   = (val_logits[:, cls_idx] > 0).astype(int)
        n_pos_train = int(train_surr.sum())

        if n_pos_train < 10 or n_pos_train > len(train_surr) - 10:
            print(
                f"  [{cls_name}] SKIP: degenerate distribution "
                f"({n_pos_train}/{len(train_surr)} positive)",
                flush=True,
            )
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
            f"  [{cls_name:<15s}] train_fid={train_fid:.1f}%  val_fid={val_fid:.1f}%",
            flush=True,
        )

        coef = probe.coef_[0]   # [n_cavs]
        for col, concept in enumerate(concepts_with_cav):
            w = float(coef[col])
            weight_rows.append((concept, cls_name, w))
            scores_for_rho[(concept, cls_name)] = w

    # --- Save weights ---
    out_path = OUT_DIR / f"{backbone_name}_pcbm_concept_weights.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "class_name", "weight"])
        writer.writerows(weight_rows)
    print(f"\nSaved {len(weight_rows)} rows -> {out_path}", flush=True)

    # --- Evaluate rho ---
    print("\n=== Faithfulness evaluation (PCBM) ===", flush=True)
    summary_rows: list[dict] = []
    for strategy in STRATEGIES:
        faith, image_counts = _load_faithfulness(FAITH_CSV, strategy)
        rows, mean_rho, mean_sign, pr, pp, pn = evaluate(scores_for_rho, faith, image_counts)
        print(
            f"  strategy={strategy:<12s}  mean_rho={mean_rho:+.4f}  "
            f"mean_sign={mean_sign:.4f}  "
            f"pooled_rho={pr:+.4f}  pooled_p={pp:.4g}  n_pairs={pn}",
            flush=True,
        )
        for r in rows:
            if not np.isnan(r["spearman_rho"]):
                print(
                    f"    [{r['class_name']:<15s}] n={r['n_pairs']:>4d}  "
                    f"rho={r['spearman_rho']:+.3f}  sign={r['sign_agreement']:.1%}",
                    flush=True,
                )
        for r in rows:
            summary_rows.append({
                "backbone": backbone_name, "strategy": strategy, **r,
                "pooled_rho": pr, "pooled_p": pp, "pooled_n": pn,
            })

    summary_path = OUT_DIR / f"{backbone_name}_pcbm_summary.csv"
    fields = ["backbone", "strategy", "class_name", "n_pairs",
              "spearman_rho", "spearman_p", "sign_agreement", "sign_p",
              "pooled_rho", "pooled_p", "pooled_n"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
