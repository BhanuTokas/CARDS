"""TCAV model-variant experiment for COCO-Stuff.

Computes aggregated TCAV sign-count scores for 3 backbone architectures:
  coco_stuff              ResNet18     hook: layer4                 [N,512,H,W]
  coco_stuff_vit_b_16     ViT-B/16     hook: encoder_layer_11      [N,seq,768]
  coco_stuff_convnext_tiny ConvNeXt-Tiny hook: features.7          [N,768,H,W]

Uses PyTorch backward hooks (no captum) to handle different hook-layer shapes:
  CNN  (4-D): mean-pool spatial dims  → [N, C]
  ViT  (3-D): CLS-token index 0       → [N, 768]

CAV fitting: LinearSVC(C=0.1) on pooled hook-layer activations:
  positives = up to N_CAV_POS Broden exemplars for the concept
  negatives = N_CAV_NEG random COCO train2017 images

Sign count: fraction of N_VAL_SAMPLES val2017 images where
  sign(∇_{h_l} f_k · CAV) > 0, for each of the 20 classes.

Architecture selected via BACKBONE_NAME env var (default: "coco_stuff").

Balanced concepts: same 103-concept filter as run_cards_coco_stuff_model_variants.py
(n_images >= 100 in coco_stuff_faithfulness_detection_counts.csv).

Outputs:
  results/coco_stuff/model_variants/{backbone_name}_tcav_scores.csv
    concept_name, class_name, mean_sign_count
  Prints Spearman rho vs. faithfulness (black_mask + black_bbox).

Usage:
    cd /scratch/rnair21/CARDS
    BACKBONE_NAME=coco_stuff_vit_b_16 \\
        /scratch/rnair21/envs/cards_env/bin/python \\
        scripts/coco_stuff/run/run_tcav_coco_stuff_model_variants.py
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
MIN_FAITH_IMGS  = 100
MIN_SAMPLES     = 3
N_CAV_POS       = 40    # Broden positives for CAV fitting
N_CAV_NEG       = 25    # random train2017 negatives for CAV fitting
N_VAL_SAMPLES   = 40    # val2017 images for sign-count computation
STRATEGIES      = ("black_mask", "black_bbox")


# ---------------------------------------------------------------------------
# Pooling helper (architecture-aware)
# ---------------------------------------------------------------------------

def _pool(act: torch.Tensor, backbone_name: str) -> torch.Tensor:
    """Pool hook-layer output to [N, D] regardless of architecture.

    ViT encoder layer  [N, seq, 768] → CLS token [N, 768]
    CNN feature layer  [N, C, H, W]  → spatial mean [N, C]
    """
    if "vit" in backbone_name:
        return act[:, 0, :]
    if act.dim() == 4:
        return act.mean(dim=[2, 3])
    return act


# ---------------------------------------------------------------------------
# Layer activation extraction (forward hook, no grad)
# ---------------------------------------------------------------------------

def _layer_activations(
    images: list[Image.Image],
    model: torch.nn.Module,
    hook_layer: torch.nn.Module,
    backbone_name: str,
    preprocess,
    device: str,
) -> np.ndarray:
    """Pooled hook-layer activations [N, D] for a list of PIL images."""
    if not images:
        return np.zeros((0,), dtype=np.float32)
    store: list[torch.Tensor] = []

    def hook(m, i, o):
        store.append(_pool(o.detach(), backbone_name).cpu())

    handle = hook_layer.register_forward_hook(hook)
    tensors = torch.stack([preprocess(img) for img in images]).to(device)
    with torch.no_grad():
        model(tensors)
    handle.remove()
    return store[0].numpy()   # [N, D]


# ---------------------------------------------------------------------------
# CAV fitting
# ---------------------------------------------------------------------------

def _fit_cav(
    pos_images: list[Image.Image],
    neg_images: list[Image.Image],
    model: torch.nn.Module,
    hook_layer: torch.nn.Module,
    backbone_name: str,
    preprocess,
    device: str,
) -> np.ndarray | None:
    """LinearSVC on hook-layer activations; return unit CAV [D] or None."""
    pos_feats = _layer_activations(pos_images, model, hook_layer, backbone_name, preprocess, device)
    neg_feats = _layer_activations(neg_images, model, hook_layer, backbone_name, preprocess, device)
    if len(pos_feats) < 2 or len(neg_feats) < 2:
        return None
    X = np.concatenate([pos_feats, neg_feats], axis=0)
    y = np.array([1] * len(pos_feats) + [0] * len(neg_feats))
    try:
        clf = LinearSVC(C=0.1, max_iter=2000, dual="auto", random_state=SEED)
        clf.fit(X, y)
        w = clf.coef_[0]
        norm = np.linalg.norm(w)
        return (w / norm) if norm > 1e-8 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# TCAV sign-count computation (backward hook)
# ---------------------------------------------------------------------------

def _tcav_sign_count(
    val_tensors: torch.Tensor,
    model: torch.nn.Module,
    hook_layer: torch.nn.Module,
    backbone_name: str,
    cav_weights: np.ndarray,
    device: str,
) -> np.ndarray:
    """Per-class sign count [n_cls]: fraction of val images with positive
    directional derivative toward the concept CAV direction."""
    n_cls = len(COCO_STUFF_20_CLASSES)
    cav_t = torch.tensor(cav_weights, dtype=torch.float32).to(device)
    h_store: list[torch.Tensor] = []

    def fwd_hook(m, i, o):
        o.retain_grad()
        h_store.clear()
        h_store.append(o)

    handle = hook_layer.register_forward_hook(fwd_hook)
    with torch.enable_grad():
        logits = model(val_tensors)   # [N, n_cls]
    handle.remove()

    h_l = h_store[0]
    sign_counts = np.zeros(n_cls, dtype=np.float32)

    for k in range(n_cls):
        if h_l.grad is not None:
            h_l.grad.zero_()
        logits[:, k].sum().backward(retain_graph=(k < n_cls - 1))
        if h_l.grad is None:
            continue
        grad_pooled = _pool(h_l.grad, backbone_name)   # [N, D]
        signs = torch.sign(grad_pooled @ cav_t)        # [N]
        sign_counts[k] = float((signs > 0).float().mean().item())

    return sign_counts   # [n_cls], values in [0, 1]


# ---------------------------------------------------------------------------
# Faithfulness + evaluation (same protocol as model_variants CARDS)
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
    print(f"Backbone: {backbone_name}  embed_dim={spec.embed_dim}  hook={spec.hook_layer}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Model + hook layer ---
    model      = spec.load_native().to(DEVICE).eval()
    preprocess = spec.preprocess
    hook_layer = model.get_submodule(spec.hook_layer)
    print(f"Hook layer: {hook_layer.__class__.__name__}  device={DEVICE}", flush=True)

    # --- Balanced concepts ---
    balanced = _load_balanced_concepts(COUNTS_CSV, MIN_FAITH_IMGS)
    concepts = [
        c for c in list_broden_concepts(BRODEN_ROOT)
        if c not in BRODEN_DROPPED_CONCEPTS and c in balanced
    ]
    print(f"Concepts: {len(concepts)}", flush=True)

    # --- Random negatives from COCO train2017 ---
    rng = np.random.default_rng(SEED)
    train_pairs   = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="train")
    all_train     = [p for p, _ in train_pairs]
    perm          = rng.permutation(len(all_train))
    neg_pool      = [all_train[i] for i in perm[: len(concepts) * N_CAV_NEG]]
    print(f"Negative pool: {len(neg_pool)} train2017 images", flush=True)

    # --- Validation images for sign-count ---
    val_pairs  = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="val")
    val_paths  = [p for p, _ in val_pairs]
    rng.shuffle(val_paths)
    val_paths  = val_paths[:N_VAL_SAMPLES]
    val_tensors = torch.stack([
        preprocess(Image.open(p).convert("RGB")) for p in val_paths
    ]).to(DEVICE)
    print(f"Val images: {len(val_paths)}", flush=True)

    # --- Per-concept TCAV scoring ---
    all_scores: dict[tuple[str, str], float] = {}
    score_rows: list[tuple] = []

    for c_i, concept_name in enumerate(concepts):
        try:
            pos_paths_all, _ = load_broden(BRODEN_ROOT, concept_name)
        except ValueError:
            pos_paths_all = []

        rng_c = np.random.default_rng(SEED + c_i)
        pos_paths = sorted(pos_paths_all)
        rng_c.shuffle(pos_paths)
        pos_paths = pos_paths[:N_CAV_POS]

        neg_start  = c_i * N_CAV_NEG
        neg_sample = neg_pool[neg_start : neg_start + N_CAV_NEG]

        pos_imgs = [Image.open(p).convert("RGB") for p in pos_paths]
        neg_imgs = [Image.open(p).convert("RGB") for p in neg_sample]

        cav = _fit_cav(pos_imgs, neg_imgs, model, hook_layer, backbone_name, preprocess, DEVICE)

        if cav is None:
            sign_counts = np.full(len(COCO_STUFF_20_CLASSES), 0.5, dtype=np.float32)
            status = "skip"
        else:
            sign_counts = _tcav_sign_count(val_tensors, model, hook_layer, backbone_name, cav, DEVICE)
            status = "ok"

        for cls_i, cls_name in enumerate(COCO_STUFF_20_CLASSES):
            sc = float(sign_counts[cls_i])
            all_scores[(concept_name, cls_name)] = sc
            score_rows.append((concept_name, cls_name, sc))

        if (c_i + 1) % 10 == 0 or c_i == 0:
            print(
                f"  [{c_i + 1:>3d}/{len(concepts)}] {concept_name:<25s} "
                f"cav={status}  mean_sign={float(sign_counts.mean()):.3f}",
                flush=True,
            )

    # --- Save scores ---
    out_path = OUT_DIR / f"{backbone_name}_tcav_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "class_name", "mean_sign_count"])
        writer.writerows(score_rows)
    print(f"\nSaved {len(score_rows)} rows -> {out_path}", flush=True)

    # --- Evaluate rho ---
    print("\n=== Faithfulness evaluation (TCAV) ===", flush=True)
    summary_rows: list[dict] = []
    for strategy in STRATEGIES:
        faith, image_counts = _load_faithfulness(FAITH_CSV, strategy)
        rows, mean_rho, mean_sign, pr, pp, pn = evaluate(all_scores, faith, image_counts)
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

    summary_path = OUT_DIR / f"{backbone_name}_tcav_summary.csv"
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
