"""PCBM-CLIP-resnet50 zero-shot baseline for COCO-Stuff.

Mirrors scripts/cub/run/train_pcbm_clip_concepts_cub.py adapted for the
143 Broden concepts × 20 COCO-Stuff co-occurrence-bias classes.

Zero-shot: each concept vector c_i is the CLIP text encoder's embedding
of the concept name (via build_concept_query, same templates CARDS uses).
No image exemplars or SVM training required. Both image embeddings f(x)
and concept vectors c_i live in the shared CLIP space.

Projection formula (Yuksekgonul et al. 2023, Eq. Section 2):
  f_C(x)[i] = <f(x), c_i> / ||c_i||^2
Since OpenClipEncoder.encode_text L2-normalizes every c_i, ||c_i||^2=1,
this simplifies to a plain dot product.

Surrogate framing: linear probe fitted against ResNet18 coco_stuff model's
own predictions (logit > 0 per class), NOT ground-truth labels -- keeping
comparison with CARDS and PCBM-resnet fair (all explain the same model).

Outputs (results/coco_stuff/):
  pcbm_clip_concept_weights.csv  -- concept_name, class_name, weight

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
from omegaconf import OmegaConf
from PIL import Image
from scipy import stats
from sklearn.linear_model import SGDClassifier

from cards.concepts.prompts import build_concept_query
from cards.data.coco_stuff import CO_OCCUR_BRODEN_CONCEPTS, COCO_STUFF_20_CLASSES
from cards.data.datasets import list_broden_concepts, load_coco_stuff
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COCO_ROOT   = Path("/scratch/rnair21/LLMConceptEditing_old/coco")
BRODEN_ROOT = Path("/scratch/rnair21/LLMConceptEditing_old/broden_concepts")
FINAL_DIR = Path("/data/hkerner/btokas/CARDS/results/coco_stuff")
RESULTS_DIR = Path("results/coco_stuff/")
FAITH_CSV   = RESULTS_DIR / "coco_stuff_faithfulness.csv"
COUNTS_CSV  = RESULTS_DIR / "coco_stuff_faithfulness_detection_counts.csv"
EMB_CACHE   = RESULTS_DIR / "pcbm_clip_rn50_coco_train_embeddings.pt"
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
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42
BATCH_SIZE  = 128
PROBE_LAM   = 1e-5   # small lam: cosine similarities are tightly bounded
MIN_SAMPLES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_batched(encoder, paths: list[Path]) -> torch.Tensor:
    chunks = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch = [Image.open(p).convert("RGB") for p in paths[start : start + BATCH_SIZE]]
        chunks.append(encoder.encode_images(batch).cpu())
        if (start // BATCH_SIZE) % 50 == 0:
            print(f"  encoded {start + len(batch)}/{len(paths)}", flush=True)
    return torch.cat(chunks, dim=0)


def _native_logits_batched(
    paths: list[Path],
    native_model: torch.nn.Module,
    preprocess,
) -> np.ndarray:
    logits = []
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

    # --- CLIP encoder (RN50/openai) ---
    print("Loading CLIP-RN50 (openai)...", flush=True)
    clip_cfg = OmegaConf.create({
        "encoder": {
            "name": "clip_rn50",
            "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
            "model_name": "RN50",
            "pretrained": "openai",
            "device": DEVICE,
        },
        "device": DEVICE,
    })
    clip_encoder = instantiate_encoder(clip_cfg)

    # --- Broden concepts + text-based concept vectors ---
    balanced = _load_balanced_concepts(COUNTS_CSV, MIN_FAITH_IMAGES)
    concepts = [c for c in list_broden_concepts(BRODEN_ROOT)
                if c not in BRODEN_DROPPED_CONCEPTS and c in balanced]
    print(f"Concepts: {len(concepts)} (>= {MIN_FAITH_IMAGES} faithfulness images)  Building concept vectors from CLIP text encoder...", flush=True)

    concept_vectors = torch.stack([
        build_concept_query(c, clip_encoder) for c in concepts
    ]).to(DEVICE)  # (n_concepts, dim), L2-normalized
    concept_norms_sq = (concept_vectors ** 2).sum(dim=1)
    print(
        f"||c_i||^2 range: {concept_norms_sq.min().item():.6f} - {concept_norms_sq.max().item():.6f}"
        f" (should be ~1.0)",
        flush=True,
    )

    # --- Encode COCO train2017 with CLIP image encoder (cached) ---
    train_pairs = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="train")
    train_paths = [p for p, _ in train_pairs]

    if EMB_CACHE.exists():
        print(f"\nLoading cached CLIP-RN50 train embeddings from {EMB_CACHE}...", flush=True)
        train_emb = torch.load(EMB_CACHE).to(DEVICE)
    else:
        print(f"\nEncoding {len(train_paths)} COCO train2017 images with CLIP-RN50...", flush=True)
        train_emb = _encode_batched(clip_encoder, train_paths).to(DEVICE)
        torch.save(train_emb.cpu(), EMB_CACHE)
        print(f"Cached → {EMB_CACHE}", flush=True)
    print(f"train embeddings: {train_emb.shape}", flush=True)

    # --- Concept projections: f_C(x)[i] = <f(x), c_i> / ||c_i||^2 ---
    train_proj = (train_emb @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)
    train_proj = train_proj.detach().cpu().numpy()   # (N_train, n_concepts)
    print(
        f"Projection scale: mean={train_proj.mean():.4f} std={train_proj.std():.4f} "
        f"range=[{train_proj.min():.4f}, {train_proj.max():.4f}]",
        flush=True,
    )

    # --- Native model logits for surrogate labels ---
    print("\nComputing native ConvNext_T logits on train images...", flush=True)
    coco_spec    = BACKBONES["coco_stuff_convnext_tiny"]
    native_model = coco_spec.load_native().to(DEVICE).eval()
    preprocess   = coco_spec.preprocess
    train_logits = _native_logits_batched(train_paths, native_model, preprocess)  # (N_train, 20)

    # Also encode val for fidelity check
    print("\nEncoding COCO val2017 for fidelity check...", flush=True)
    val_pairs  = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="val")
    val_paths  = [p for p, _ in val_pairs]
    val_emb    = _encode_batched(clip_encoder, val_paths).to(DEVICE)
    val_proj   = ((val_emb @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)).detach().cpu().numpy()
    val_logits = _native_logits_batched(val_paths, native_model, preprocess)

    # --- Fit binary linear probe per class ---
    print("\n=== Fitting linear probes ===", flush=True)
    weight_rows: list[tuple[str, str, float]] = []
    scores_for_rho: dict[tuple[str, str], float] = {}

    for cls_idx, cls_name in enumerate(COCO_STUFF_20_CLASSES):
        train_surr = (train_logits[:, cls_idx] > 0).astype(int)
        val_surr   = (val_logits[:, cls_idx] > 0).astype(int)
        n_pos = train_surr.sum()
        if n_pos < 10 or n_pos > len(train_surr) - 10:
            print(f"  [{cls_name}] SKIP: degenerate ({n_pos}/{len(train_surr)} positive)", flush=True)
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

        coef = probe.coef_[0]   # (n_concepts,)
        for col, concept in enumerate(concepts):
            w = float(coef[col])
            weight_rows.append((concept, cls_name, w))
            scores_for_rho[(concept, cls_name)] = w

    # --- Save ---
    out_path = FINAL_DIR / "pcbm_clip_convnext_concept_weights.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "class_name", "weight"])
        writer.writerows(weight_rows)
    print(f"\nSaved {len(weight_rows)} rows → {out_path}", flush=True)

    # --- Evaluate vs. faithfulness ---
    if FAITH_CSV.exists():
        faith = _load_faithfulness(FAITH_CSV)
        mean_rho_all, mean_rho_co = _eval_rho(scores_for_rho, faith)
        print(f"\nSpearman rho vs. faithfulness (PCBM-CLIP-RN50 zero-shot):", flush=True)
        print(f"  all concepts:    {mean_rho_all:+.3f}", flush=True)
        print(f"  co-occur subset: {mean_rho_co:+.3f}", flush=True)
    else:
        print(f"\nFaithfulness CSV not found ({FAITH_CSV}) -- skipping", flush=True)


if __name__ == "__main__":
    main()

