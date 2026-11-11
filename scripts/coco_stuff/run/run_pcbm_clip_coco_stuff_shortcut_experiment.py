"""PCBM-CLIP (zero-shot) attribution across 11 shortcut-rate COCO-Stuff models.

Concept vectors = CLIP RN50 text embeddings (zero-shot, model-independent).
Train-split images are encoded with CLIP ONCE and cached. For each of 11 models,
only surrogate labels change -> probe is refit per model.

Output: results/coco_stuff/shortcut/pcbm_clip_shortcut_weights.csv
  rate_pct, concept_name, class_name, weight
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from sklearn.linear_model import SGDClassifier

from cards.concepts.prompts import build_concept_query
from cards.data.coco_stuff import COCO_STUFF_20_CLASSES
from cards.data.datasets import list_broden_concepts
from cards.pipeline import instantiate_encoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BRODEN_ROOT = Path("/scratch/rnair21/LLMConceptEditing_old/broden_concepts")
RESULTS_DIR = Path("results/coco_stuff")
OUT_DIR     = Path("results/coco_stuff/shortcut")
CKPT_DIR    = Path("trained_models_new/coco_stuff/shortcut")
SPLIT_FILE  = OUT_DIR / "train_test_split.json"
COUNTS_CSV  = RESULTS_DIR / "coco_stuff_faithfulness_detection_counts.csv"
EMB_CACHE   = OUT_DIR / "pcbm_clip_train_split_embeddings.pt"

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
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEED           = 42
BATCH_SIZE     = 128
RATES_PCT      = list(range(0, 101, 10))
MIN_FAITH_IMGS = 100
PROBE_LAM      = 1e-5


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


def _encode_batched(encoder, paths: list[Path]) -> torch.Tensor:
    chunks = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch = [Image.open(p).convert("RGB") for p in paths[start: start + BATCH_SIZE]]
        chunks.append(encoder.encode_images(batch).cpu())
        if (start // BATCH_SIZE) % 50 == 0:
            print(f"    encoded {start + len(batch)}/{len(paths)}", flush=True)
    return torch.cat(chunks, dim=0)


def _native_logits_batched(paths: list[Path], model, preprocess) -> np.ndarray:
    from torchvision import transforms
    chunks = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch = torch.stack([
            preprocess(Image.open(p).convert("RGB")) for p in paths[start: start + BATCH_SIZE]
        ]).to(DEVICE)
        with torch.no_grad():
            chunks.append(model(batch).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _load_shortcut_model(rate_pct: int):
    import torch.nn as nn
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(COCO_STUFF_20_CLASSES))
    ckpt = CKPT_DIR / f"resnet18_coco_stuff_shortcut_{rate_pct}.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(DEVICE).eval()


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

    # --- CLIP encoder ---
    clip_cfg = OmegaConf.create({
        "encoder": {
            "name": "clip",
            "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
            "model_name": "RN50",
            "pretrained": "openai",
            "device": DEVICE,
        },
        "device": DEVICE,
    })
    encoder = instantiate_encoder(clip_cfg)

    # --- Concept text vectors (once) ---
    concept_vectors = torch.stack([
        build_concept_query(c, encoder) for c in concepts
    ]).to(DEVICE)
    concept_norms_sq = (concept_vectors ** 2).sum(dim=1)

    # --- Encode train-split with CLIP (cached) ---
    if EMB_CACHE.exists():
        print(f"Loading cached CLIP train-split embeddings from {EMB_CACHE}...", flush=True)
        train_emb = torch.load(EMB_CACHE).to(DEVICE)
    else:
        print(f"Encoding {len(train_paths)} train-split images with CLIP...", flush=True)
        train_emb = _encode_batched(encoder, train_paths).to(DEVICE)
        torch.save(train_emb.cpu(), EMB_CACHE)
        print(f"Cached -> {EMB_CACHE}", flush=True)

    # Concept projections (fixed for all models)
    train_proj = ((train_emb @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)).detach().cpu().numpy()
    print(f"Projection shape: {train_proj.shape}", flush=True)

    from cards.models.coco_stuff_black_box import _coco_stuff_preprocess
    preprocess = _coco_stuff_preprocess()

    all_rows: list[dict] = []
    scores_by_rate: dict[int, float] = {}

    for rate_pct in RATES_PCT:
        print(f"\n=== rate={rate_pct}% ===", flush=True)
        model = _load_shortcut_model(rate_pct)

        train_logits = _native_logits_batched(train_paths, model, preprocess)

        abs_weights: list[float] = []
        for cls_idx, cls_name in enumerate(COCO_STUFF_20_CLASSES):
            surr  = (train_logits[:, cls_idx] > 0).astype(int)
            n_pos = surr.sum()
            if n_pos < 10 or n_pos > len(surr) - 10:
                continue
            probe = SGDClassifier(
                loss="log_loss", alpha=PROBE_LAM, max_iter=1000,
                class_weight="balanced", random_state=SEED,
            )
            probe.fit(train_proj, surr)
            coef = probe.coef_[0]
            for col, concept_name in enumerate(concepts):
                w = float(coef[col])
                all_rows.append({"rate_pct": rate_pct, "concept_name": concept_name,
                                 "class_name": cls_name, "weight": w})
                abs_weights.append(abs(w))

        mean_abs = float(np.mean(abs_weights)) if abs_weights else 0.0
        scores_by_rate[rate_pct] = mean_abs
        print(f"  mean |weight|: {mean_abs:.4f}", flush=True)

    out_path = OUT_DIR / "pcbm_clip_shortcut_weights.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rate_pct", "concept_name", "class_name", "weight"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows -> {out_path}", flush=True)

    print("\n=== PCBM-CLIP mean |weight| per rate ===", flush=True)
    for rate_pct in RATES_PCT:
        print(f"  rate={rate_pct:>3d}%: {scores_by_rate.get(rate_pct, float('nan')):.4f}", flush=True)


if __name__ == "__main__":
    main()
