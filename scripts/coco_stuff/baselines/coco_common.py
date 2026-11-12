"""Shared pieces for the COCO-Stuff ConceptSAE / ConceptSHAP / EAC baselines. The black box is chosen with env
CARDS_ARCH in {resnet18 (default), vit, convnext} (20-class BCE head each), mirroring scripts/celeba/baselines/black_boxes.py.

Image-level concept labels come from the ground-truth pipeline itself: an image is concept-positive when Grounding DINO
detected the concept in it (a row exists in coco_stuff_faithfulness.csv). The pool is every COCO train2017 image with
>=1 of the 20 target classes (the pool the ground truth was computed on), so a concept's positives are exactly the
images its faithfulness ground truth was measured on. Run everything from the repo root."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
from cards.data.coco_stuff import COCO_STUFF_20_CLASSES
from cards.data.datasets import list_broden_concepts, load_coco_stuff
from cards.models.backbones import BACKBONES

COCO_ROOT = Path(os.environ.get("COCO_ROOT", "/scratch/rnair21/LLMConceptEditing_old/coco"))
BRODEN_ROOT = Path(os.environ.get("BRODEN_ROOT", "/scratch/rnair21/LLMConceptEditing_old/broden_concepts"))
RESULTS = Path(os.environ.get("CARDS_RESULTS_DIR", "results/coco_stuff"))
FAITH_CSV = RESULTS / "coco_stuff_faithfulness.csv"
COUNTS_CSV = RESULTS / "coco_stuff_faithfulness_detection_counts.csv"
ARCH = os.environ.get("CARDS_ARCH", "resnet18")
BACKBONE = {"resnet18": "coco_stuff", "vit": "coco_stuff_vit_b_16", "convnext": "coco_stuff_convnext_tiny"}
if ARCH not in BACKBONE:
    raise ValueError(f"CARDS_ARCH must be one of {sorted(BACKBONE)}, got {ARCH!r}")
SUFFIX = {"resnet18": "", "vit": "_vit", "convnext": "_convnext"}[ARCH]
OUT_DIR = RESULTS / "baselines"
FEAT_CACHE = Path(os.environ.get("CARDS_FEAT_CACHE", str(OUT_DIR / f"feat_cache{SUFFIX}")))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_FAITH_IMAGES = 100
DROPPED = {
    "air_conditioner", "apron", "awning", "bathroom_s", "beak", "bedroom_s", "blotchy", "bumper", "can", "candlestick",
    "cap", "clock", "dining_room_s", "drinking_glass", "eye", "eyebrow", "faucet", "figurine", "handle", "inside_arm",
    "jar", "knob", "loudspeaker", "mouse", "mouth", "outside_arm", "street_s",
}
CLASSES = COCO_STUFF_20_CLASSES


def concept_list() -> list[str]:
    """Same concept set every other COCO method is scored on (>= MIN_FAITH_IMAGES detected images)."""
    ok = set()
    with open(COUNTS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["n_images"]) >= MIN_FAITH_IMAGES:
                ok.add(r["concept_name"])
    return [c for c in list_broden_concepts(BRODEN_ROOT) if c not in DROPPED and c in ok]


def pool_paths() -> list[str]:
    return [str(p) for p, lbl in load_coco_stuff(COCO_ROOT, CLASSES, split="train") if lbl >= 0]


def concept_labels(paths: list[str], concepts: list[str]) -> np.ndarray:
    """(N, n_concepts) bool: concept detected in image. Streams the 2.6 GB faithfulness CSV once, cached next to features."""
    cache = OUT_DIR / "concept_labels.npz"  # model-independent: shared by every architecture
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["concepts"]) == concepts and len(z["paths"]) == len(paths):
            return z["labels"]
    pidx = {p: i for i, p in enumerate(paths)}; cidx = {c: i for i, c in enumerate(concepts)}
    Y = np.zeros((len(paths), len(concepts)), bool)
    for chunk in pd.read_csv(FAITH_CSV, usecols=["concept_name", "image_path"], chunksize=5_000_000):
        chunk = chunk.drop_duplicates()
        for c, p in zip(chunk["concept_name"], chunk["image_path"]):
            if c in cidx and p in pidx:
                Y[pidx[p], cidx[c]] = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, labels=Y, concepts=np.array(concepts), paths=np.array(paths))
    return Y


def load_black_box():
    """(native 20-way model for CARDS_ARCH, its preprocess)."""
    spec = BACKBONES[BACKBONE[ARCH]]
    return spec.load_native().to(DEVICE).eval(), spec.preprocess


def head(m: torch.nn.Module) -> torch.nn.Linear:
    """The final 20-way Linear."""
    return {"resnet18": lambda: m.fc, "vit": lambda: m.heads.head, "convnext": lambda: m.classifier[2]}[ARCH]()


def feats(m: torch.nn.Module, x: torch.Tensor):
    """(pooled (N,d), maps (N,d,7,7)); pooled is exactly the input of head(m).
      resnet18: layer4 maps; pooled = spatial mean.
      vit:      final encoder tokens (after encoder.ln); pooled = CLS token; maps = the 14x14 patch tokens average-pooled 2x2 to 7x7.
      convnext: features maps; pooled = avgpool -> LayerNorm2d -> flatten.
    All three give ConceptSHAP a 49-patch grid."""
    if ARCH == "resnet18":
        f = m.layer4(m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x))))))))
        return f.mean((2, 3)), f
    if ARCH == "vit":
        t = m.encoder(torch.cat([m.class_token.expand(len(x), -1, -1), m._process_input(x)], 1))
        g = t[:, 1:].transpose(1, 2).reshape(len(x), m.hidden_dim, 14, 14)
        return t[:, 0], torch.nn.functional.avg_pool2d(g, 2)
    f = m.features(x)
    return m.classifier[1](m.classifier[0](m.avgpool(f))), f
