"""PCBM surrogate fit on the official-train classifier, both tasks,
prompted directly ("Do we also have results of PCBM and TCAV on this
version of ground truth?"). Mirrors `train_pcbm_surrogate_celeba_male.
py`'s own design (both known bug fixes: float32 cast before
`set_weights`, `weight[0,:]` not `.argmax` for the binary decision
function) but sources embeddings/surrogate-label images from standard
CelebA's OFFICIAL train/val partitions (162,770 / 19,867 images)
instead of CelebAMask-HQ's train_hq/val_hq -- matching what this
classifier actually trained on.

**Generalized to any official-train backbone** (`CARDS_BACKBONE_NAME`
env var, default `celeba_official_train_attractive_male`), prompted
directly ("The full pipeline with PCBM variants" for ViT-B/16 and
ConvNeXt-Tiny) -- already used `spec.feature_extractor(native_model)`
generically, so this only needed the backbone-name lookup, CAV-bank
path, and output paths keyed by BACKBONE_NAME (matching `fit_celeba_
official_train_cavs.py`'s own filename convention), so each
architecture's surrogate never overwrites another's.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.models.backbones import BACKBONES

CELEBA_ROOT = Path(os.environ.get("CELEBA_ROOT", r"C:\Users\btokas\Projects\Datasets\CelebA\celeba"))
BACKBONE_NAME = os.environ.get("CARDS_BACKBONE_NAME", "celeba_official_train_attractive_male")
CONCEPT_BANK_PATH = (Path(os.environ.get("CARDS_TRAINED_CONCEPTS_DIR", "trained_concepts_new/celeba_full")) /
                      BACKBONE_NAME / f"celeba_full_{BACKBONE_NAME}_0.1_100.pkl")
OUT_DIR = Path(os.environ.get("CARDS_TRAINED_MODELS_DIR", "trained_models_new/celeba_full")) / BACKBONE_NAME
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
TASKS = ["Attractive", "Male"]
TASK_SLICES: dict[str, slice] = {"Attractive": slice(0, 2), "Male": slice(2, 4)}


def load_attr_names(path: Path) -> list[str]:
    return Path(path).read_text().splitlines()[1].split()


def load_attr_labels(path: Path) -> dict[str, np.ndarray]:
    lines = Path(path).read_text().splitlines()
    result: dict[str, np.ndarray] = {}
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split()
        result[parts[0]] = np.array([v == "1" for v in parts[1:]], dtype=bool)
    return result


def embed_images(filenames, feature_extractor, preprocess, device, label):
    embeddings = []
    for start in range(0, len(filenames), BATCH_SIZE):
        batch_files = filenames[start : start + BATCH_SIZE]
        batch = torch.stack([
            preprocess(Image.open(CELEBA_ROOT / "img_align_celeba" / f).convert("RGB")) for f in batch_files
        ]).to(device)
        with torch.no_grad():
            emb = torch.flatten(feature_extractor(batch), 1).cpu().numpy()
        embeddings.append(emb)
        if (start // BATCH_SIZE) % 200 == 0:
            print(f"[{label}] {start + len(batch_files)}/{len(filenames)}", flush=True)
    return np.concatenate(embeddings, axis=0)


def native_task_logits(filenames, native_model, preprocess, device):
    logits = []
    for start in range(0, len(filenames), BATCH_SIZE):
        batch_files = filenames[start : start + BATCH_SIZE]
        batch = torch.stack([
            preprocess(Image.open(CELEBA_ROOT / "img_align_celeba" / f).convert("RGB")) for f in batch_files
        ]).to(device)
        with torch.no_grad():
            logits.append(native_model(batch).cpu().numpy())
    return np.concatenate(logits, axis=0)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"BACKBONE_NAME={BACKBONE_NAME}  CONCEPT_BANK_PATH={CONCEPT_BANK_PATH}  OUT_DIR={OUT_DIR}", flush=True)
    spec = BACKBONES[BACKBONE_NAME]
    native_model = spec.load_native().to(DEVICE).eval()
    feature_extractor = spec.feature_extractor(native_model).to(DEVICE).eval()
    preprocess = spec.preprocess

    print("Loading official CelebA metadata...", flush=True)
    attr_names = load_attr_names(CELEBA_ROOT / "list_attr_celeba.txt")
    attr_labels = load_attr_labels(CELEBA_ROOT / "list_attr_celeba.txt")
    partition = {}
    with open(CELEBA_ROOT / "list_eval_partition.txt") as f:
        for line in f:
            fname, part = line.split()
            partition[fname] = part
    train_files = sorted(f for f, p in partition.items() if p == "0")
    val_files = sorted(f for f, p in partition.items() if p == "1")
    print(f"{len(train_files)} train images, {len(val_files)} val images", flush=True)

    print("Loading region-crop concept bank...", flush=True)
    with open(CONCEPT_BANK_PATH, "rb") as f:
        all_concepts = pickle.load(f)
    print(f"{len(all_concepts)} concepts in bank: {sorted(all_concepts)}", flush=True)
    assert set(all_concepts) == set(GROUNDABLE_CONCEPTS), "concept bank doesn't match GROUNDABLE_CONCEPTS"

    from concepts import ConceptBank
    from models import PosthocLinearCBM
    from train_pcbm import run_linear_probe

    concept_bank = ConceptBank(all_concepts, DEVICE)

    print("\n=== computing shared embeddings + concept-margin projections (task-independent) ===", flush=True)
    train_emb = embed_images(train_files, feature_extractor, preprocess, DEVICE, "train")
    val_emb = embed_images(val_files, feature_extractor, preprocess, DEVICE, "val")

    probe_layer = PosthocLinearCBM(concept_bank, backbone_name=BACKBONE_NAME, n_classes=2).to(DEVICE)
    train_proj = probe_layer.compute_dist(torch.tensor(train_emb, device=DEVICE).float()).detach().cpu().numpy()
    val_proj = probe_layer.compute_dist(torch.tensor(val_emb, device=DEVICE).float()).detach().cpu().numpy()

    print("\n=== computing native model's task logits (for surrogate labels + true labels) ===", flush=True)
    train_native_logits = native_task_logits(train_files, native_model, preprocess, DEVICE)
    val_native_logits = native_task_logits(val_files, native_model, preprocess, DEVICE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task_name in TASKS:
        task_slice = TASK_SLICES[task_name]
        task_attr_idx = attr_names.index(task_name)
        print(f"\n########## target task: {task_name} ##########", flush=True)

        train_surrogate = train_native_logits[:, task_slice].argmax(axis=1)
        val_surrogate = val_native_logits[:, task_slice].argmax(axis=1)
        val_true = np.array([int(attr_labels[f][task_attr_idx]) for f in val_files])

        native_true_label_acc_val = (val_surrogate == val_true).mean()
        print(f"Native model's own true-label accuracy on val (informational): {native_true_label_acc_val:.4f}", flush=True)

        posthoc_layer = PosthocLinearCBM(
            concept_bank, backbone_name=BACKBONE_NAME,
            idx_to_class={0: f"not_{task_name}", 1: task_name}, n_classes=2,
        ).to(DEVICE)

        class Args:
            seed = SEED
            lam = 0.0002
            alpha = 0.99

        args = Args()
        run_info, weights, bias = run_linear_probe(args, (train_proj, train_surrogate), (val_proj, val_surrogate))
        print(f"train fidelity (agreement with native model, train slice): {run_info['train_acc']:.2f}%", flush=True)
        print(f"val fidelity (agreement with native model, held-out val slice): {run_info['test_acc']:.2f}%", flush=True)

        posthoc_layer.set_weights(weights=weights.astype(np.float32), bias=bias.astype(np.float32))

        with torch.no_grad():
            val_logits = posthoc_layer.forward_projs(torch.tensor(val_proj, device=DEVICE).float())
            pcbm_pred = (val_logits.squeeze(-1) > 0).long().cpu().numpy()
        pcbm_true_label_acc = (pcbm_pred == val_true).mean()
        pcbm_native_fidelity = (pcbm_pred == val_surrogate).mean()
        print(f"PCBM surrogate's own true-label accuracy on val: {pcbm_true_label_acc:.4f}", flush=True)
        print(f"PCBM surrogate's fidelity to native model's predictions on val: {pcbm_native_fidelity:.4f}", flush=True)

        model_path = (OUT_DIR /
                       f"pcbm_celeba_full__{BACKBONE_NAME}__{task_name.lower()}__surrogate__seed_{SEED}__linear.ckpt")
        torch.save(posthoc_layer, model_path)
        print(f"Saved surrogate PCBM to {model_path}", flush=True)
        print(posthoc_layer.analyze_classifier(k=5), flush=True)


if __name__ == "__main__":
    main()
