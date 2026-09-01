"""Phase 7 scale-up of the CelebA plan: fits PCBM's own linear classifier
as a SURROGATE of the native celeba_attractive_young model's own
predictions, using fit_celeba_full_cavs.py's ALL-26-concept region-crop
bank -- extends train_pcbm_surrogate_celeba_pilot.py's own 8-concept
pilot fit (v71) the same way. Logic, including both bugs that script
found and fixed (the float64-vs-float32 dtype mismatch, and the binary-
task single-decision-function-row shape gotcha), is otherwise IDENTICAL
-- see that script's own docstring for the full design rationale and the
bug writeups.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from cards.data.celeba import load_celebamask_hq_image_paths, split_celebamask_hq
from cards.data.celeba_attributes import (
    GROUNDABLE_CONCEPTS,
    TARGET_CLASSES,
    load_attribute_labels,
    load_attribute_names,
)
from cards.models.backbones import BACKBONES

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
CONCEPT_BANK_PATH = "trained_concepts_new/celeba_full/celeba_attractive_young/celeba_full_celeba_attractive_young_0.1_100.pkl"
OUT_DIR = Path("trained_models_new/celeba_full/celeba_attractive_young")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

TASK_SLICES: dict[str, slice] = {"Attractive": slice(0, 2), "Young": slice(2, 4)}


def embed_images(
    image_paths: list[Path], feature_extractor: torch.nn.Module, preprocess, device: str, label: str
) -> np.ndarray:
    embeddings = []
    for start in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[start : start + BATCH_SIZE]
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        with torch.no_grad():
            emb = torch.flatten(feature_extractor(batch), 1).cpu().numpy()
        embeddings.append(emb)
        if (start // BATCH_SIZE) % 20 == 0:
            print(f"[{label}] {start + len(batch_paths)}/{len(image_paths)}", flush=True)
    return np.concatenate(embeddings, axis=0)


def native_task_logits(
    image_paths: list[Path], native_model: torch.nn.Module, preprocess, device: str
) -> np.ndarray:
    logits = []
    for start in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[start : start + BATCH_SIZE]
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        with torch.no_grad():
            logits.append(native_model(batch).cpu().numpy())
    return np.concatenate(logits, axis=0)  # (N, 4)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()
    feature_extractor = spec.feature_extractor(native_model).to(DEVICE).eval()
    preprocess = spec.preprocess

    print("Loading CelebAMask-HQ metadata...", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]
    train_hq, val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices)
    train_paths = [image_paths_by_idx[i] for i in train_hq]
    val_paths = [image_paths_by_idx[i] for i in val_hq]
    print(f"{len(train_paths)} train images, {len(val_paths)} val images", flush=True)

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
    train_emb = embed_images(train_paths, feature_extractor, preprocess, DEVICE, "train")
    val_emb = embed_images(val_paths, feature_extractor, preprocess, DEVICE, "val")

    probe_layer = PosthocLinearCBM(concept_bank, backbone_name="celeba_attractive_young", n_classes=2).to(DEVICE)
    train_proj = probe_layer.compute_dist(torch.tensor(train_emb, device=DEVICE).float()).detach().cpu().numpy()
    val_proj = probe_layer.compute_dist(torch.tensor(val_emb, device=DEVICE).float()).detach().cpu().numpy()

    print("\n=== computing native model's task logits (for surrogate labels + true labels) ===", flush=True)
    train_native_logits = native_task_logits(train_paths, native_model, preprocess, DEVICE)
    val_native_logits = native_task_logits(val_paths, native_model, preprocess, DEVICE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task_name in TARGET_CLASSES:
        task_slice = TASK_SLICES[task_name]
        task_attr_idx = attr_names.index(task_name)
        print(f"\n########## target task: {task_name} ##########", flush=True)

        train_surrogate = train_native_logits[:, task_slice].argmax(axis=1)
        val_surrogate = val_native_logits[:, task_slice].argmax(axis=1)
        val_true = np.array([int(attr_labels_by_file[f"{i}.jpg"][task_attr_idx]) for i in val_hq])

        native_true_label_acc_val = (val_surrogate == val_true).mean()
        print(f"Native model's own true-label accuracy on val (informational): {native_true_label_acc_val:.4f}", flush=True)

        posthoc_layer = PosthocLinearCBM(
            concept_bank, backbone_name="celeba_attractive_young",
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

        model_path = OUT_DIR / f"pcbm_celeba_full__celeba_attractive_young__{task_name.lower()}__surrogate__seed_{SEED}__linear.ckpt"
        torch.save(posthoc_layer, model_path)
        print(f"Saved surrogate PCBM to {model_path}", flush=True)
        print(posthoc_layer.analyze_classifier(k=5), flush=True)


if __name__ == "__main__":
    main()
