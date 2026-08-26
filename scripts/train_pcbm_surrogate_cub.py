"""CUB-track analogue of train_pcbm_surrogate.py: fits PCBM's own linear
classifier as a *surrogate* of the native resnet18_cub's own 200-way
predictions (not ground-truth CUB labels), using the new real-part-crop
concept bank (scripts/fit_cub_part_cavs.py's output) instead of the
official 112-attribute bank -- per the same surrogate-modeling
correction established for the ImageNet track (notes/
pcbm_correlation_investigation.md, Phase 1): PCBM should be judged on
fidelity to the model it's explaining, not accuracy against truth.

Unlike the ImageNet track, no output-index restriction is needed --
resnet18_cub's native 200-way head already matches CUB's own 200 classes
one-to-one (no 1000-vs-25 subset gap), so `argmax` over the raw logits
IS the local class index directly.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from cards.data.cub_parts import load_images_txt  # noqa: E402
from cards.models.backbones import BACKBONES  # noqa: E402

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
CONCEPT_BANK_PATH = "trained_concepts_new/cub_parts/resnet18_cub/cub_parts_resnet18_cub_0.1_100.pkl"
OUT_DIR = Path("trained_models_new/cub_parts/resnet18_cub")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64


def load_classes(cub_root: Path) -> dict[int, str]:
    """1-indexed class_id -> name (e.g. "Black_footed_Albatross", "001."
    prefix dropped for readability)."""
    result = {}
    for line in (cub_root / "classes.txt").read_text().splitlines():
        class_id, name = line.split(maxsplit=1)
        result[int(class_id)] = name.split(".", 1)[1] if "." in name else name
    return result


def load_split_ids(cub_root: Path) -> tuple[list[str], list[str]]:
    """(train_image_ids, test_image_ids), per train_test_split.txt's own
    1/0 flag."""
    train_ids, test_ids = [], []
    for line in (cub_root / "train_test_split.txt").read_text().splitlines():
        image_id, is_train = line.split()
        (train_ids if is_train == "1" else test_ids).append(image_id)
    return train_ids, test_ids


def load_image_class_labels(cub_root: Path) -> dict[str, int]:
    """image_id -> 1-indexed class_id."""
    result = {}
    for line in (cub_root / "image_class_labels.txt").read_text().splitlines():
        image_id, class_id = line.split()
        result[image_id] = int(class_id)
    return result


def embed_and_predict(
    image_ids: list[str],
    image_paths: dict[str, Path],
    class_labels: dict[str, int],
    native_model: torch.nn.Module,
    feature_extractor: torch.nn.Module,
    preprocess,
    device: str,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (embeddings (N, embed_dim), surrogate_labels (N,) 0-indexed,
    true_labels (N,) 0-indexed) for image_ids."""
    embeddings, surrogate_labels, true_labels = [], [], []
    for start in range(0, len(image_ids), BATCH_SIZE):
        batch_ids = image_ids[start : start + BATCH_SIZE]
        batch = torch.stack([preprocess(Image.open(image_paths[i]).convert("RGB")) for i in batch_ids]).to(device)
        with torch.no_grad():
            native_logits = native_model(batch)  # (n, 200)
            surrogate = native_logits.argmax(dim=1).cpu().numpy()
            emb = feature_extractor(batch)
            emb = torch.flatten(emb, 1).cpu().numpy()
        embeddings.append(emb)
        surrogate_labels.append(surrogate)
        true_labels.append(np.array([class_labels[i] - 1 for i in batch_ids]))
        if (start // BATCH_SIZE) % 20 == 0:
            print(f"[{label}] {start + len(batch_ids)}/{len(image_ids)}", flush=True)

    return (
        np.concatenate(embeddings, axis=0),
        np.concatenate(surrogate_labels, axis=0),
        np.concatenate(true_labels, axis=0),
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    spec = BACKBONES["resnet18_cub"]
    native_model = spec.load_native().to(DEVICE).eval()
    feature_extractor = spec.feature_extractor(native_model).to(DEVICE).eval()
    preprocess = spec.preprocess

    image_paths = load_images_txt(CUB_ROOT)
    class_labels = load_image_class_labels(CUB_ROOT)
    classes = load_classes(CUB_ROOT)
    idx_to_class = {i: classes[class_id] for i, class_id in enumerate(sorted(classes))}
    train_ids, test_ids = load_split_ids(CUB_ROOT)
    print(f"{len(train_ids)} train images, {len(test_ids)} test images, {len(classes)} classes", flush=True)

    print("Loading part-crop concept bank...", flush=True)
    with open(CONCEPT_BANK_PATH, "rb") as f:
        all_concepts = pickle.load(f)
    print(f"{len(all_concepts)} concepts in bank: {sorted(all_concepts)}", flush=True)

    from concepts import ConceptBank
    from models import PosthocLinearCBM
    from train_pcbm import run_linear_probe

    concept_bank = ConceptBank(all_concepts, DEVICE)

    class Args:
        seed = SEED
        lam = 0.0002
        alpha = 0.99

    posthoc_layer = PosthocLinearCBM(
        concept_bank, backbone_name="resnet18_cub", idx_to_class=idx_to_class, n_classes=len(classes)
    ).to(DEVICE)

    print("\n=== computing train embeddings/projections (surrogate labels) ===", flush=True)
    train_emb, train_surrogate, train_true = embed_and_predict(
        train_ids, image_paths, class_labels, native_model, feature_extractor, preprocess, DEVICE, "train"
    )
    train_proj = posthoc_layer.compute_dist(torch.tensor(train_emb, device=DEVICE).float()).detach().cpu().numpy()

    print("\n=== computing test embeddings/projections (held out) ===", flush=True)
    test_emb, test_surrogate, test_true = embed_and_predict(
        test_ids, image_paths, class_labels, native_model, feature_extractor, preprocess, DEVICE, "test"
    )
    test_proj = posthoc_layer.compute_dist(torch.tensor(test_emb, device=DEVICE).float()).detach().cpu().numpy()

    native_true_label_acc_test = (test_surrogate == test_true).mean()
    print(f"\nNative resnet18_cub's own true-label accuracy on test (informational): {native_true_label_acc_test:.4f}", flush=True)

    print("\n=== fitting PCBM's linear head against SURROGATE labels ===", flush=True)
    args = Args()
    run_info, weights, bias = run_linear_probe(args, (train_proj, train_surrogate), (test_proj, test_surrogate))
    print(f"train fidelity (agreement with native model, train slice): {run_info['train_acc']:.2f}%")
    print(f"test fidelity (agreement with native model, held-out test slice): {run_info['test_acc']:.2f}%")

    posthoc_layer.set_weights(weights=weights, bias=bias)

    with torch.no_grad():
        test_logits = posthoc_layer.forward_projs(torch.tensor(test_proj, device=DEVICE).float())
        pcbm_pred = test_logits.argmax(dim=1).cpu().numpy()
    pcbm_true_label_acc = (pcbm_pred == test_true).mean()
    pcbm_native_fidelity = (pcbm_pred == test_surrogate).mean()
    print(f"\nPCBM surrogate's own true-label accuracy on test: {pcbm_true_label_acc:.4f}")
    print(f"PCBM surrogate's fidelity to native model's predictions on test: {pcbm_native_fidelity:.4f}")
    print(f"(native model's own true-label accuracy on test, for reference: {native_true_label_acc_test:.4f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / f"pcbm_cub__resnet18_cub__cub_parts__surrogate__seed_{SEED}__linear.ckpt"
    torch.save(posthoc_layer, model_path)
    print(f"\nSaved surrogate PCBM to {model_path}")

    print("\nTop-5 concept weights for a few classes:")
    print(posthoc_layer.analyze_classifier(k=5))


if __name__ == "__main__":
    main()
