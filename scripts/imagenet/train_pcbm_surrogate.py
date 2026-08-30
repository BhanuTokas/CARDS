"""Phase 1 (notes/pcbm_correlation_investigation.md, CARDS vs. TCAV vs.
PCBM plan): fits PCBM's own linear classifier as a *surrogate* of the
native, off-the-shelf ImageNet-pretrained resnet18's own predictions --
not against ImageNet ground-truth labels. Per the plan's surrogate-
modeling correction: PCBM is the one method among the three that must
construct its own model, and it should be judged on fidelity to the
model actually being explained (the native resnet18), not accuracy
against truth -- so `train_pcbm.py`'s usual label input (ground truth)
is replaced here with `native_model(x).argmax()`, restricted to the 25
target-class output indices, for every image in the train-fitting slice.

Reuses post_hoc_cbm's own `ConceptBank`, `PosthocLinearCBM`, and
`run_linear_probe` (the exact same SGDClassifier/elasticnet fitting
logic `train_pcbm.py` uses) directly -- only the *label source* and data
loading are custom, since surrogate-labeling from a second model isn't
something post_hoc_cbm's own generic dataset dispatch was built for.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from build_imagenet_slice import TARGET_CLASSES

from cards.models.backbones import BACKBONES

CONCEPT_BANK_PATH = "trained_concepts_new/broden_baseline/resnet18_torchvision/broden_resnet18_torchvision_0.01_50.pkl"
IMAGENET_SLICE_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\imagenet_slice")
OUT_DIR = Path("trained_models_new/imagenet_slice_baseline/resnet18_torchvision")
SEED = 42
DEVICE = "cuda"

CLASS_NAMES = [name for _, _, name in TARGET_CLASSES]
NATIVE_LABEL_IDX = [idx for idx, _, _ in TARGET_CLASSES]  # into the native model's 1000-way output
LOCAL_IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}


def load_split(split: str, native_model: torch.nn.Module, feature_extractor: torch.nn.Module, preprocess, device: str):
    """Returns (embeddings (N, embed_dim), surrogate_labels (N,),
    true_labels (N,)) for every image under IMAGENET_SLICE_ROOT/split/<class>/."""
    embeddings = []
    surrogate_labels = []
    true_labels = []
    native_label_tensor = torch.tensor(NATIVE_LABEL_IDX, device=device)

    for local_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = IMAGENET_SLICE_ROOT / split / class_name
        paths = sorted(class_dir.glob("*.jpg"))
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)
        with torch.no_grad():
            native_logits = native_model(batch)  # (n, 1000)
            restricted = native_logits[:, native_label_tensor]  # (n, 25), column order == CLASS_NAMES order
            surrogate = restricted.argmax(dim=1).cpu().numpy()  # local 0-24 index
            emb = feature_extractor(batch).flatten(1).cpu().numpy()
        embeddings.append(emb)
        surrogate_labels.append(surrogate)
        true_labels.append(np.full(len(paths), local_idx))
        print(f"[{split}] {class_name}: {len(paths)} images, "
              f"native model agrees with true label on {(surrogate == local_idx).mean():.3f}", flush=True)

    return (
        np.concatenate(embeddings, axis=0),
        np.concatenate(surrogate_labels, axis=0),
        np.concatenate(true_labels, axis=0),
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    spec = BACKBONES["resnet18"]
    native_model = spec.load_native().to(DEVICE).eval()
    feature_extractor = spec.feature_extractor(native_model).to(DEVICE).eval()
    preprocess = spec.preprocess

    print("Loading concept bank...", flush=True)
    with open(CONCEPT_BANK_PATH, "rb") as f:
        all_concepts = pickle.load(f)
    print(f"{len(all_concepts)} concepts in bank.", flush=True)

    from concepts import ConceptBank
    from models import PosthocLinearCBM
    from train_pcbm import run_linear_probe

    concept_bank = ConceptBank(all_concepts, DEVICE)

    class Args:
        seed = SEED
        lam = 0.0002
        alpha = 0.99

    posthoc_layer = PosthocLinearCBM(
        concept_bank, backbone_name="resnet18_torchvision", idx_to_class=LOCAL_IDX_TO_CLASS, n_classes=len(CLASS_NAMES)
    ).to(DEVICE)

    print("\n=== computing train embeddings/projections (surrogate labels) ===", flush=True)
    train_emb, train_surrogate, _train_true = load_split("train", native_model, feature_extractor, preprocess, DEVICE)
    train_proj = posthoc_layer.compute_dist(torch.tensor(train_emb, device=DEVICE).float()).detach().cpu().numpy()

    print("\n=== computing val embeddings/projections (held out) ===", flush=True)
    val_emb, val_surrogate, val_true = load_split("val", native_model, feature_extractor, preprocess, DEVICE)
    val_proj = posthoc_layer.compute_dist(torch.tensor(val_emb, device=DEVICE).float()).detach().cpu().numpy()

    native_true_label_acc_val = (val_surrogate == val_true).mean()
    print(f"\nNative model's own true-label accuracy on val (informational, not about PCBM): {native_true_label_acc_val:.4f}", flush=True)

    print("\n=== fitting PCBM's linear head against SURROGATE labels ===", flush=True)
    args = Args()
    run_info, weights, bias = run_linear_probe(args, (train_proj, train_surrogate), (val_proj, val_surrogate))
    print(f"train fidelity (agreement with native model, train slice): {run_info['train_acc']:.2f}%")
    print(f"val fidelity (agreement with native model, held-out val slice): {run_info['test_acc']:.2f}%")

    posthoc_layer.set_weights(weights=weights, bias=bias)

    # Fidelity already reported by run_linear_probe (surrogate vs surrogate).
    # Also report: does PCBM's surrogate agree with the TRUE label as often
    # as the native model itself does (a secondary, informational check).
    with torch.no_grad():
        val_logits = posthoc_layer.forward_projs(torch.tensor(val_proj, device=DEVICE).float())
        pcbm_pred = val_logits.argmax(dim=1).cpu().numpy()
    pcbm_true_label_acc = (pcbm_pred == val_true).mean()
    pcbm_native_fidelity = (pcbm_pred == val_surrogate).mean()
    print(f"\nPCBM surrogate's own true-label accuracy on val: {pcbm_true_label_acc:.4f}")
    print(f"PCBM surrogate's fidelity to native model's predictions on val: {pcbm_native_fidelity:.4f}")
    print(f"(native model's own true-label accuracy on val, for reference: {native_true_label_acc_val:.4f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / f"pcbm_imagenet_slice__resnet18_torchvision__broden_baseline__surrogate__seed_{SEED}__linear.ckpt"
    torch.save(posthoc_layer, model_path)
    print(f"\nSaved surrogate PCBM to {model_path}")

    print("\nTop-5 concept weights per class:")
    print(posthoc_layer.analyze_classifier(k=5))


if __name__ == "__main__":
    main()
