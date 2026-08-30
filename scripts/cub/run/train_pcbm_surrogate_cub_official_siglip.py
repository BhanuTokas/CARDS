"""Fits PCBM's linear surrogate classifier on top of the SigLIP-backbone
official-112-attribute CAV bank (scripts/fit_cub_official_cavs_siglip.py's
output) -- the SigLIP-backbone analogue of train_pcbm_surrogate_cub.py
(which does this for the part-crop bank on resnet18_cub) and of the
existing official-bank resnet18_cub PCBM checkpoint used throughout this
investigation (score_all_methods_against_cub_faithfulness.py's
`load_pcbm_official_scores`).

Same surrogate-modeling framing as every other PCBM variant built
specifically for this investigation (labels = resnet18_cub's own argmax,
not ground truth -- PCBM judged on fidelity to the model it explains,
not accuracy against truth). BUT: checked directly (not assumed) that the
existing official-112-bank resnet18_cub checkpoint used as the "PCBM
(official 112-bank weight)" comparator throughout v39-v55
(`../post_hoc_cbm/trained_models_new/cub/resnet18_cub/
pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam_0.0002...ckpt`) predates
that surrogate-modeling correction -- it's post_hoc_cbm's own
`train_pcbm.py` CLI's stock output, which fits against CUB's real
ground-truth class labels (confirmed via `get_dataset`'s own
`CUBDataset.__getitem__`, no argmax anywhere in that CLI path), left over
from the original literature-reproduction step (v34). So a strict,
backbone-only ablation against that specific existing baseline needs
GROUND-TRUTH labels too, not surrogate ones -- this script fits BOTH
(reusing the same embeddings) and saves two separate score files, so
either comparison (backbone-only vs. framing-consistent-with-the-rest-
of-this-investigation) is available, not just one silently assumed
correct.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from cards.data.cub_parts import load_images_txt
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
CONCEPT_BANK_PATH = "trained_concepts_new/cub_official/siglip/cub_official_siglip_0.1_100.pkl"
RESULTS_DIR = Path("results")
OUT_DIR = Path("trained_models_new/cub_official_siglip")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128


def load_classes(cub_root: Path) -> dict[int, str]:
    result = {}
    for line in (cub_root / "classes.txt").read_text().splitlines():
        class_id, name = line.split(maxsplit=1)
        result[int(class_id)] = name.split(".", 1)[1] if "." in name else name
    return result


def load_split_ids(cub_root: Path) -> tuple[list[str], list[str]]:
    train_ids, test_ids = [], []
    for line in (cub_root / "train_test_split.txt").read_text().splitlines():
        image_id, is_train = line.split()
        (train_ids if is_train == "1" else test_ids).append(image_id)
    return train_ids, test_ids


def load_image_class_labels(cub_root: Path) -> dict[str, int]:
    result = {}
    for line in (cub_root / "image_class_labels.txt").read_text().splitlines():
        image_id, class_id = line.split()
        result[image_id] = int(class_id)
    return result


def embed_and_predict(
    image_ids: list[str], image_paths: dict[str, Path], class_labels: dict[str, int],
    native_model: torch.nn.Module, siglip_model, preprocess, native_preprocess, device: str, label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (embeddings (N, embed_dim), surrogate_labels (N,) 0-indexed,
    true_labels (N,) 0-indexed) for image_ids. Two separate preprocess
    passes per image (SigLIP's own for the embedding, resnet18_cub's own
    for the surrogate label) since the two backbones expect different
    input resolutions/normalization -- correctness over speed here."""
    embeddings, surrogate_labels, true_labels = [], [], []
    for start in range(0, len(image_ids), BATCH_SIZE):
        batch_ids = image_ids[start : start + BATCH_SIZE]
        images = [Image.open(image_paths[i]).convert("RGB") for i in batch_ids]
        native_batch = torch.stack([native_preprocess(img) for img in images]).to(device)
        siglip_batch = torch.stack([preprocess(img) for img in images]).to(device)
        with torch.no_grad():
            native_logits = native_model(native_batch)
            surrogate = native_logits.argmax(dim=1).cpu().numpy()
            emb = F.normalize(siglip_model.encode_image(siglip_batch), dim=-1).cpu().numpy()
        embeddings.append(emb)
        surrogate_labels.append(surrogate)
        true_labels.append(np.array([class_labels[i] - 1 for i in batch_ids]))
        if (start // BATCH_SIZE) % 10 == 0:
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
    native_preprocess = spec.preprocess

    print("Loading SigLIP (ViT-B-16-SigLIP/webli)...", flush=True)
    siglip_cfg = OmegaConf.create({"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE})
    siglip_encoder = instantiate_encoder(OmegaConf.create({"encoder": siglip_cfg, "device": DEVICE}))

    image_paths = load_images_txt(CUB_ROOT)
    class_labels = load_image_class_labels(CUB_ROOT)
    classes = load_classes(CUB_ROOT)
    idx_to_class = {i: classes[class_id] for i, class_id in enumerate(sorted(classes))}
    train_ids, test_ids = load_split_ids(CUB_ROOT)
    print(f"{len(train_ids)} train images, {len(test_ids)} test images, {len(classes)} classes", flush=True)

    print("Loading SigLIP-backbone official CAV bank...", flush=True)
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
        concept_bank, backbone_name="siglip", idx_to_class=idx_to_class, n_classes=len(classes)
    ).to(DEVICE)

    print("\n=== computing train embeddings/projections (surrogate labels) ===", flush=True)
    train_emb, train_surrogate, train_true = embed_and_predict(
        train_ids, image_paths, class_labels, native_model, siglip_encoder.model, siglip_encoder.preprocess,
        native_preprocess, DEVICE, "train",
    )
    train_proj = posthoc_layer.compute_dist(torch.tensor(train_emb, device=DEVICE).float()).detach().cpu().numpy()

    print("\n=== computing test embeddings/projections (held out) ===", flush=True)
    test_emb, test_surrogate, test_true = embed_and_predict(
        test_ids, image_paths, class_labels, native_model, siglip_encoder.model, siglip_encoder.preprocess,
        native_preprocess, DEVICE, "test",
    )
    test_proj = posthoc_layer.compute_dist(torch.tensor(test_emb, device=DEVICE).float()).detach().cpu().numpy()

    native_true_label_acc_test = (test_surrogate == test_true).mean()
    print(f"\nNative resnet18_cub's own true-label accuracy on test (informational): {native_true_label_acc_test:.4f}", flush=True)

    import csv

    RESULTS_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def fit_and_save(train_labels, test_labels, tag, ckpt_suffix, out_csv):
        layer = PosthocLinearCBM(concept_bank, backbone_name="siglip", idx_to_class=idx_to_class, n_classes=len(classes)).to(DEVICE)
        print(f"\n=== fitting PCBM's linear head against {tag} labels ===", flush=True)
        run_info, weights, bias = run_linear_probe(Args(), (train_proj, train_labels), (test_proj, test_labels))
        print(f"[{tag}] train acc={run_info['train_acc']:.2f}%  test acc={run_info['test_acc']:.2f}%", flush=True)
        layer.set_weights(weights=weights, bias=bias)

        with torch.no_grad():
            test_logits = layer.forward_projs(torch.tensor(test_proj, device=DEVICE).float())
            pcbm_pred = test_logits.argmax(dim=1).cpu().numpy()
        print(f"[{tag}] PCBM(official, SigLIP backbone) true-label accuracy on test: {(pcbm_pred == test_true).mean():.4f}", flush=True)
        print(f"[{tag}] PCBM(official, SigLIP backbone) fidelity to native model's predictions on test: {(pcbm_pred == test_surrogate).mean():.4f}", flush=True)

        model_path = OUT_DIR / f"pcbm_cub__siglip__cub_official__{ckpt_suffix}__seed_{SEED}__linear.ckpt"
        torch.save(layer, model_path)
        print(f"[{tag}] Saved to {model_path}", flush=True)

        weight = layer.classifier.weight.detach().cpu().numpy()  # (200, 112)
        with open(RESULTS_DIR / out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["attribute_index", "native_class_idx", "weight"])
            for attr_idx in range(weight.shape[1]):
                for local_idx in range(weight.shape[0]):
                    writer.writerow([attr_idx, local_idx, float(weight[local_idx, attr_idx])])
        print(f"[{tag}] Saved {weight.shape[0] * weight.shape[1]} (attribute, class) scores to results/{out_csv}", flush=True)

    # (1) Surrogate framing -- this investigation's own stated best practice
    # (fidelity to the model being explained), matching the part-crop bank
    # and CLIP/SigLIP-concepts variants.
    fit_and_save(train_surrogate, test_surrogate, "SURROGATE", "surrogate",
                 "pcbm_official_siglip_cub_scores.csv")

    # (2) Ground-truth framing -- matches the EXISTING resnet18_cub
    # official-bank baseline's own actual fitting convention (see module
    # docstring), for a strict backbone-only ablation against it.
    fit_and_save(train_true, test_true, "GROUND-TRUTH", "groundtruth",
                 "pcbm_official_siglip_groundtruth_cub_scores.csv")


if __name__ == "__main__":
    main()
