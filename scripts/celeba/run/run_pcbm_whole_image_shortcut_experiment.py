"""Whole-image PCBM variant of the shortcut experiment -- same 4 shortcut-
injected classifiers (0%/33%/67%/100%) as run_pcbm_shortcut_experiment.py,
but CAVs are fit from whole train images filtered by each concept's own
attribute label (True/False), matching fit_celeba_full_whole_image_cavs.py's
convention, instead of that script's static region-crop bank. Prompted
directly ("Can we also run the PCBM whole image variant for the Shortcut
experiment?") as the shortcut-experiment counterpart to the whole-image
PCBM global-attribution variant just added.

Everything else -- per-rate CAV refitting (CAVs are tied to a SPECIFIC
model's own activation space, so must be refit per classifier, not
reused), full-scale train embedding, official-val fidelity diagnostic,
binary-task weight-row extraction -- is IDENTICAL to the region-crop
script; see that script's own docstring for the full design rationale.
Only the concept-example SOURCE changes (whole images by attribute label,
not region crops).
"""

from __future__ import annotations

import csv
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from concepts.concept_utils import ListDataset, learn_concept_bank
from run_cards_celeba_masking_hybrid_official_val_zscore import build_clean_official_val_paths
from torch.utils.data import DataLoader

from cards.data.celeba import load_celebamask_hq_image_paths, split_celebamask_hq
from cards.data.celeba_attributes import (
    GROUNDABLE_CONCEPTS,
    TARGET_CLASSES,
    load_attribute_labels,
    load_attribute_names,
)
from cards.models.backbones import BACKBONES

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
CELEBA_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebA\celeba")
RESULTS_DIR = Path("results")
CKPT_DIR = Path("trained_models_new/celeba")
OUT_DIR = Path("trained_models_new/celeba_shortcut_whole_image")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 64
N_CAV_SAMPLES = 50  # matches fit_celeba_full_whole_image_cavs.py -- needs 2*50=100 pos/neg per concept
MAX_PER_CLASS = 150  # matches fit_celeba_full_whole_image_cavs.py's own MAX_PER_CLASS
CAV_C_VALUE = 0.1   # matches train_pcbm_surrogate_celeba_full_whole_image.py's own CONCEPT_BANK_PATH C value
RATES_PCT = [0, 33, 67, 100]


class FlattenedFeatureExtractor(nn.Module):
    def __init__(self, feature_extractor: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, x):
        return torch.flatten(self.feature_extractor(x), 1)


def build_model(rate_pct: int) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(CKPT_DIR / f"resnet18_attractive_shortcut_{rate_pct}.pt", map_location="cpu")
    model.load_state_dict(state)
    return model.eval()


def embed_images(image_paths: list, feature_extractor: nn.Module, preprocess, device: str, label: str) -> np.ndarray:
    embeddings = []
    for start in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[start : start + BATCH_SIZE]
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        with torch.no_grad():
            emb = torch.flatten(feature_extractor(batch), 1).cpu().numpy()
        embeddings.append(emb)
        if (start // BATCH_SIZE) % 40 == 0:
            print(f"  [{label}] {start + len(batch_paths)}/{len(image_paths)}", flush=True)
    return np.concatenate(embeddings, axis=0)


def native_logits(image_paths: list, model: nn.Module, preprocess, device: str) -> np.ndarray:
    logits = []
    for start in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[start : start + BATCH_SIZE]
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        with torch.no_grad():
            logits.append(model(batch).cpu().numpy())
    return np.concatenate(logits, axis=0)  # (N, 2)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng_py = random.Random(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spec = BACKBONES["celeba_attractive_young"]  # preprocess only -- architecture-generic
    preprocess = spec.preprocess

    print("Loading CelebAMask-HQ metadata (surrogate TRAINING data)...", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]
    train_hq, _val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices)
    train_paths = [image_paths_by_idx[i] for i in train_hq]
    print(f"{len(train_paths)} train images (FULL scale, matching train_pcbm_surrogate_celeba_full.py).", flush=True)

    print("\nBuilding official-val validation sample (surrogate fidelity diagnostic only)...", flush=True)
    official_paths = build_clean_official_val_paths()
    official_attr_names = load_attribute_names(CELEBA_ROOT / "list_attr_celeba.txt")
    official_attr_labels = load_attribute_labels(CELEBA_ROOT / "list_attr_celeba.txt")
    official_attractive_idx = official_attr_names.index("Attractive")
    val_paths = official_paths
    val_true = np.array([int(official_attr_labels[p.name][official_attractive_idx]) for p in val_paths])
    print(f"{len(val_paths)} official-val images.", flush=True)

    print("\nBuilding whole-image concept example paths (STATIC, by attribute label, reused across all 4 rates)...", flush=True)
    concept_whole_image_paths = {}
    for concept_name in GROUNDABLE_CONCEPTS:
        concept_attr_idx = attr_names.index(concept_name)
        pos_hq = [i for i in train_hq if attr_labels_by_file[f"{i}.jpg"][concept_attr_idx]]
        neg_hq = [i for i in train_hq if not attr_labels_by_file[f"{i}.jpg"][concept_attr_idx]]
        rng_py.shuffle(pos_hq)
        rng_py.shuffle(neg_hq)
        pos_paths = [image_paths_by_idx[i] for i in pos_hq[:MAX_PER_CLASS]]
        neg_paths = [image_paths_by_idx[i] for i in neg_hq[:MAX_PER_CLASS]]
        if min(len(pos_paths), len(neg_paths)) < 2 * N_CAV_SAMPLES:
            raise ValueError(f"{concept_name} has too few whole images for n_samples={N_CAV_SAMPLES}")
        concept_whole_image_paths[concept_name] = (pos_paths, neg_paths)
    print(f"{len(GROUNDABLE_CONCEPTS)} concepts' whole-image example sets built.", flush=True)

    from concepts import ConceptBank
    from models import PosthocLinearCBM
    from train_pcbm import run_linear_probe

    all_rows = []  # (rate_pct, concept_name, weight)
    scores_by_rate: dict[int, dict[str, float]] = {}

    for rate_pct in RATES_PCT:
        print(f"\n=== rate={rate_pct}% ===", flush=True)
        model = build_model(rate_pct).to(DEVICE)
        feature_extractor = FlattenedFeatureExtractor(nn.Sequential(*list(model.children())[:-1])).to(DEVICE).eval()

        print(f"  fitting CAVs against rate={rate_pct}%'s own backbone...", flush=True)
        concept_dict = {}
        for concept_name in GROUNDABLE_CONCEPTS:
            pos_paths, neg_paths = concept_whole_image_paths[concept_name]
            pos_loader = DataLoader(ListDataset(pos_paths, preprocess), batch_size=25, shuffle=False)
            neg_loader = DataLoader(ListDataset(neg_paths, preprocess), batch_size=25, shuffle=False)
            cav_info = learn_concept_bank(pos_loader, neg_loader, feature_extractor, N_CAV_SAMPLES, [CAV_C_VALUE], device=DEVICE)
            concept_dict[concept_name] = cav_info[CAV_C_VALUE]
        cav_out_path = OUT_DIR / f"celeba_shortcut_whole_image_{rate_pct}_{CAV_C_VALUE}_{2 * N_CAV_SAMPLES}.pkl"
        with open(cav_out_path, "wb") as f:
            pickle.dump(concept_dict, f)
        print(f"  Saved CAVs to {cav_out_path}", flush=True)

        concept_bank = ConceptBank(concept_dict, DEVICE)
        probe_layer = PosthocLinearCBM(concept_bank, backbone_name=f"celeba_shortcut_whole_image_{rate_pct}", n_classes=2).to(DEVICE)

        print(f"  embedding {len(train_paths)} train + {len(val_paths)} val images...", flush=True)
        train_emb = embed_images(train_paths, feature_extractor, preprocess, DEVICE, "train")
        val_emb = embed_images(val_paths, feature_extractor, preprocess, DEVICE, "val")
        train_proj = probe_layer.compute_dist(torch.tensor(train_emb, device=DEVICE).float()).detach().cpu().numpy()
        val_proj = probe_layer.compute_dist(torch.tensor(val_emb, device=DEVICE).float()).detach().cpu().numpy()

        train_native_logits = native_logits(train_paths, model, preprocess, DEVICE)
        val_native_logits = native_logits(val_paths, model, preprocess, DEVICE)
        train_surrogate = train_native_logits.argmax(axis=1)
        val_surrogate = val_native_logits.argmax(axis=1)

        posthoc_layer = PosthocLinearCBM(
            concept_bank, backbone_name=f"celeba_shortcut_whole_image_{rate_pct}",
            idx_to_class={0: "not_Attractive", 1: "Attractive"}, n_classes=2,
        ).to(DEVICE)

        class Args:
            seed = SEED
            lam = 0.0002
            alpha = 0.99

        run_info, weights, bias = run_linear_probe(Args(), (train_proj, train_surrogate), (val_proj, val_surrogate))
        print(f"  train fidelity (agreement with native model): {run_info['train_acc']:.2f}%", flush=True)
        print(f"  val fidelity (agreement with native model, official-val): {run_info['test_acc']:.2f}%", flush=True)

        posthoc_layer.set_weights(weights=weights.astype(np.float32), bias=bias.astype(np.float32))
        with torch.no_grad():
            val_logits = posthoc_layer.forward_projs(torch.tensor(val_proj, device=DEVICE).float())
            pcbm_pred = (val_logits.squeeze(-1) > 0).long().cpu().numpy()
        pcbm_true_label_acc = (pcbm_pred == val_true).mean()
        print(f"  PCBM surrogate's own true-Attractive-label accuracy on official-val: {pcbm_true_label_acc:.4f}", flush=True)

        # binary task: sklearn's coef_ is a single (1, n_concepts) row, not
        # one row per class -- weight[0, :] is the task-positive-class row
        # (see notes/celeba_correlation_investigation.md's own PCBM gotcha).
        weight_row = weights[0, :] if weights.ndim == 2 else weights
        scores = dict(zip(concept_bank.concept_names, weight_row.tolist()))
        scores_by_rate[rate_pct] = scores
        for concept_name, w in scores.items():
            all_rows.append((rate_pct, concept_name, w))

        model_path = OUT_DIR / f"pcbm_celeba_shortcut_whole_image_{rate_pct}__surrogate__seed_{SEED}__linear.ckpt"
        torch.save(posthoc_layer, model_path)
        print(f"  Saved surrogate PCBM to {model_path}", flush=True)

        ranked = sorted(scores.items(), key=lambda kv: -abs(kv[1]))
        print(f"  top-5 by |weight|: {[(c, round(s, 4)) for c, s in ranked[:5]]}", flush=True)
        print(f"  mean |weight| across all 26 concepts: {np.mean([abs(s) for s in scores.values()]):.4f}", flush=True)

    with open(RESULTS_DIR / "pcbm_whole_image_celeba_shortcut_experiment.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rate_pct", "concept_name", "weight"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to results/pcbm_whole_image_celeba_shortcut_experiment.csv")

    print("\n=== mean |weight| across all 26 concepts, per rate (the headline decline check) ===")
    for rate_pct in RATES_PCT:
        mean_abs = np.mean([abs(s) for s in scores_by_rate[rate_pct].values()])
        print(f"  rate={rate_pct:>3d}%: mean |weight| = {mean_abs:.4f}")


if __name__ == "__main__":
    main()
