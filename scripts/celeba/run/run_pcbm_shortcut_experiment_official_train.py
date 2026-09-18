"""Official-train counterpart of `run_pcbm_shortcut_experiment.py` --
"conventional" PCBM (CAVs fit as a linear probe directly in each
classifier's OWN ResNet18 activation space, the original post-hoc CBM
paper's default backbone convention -- confirmed directly, "conventional
means the resnet-18 backbone which was used as the default backbone in
posthoc cbm paper") against the 22 official-train shortcut checkpoints
(2 tasks x 11 rates) instead of the original 4 HQ-trained ones. See
`run_pcbm_clip_concepts_shortcut_experiment_official_train.py` for the
other 2 variants (CLIP-RN50, SigLIP).

WHOLE-IMAGE concept exemplars, NOT region crops -- confirmed directly
("Also, I need complete image version not region crops"), matching
`run_pcbm_whole_image_shortcut_experiment.py`'s own design (CAVs fit
from whole train images filtered by each concept's own attribute label,
True/False) rather than the crop-based `run_pcbm_shortcut_experiment.py`'s
`CONCEPT_ROOT` bank. This also sidesteps needing CelebA-HQ at all
(confirmed directly, "I do not have the CELEBA_HQ downloaded there") --
region crops only exist because CelebA-HQ ships real per-pixel
segmentation masks to crop from; whole images filtered by attribute
label need nothing but `list_attr_celeba.txt`.

Concept exemplars AND the surrogate's own TRAINING images are both now
sourced from official CelebA TRAIN (partition==0) instead of CelebA-HQ's
train split. Surrogate training uses a SEED-shuffled sample of
N_SURROGATE_TRAIN=25,500 official-train images (matching the original
HQ-scale convention's own image count, kept for rough compute parity --
official train has 162,770 images total, ~6.4x more than HQ's, so this
is a deliberate subsample, not "full scale" the way the original
paper-bound HQ run was). CAVs are refit per (task, rate) -- tied to that
specific model's own activation space, same necessity as the original
script's own per-rate refit, just widened by the task axis.

VALIDATION (fidelity diagnostic only, not used to compute attribution
weights) is the plain official CelebA val partition (partition==1) --
no CelebA-HQ-leakage exclusion needed: these classifiers train on
partition==0 only, already disjoint from partition==1 by construction.
"""

from __future__ import annotations

import csv
import os
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
from torch.utils.data import DataLoader

from cards.data.celeba_attributes import (
    GROUNDABLE_CONCEPTS,
    load_attribute_labels,
    load_attribute_names,
)
from cards.models.backbones import BACKBONES

CELEBA_ROOT = Path(os.environ.get("CELEBA_ROOT", r"C:\Users\btokas\Projects\Datasets\CelebA\celeba"))
RESULTS_DIR = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
CKPT_DIR = Path(os.environ.get("CARDS_CKPT_DIR", "trained_models_new/celeba"))
OUT_DIR = Path(os.environ.get("CARDS_PCBM_OUT_DIR", "trained_models_new/celeba_shortcut_official_train"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 64
N_CAV_SAMPLES = 50  # needs 2*50=100 pos/neg whole-image exemplars per concept
MAX_PER_CLASS = 150  # matches fit_celeba_full_whole_image_cavs.py's own MAX_PER_CLASS
CAV_C_VALUE = 0.1
N_SURROGATE_TRAIN = 25_500  # subsample of official train, matching the original HQ-scale image count
RATES_PCT = list(range(0, 101, 10))
TASKS = ["Attractive", "Male"]


class FlattenedFeatureExtractor(nn.Module):
    def __init__(self, feature_extractor: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, x):
        return torch.flatten(self.feature_extractor(x), 1)


def build_model(task: str, rate_pct: int) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    ckpt_path = CKPT_DIR / f"resnet18_official_train_{task.lower()}_shortcut_{rate_pct}.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    return model.eval()


def load_official_partition_paths() -> tuple[list[Path], list[Path]]:
    """(train_paths, val_paths) from list_eval_partition.txt (0=train, 1=val)."""
    train_paths, val_paths = [], []
    with open(CELEBA_ROOT / "list_eval_partition.txt") as f:
        for line in f:
            fname, part = line.split()
            if part == "0":
                train_paths.append(CELEBA_ROOT / "img_align_celeba" / fname)
            elif part == "1":
                val_paths.append(CELEBA_ROOT / "img_align_celeba" / fname)
    return train_paths, val_paths


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
    return np.concatenate(logits, axis=0)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng_py = random.Random(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spec = BACKBONES["celeba_attractive_young"]  # preprocess only -- architecture-generic
    preprocess = spec.preprocess

    print("Loading official CelebA metadata...", flush=True)
    all_train_paths, val_paths = load_official_partition_paths()
    attr_names = load_attribute_names(CELEBA_ROOT / "list_attr_celeba.txt")
    attr_labels = load_attribute_labels(CELEBA_ROOT / "list_attr_celeba.txt")

    surrogate_train_paths = list(all_train_paths)
    rng_py.shuffle(surrogate_train_paths)
    surrogate_train_paths = surrogate_train_paths[:N_SURROGATE_TRAIN]
    print(f"{len(surrogate_train_paths)} surrogate-training images (sampled from {len(all_train_paths)} official train).", flush=True)

    print(f"{len(val_paths)} official-val images (fidelity diagnostic only).", flush=True)
    val_true_by_task = {
        task: np.array([int(attr_labels[p.name][attr_names.index(task)]) for p in val_paths])
        for task in TASKS
    }

    print("\nBuilding whole-image concept example paths (STATIC, by attribute label, reused across both tasks, all rates)...", flush=True)
    concept_whole_image_paths = {}
    for concept_name in GROUNDABLE_CONCEPTS:
        concept_attr_idx = attr_names.index(concept_name)
        pos_all = [p for p in all_train_paths if attr_labels[p.name][concept_attr_idx]]
        neg_all = [p for p in all_train_paths if not attr_labels[p.name][concept_attr_idx]]
        rng_py.shuffle(pos_all)
        rng_py.shuffle(neg_all)
        pos_paths = pos_all[:MAX_PER_CLASS]
        neg_paths = neg_all[:MAX_PER_CLASS]
        if min(len(pos_paths), len(neg_paths)) < 2 * N_CAV_SAMPLES:
            raise ValueError(f"{concept_name} has too few whole images for n_samples={N_CAV_SAMPLES}")
        concept_whole_image_paths[concept_name] = (pos_paths, neg_paths)
    print(f"{len(GROUNDABLE_CONCEPTS)} concepts' whole-image example sets built.", flush=True)

    from concepts import ConceptBank
    from models import PosthocLinearCBM
    from train_pcbm import run_linear_probe

    all_rows = []  # (task, rate_pct, concept_name, weight)
    scores_by_task_rate: dict[str, dict[int, dict[str, float]]] = {t: {} for t in TASKS}

    for task in TASKS:
        val_true = val_true_by_task[task]
        for rate_pct in RATES_PCT:
            print(f"\n=== task={task} rate={rate_pct}% ===", flush=True)
            model = build_model(task, rate_pct).to(DEVICE)
            feature_extractor = FlattenedFeatureExtractor(nn.Sequential(*list(model.children())[:-1])).to(DEVICE).eval()

            print(f"  fitting CAVs against task={task} rate={rate_pct}%'s own backbone...", flush=True)
            concept_dict = {}
            for concept_name in GROUNDABLE_CONCEPTS:
                pos_paths, neg_paths = concept_whole_image_paths[concept_name]
                pos_loader = DataLoader(ListDataset(pos_paths, preprocess), batch_size=25, shuffle=False)
                neg_loader = DataLoader(ListDataset(neg_paths, preprocess), batch_size=25, shuffle=False)
                cav_info = learn_concept_bank(pos_loader, neg_loader, feature_extractor, N_CAV_SAMPLES, [CAV_C_VALUE], device=DEVICE)
                concept_dict[concept_name] = cav_info[CAV_C_VALUE]
            cav_out_path = OUT_DIR / f"celeba_shortcut_official_train_{task.lower()}_{rate_pct}_{CAV_C_VALUE}_{2 * N_CAV_SAMPLES}.pkl"
            with open(cav_out_path, "wb") as f:
                pickle.dump(concept_dict, f)

            concept_bank = ConceptBank(concept_dict, DEVICE)

            print(f"  embedding {len(surrogate_train_paths)} train + {len(val_paths)} val images...", flush=True)
            train_emb = embed_images(surrogate_train_paths, feature_extractor, preprocess, DEVICE, "train")
            val_emb = embed_images(val_paths, feature_extractor, preprocess, DEVICE, "val")

            probe_layer = PosthocLinearCBM(concept_bank, backbone_name=f"celeba_shortcut_official_train_{task.lower()}_{rate_pct}", n_classes=2).to(DEVICE)
            train_proj = probe_layer.compute_dist(torch.tensor(train_emb, device=DEVICE).float()).detach().cpu().numpy()
            val_proj = probe_layer.compute_dist(torch.tensor(val_emb, device=DEVICE).float()).detach().cpu().numpy()

            train_native_logits = native_logits(surrogate_train_paths, model, preprocess, DEVICE)
            val_native_logits = native_logits(val_paths, model, preprocess, DEVICE)
            train_surrogate = train_native_logits.argmax(axis=1)
            val_surrogate = val_native_logits.argmax(axis=1)

            posthoc_layer = PosthocLinearCBM(
                concept_bank, backbone_name=f"celeba_shortcut_official_train_{task.lower()}_{rate_pct}",
                idx_to_class={0: f"not_{task}", 1: task}, n_classes=2,
            ).to(DEVICE)

            class Args:
                seed = SEED
                lam = 0.0002
                alpha = 0.99

            run_info, weights, bias = run_linear_probe(Args(), (train_proj, train_surrogate), (val_proj, val_surrogate))
            print(f"  train fidelity: {run_info['train_acc']:.2f}%  val fidelity: {run_info['test_acc']:.2f}%", flush=True)

            posthoc_layer.set_weights(weights=weights.astype(np.float32), bias=bias.astype(np.float32))
            with torch.no_grad():
                val_logits = posthoc_layer.forward_projs(torch.tensor(val_proj, device=DEVICE).float())
                pcbm_pred = (val_logits.squeeze(-1) > 0).long().cpu().numpy()
            pcbm_true_label_acc = (pcbm_pred == val_true).mean()
            print(f"  PCBM surrogate's own true-{task}-label accuracy on official-val: {pcbm_true_label_acc:.4f}", flush=True)

            weight_row = weights[0, :] if weights.ndim == 2 else weights
            scores = dict(zip(concept_bank.concept_names, weight_row.tolist()))
            scores_by_task_rate[task][rate_pct] = scores
            for concept_name, w in scores.items():
                all_rows.append((task, rate_pct, concept_name, w))

            model_path = OUT_DIR / f"pcbm_celeba_shortcut_official_train_{task.lower()}_{rate_pct}__surrogate__seed_{SEED}__linear.ckpt"
            torch.save(posthoc_layer, model_path)

            ranked = sorted(scores.items(), key=lambda kv: -abs(kv[1]))
            print(f"  top-5 by |weight|: {[(c, round(s, 4)) for c, s in ranked[:5]]}", flush=True)
            print(f"  mean |weight| across all {len(GROUNDABLE_CONCEPTS)} concepts: "
                  f"{np.mean([abs(s) for s in scores.values()]):.4f}", flush=True)

    out_path = RESULTS_DIR / "pcbm_celeba_official_train_shortcut_experiment.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "rate_pct", "concept_name", "weight"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")

    print("\n=== mean |weight| across all concepts, per task, per rate (the headline decline check) ===")
    for task in TASKS:
        for rate_pct in RATES_PCT:
            mean_abs = np.mean([abs(s) for s in scores_by_task_rate[task][rate_pct].values()])
            print(f"  task={task:<10s} rate={rate_pct:>3d}%: mean |weight| = {mean_abs:.4f}")


if __name__ == "__main__":
    main()
