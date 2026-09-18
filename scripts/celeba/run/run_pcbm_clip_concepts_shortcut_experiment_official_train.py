"""Official-train counterpart of `run_pcbm_clip_concepts_shortcut_
experiment.py` -- PCBM's "CLIP concepts" variant (concept vectors and
image embeddings are pure functions of a frozen SigLIP or CLIP-RN50
backbone, no gradient/CAV-fit through the classifier at all) against
the 22 official-train shortcut checkpoints (2 tasks x 11 rates) instead
of the original 4 HQ-trained ones. See `run_pcbm_shortcut_experiment_
official_train.py` for the "conventional" (native ResNet18 CAV-based)
variant.

Image embeddings are classifier- AND task-independent (frozen backbone,
same images regardless of which checkpoint is being explained) -- computed
ONCE per encoder choice and reused across both tasks and all 11 rates,
same reasoning as the original script's own per-rate reuse, just widened
by the task axis. Only each (task, rate)'s own argmax predictions (the
surrogate's training labels) and task-specific val ground truth differ.

Surrogate TRAINING data is a SEED-shuffled sample of N_SURROGATE_TRAIN=
25,500 official CelebA TRAIN (partition==0) images, matching the
original HQ-scale convention's own image count for rough compute parity
(official train has 162,770 images total, ~6.4x more than HQ's) -- not
CelebA-HQ's own train split, confirmed directly ("I do not have the
CELEBA_HQ downloaded there"). VALIDATION is the plain official CelebA
val partition (partition==1) -- no CelebA-HQ-leakage exclusion needed:
these classifiers train on partition==0 only, already disjoint from
partition==1 by construction.

Usage: run_pcbm_clip_concepts_shortcut_experiment_official_train.py <siglip|clip_rn50>
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
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.data.celeba_attributes import (
    GROUNDABLE_CONCEPTS,
    load_attribute_labels,
    load_attribute_names,
)
from cards.pipeline import instantiate_encoder

CELEBA_ROOT = Path(os.environ.get("CELEBA_ROOT", r"C:\Users\btokas\Projects\Datasets\CelebA\celeba"))
RESULTS_DIR = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
CKPT_DIR = Path(os.environ.get("CARDS_CKPT_DIR", "trained_models_new/celeba"))
OUT_ROOT = Path(os.environ.get("CARDS_PCBM_CLIP_OUT_DIR", "trained_models_new/celeba_shortcut_official_train_clip_concepts"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 128
N_SURROGATE_TRAIN = 25_500  # subsample of official train, matching the original HQ-scale image count
RATES_PCT = list(range(0, 101, 10))
TASKS = ["Attractive", "Male"]
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}

BACKBONE_CFGS = {
    "siglip": {"model_name": "ViT-B-16-SigLIP", "pretrained": "webli"},
    "clip_rn50": {"model_name": "RN50", "pretrained": "openai"},
}

from cards.models.backbones import BACKBONES

NATIVE_PREPROCESS = BACKBONES["celeba_attractive_young"].preprocess


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


def build_model(task: str, rate_pct: int) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    ckpt_path = CKPT_DIR / f"resnet18_official_train_{task.lower()}_shortcut_{rate_pct}.pt"
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    return model.eval()


def encode_images_batched(encoder, paths: list[Path]) -> torch.Tensor:
    chunks = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch = [Image.open(p).convert("RGB") for p in paths[start : start + BATCH_SIZE]]
        chunks.append(encoder.encode_images(batch))
        if (start // BATCH_SIZE) % 20 == 0:
            print(f"  {start + len(batch)}/{len(paths)}", flush=True)
    return torch.cat(chunks, dim=0)


def native_logits(paths: list[Path], model: nn.Module, device: str) -> np.ndarray:
    logits = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start : start + BATCH_SIZE]
        batch = torch.stack([NATIVE_PREPROCESS(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        with torch.no_grad():
            logits.append(model(batch).cpu().numpy())
    return np.concatenate(logits, axis=0)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in BACKBONE_CFGS:
        raise SystemExit(f"Usage: {sys.argv[0]} <{'|'.join(BACKBONE_CFGS)}>")
    backbone_name = sys.argv[1]

    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir = OUT_ROOT / backbone_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {backbone_name} -- PCBM's 'CLIP concepts' backbone...", flush=True)
    cfg_dict = {"name": backbone_name, "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                "device": DEVICE, **BACKBONE_CFGS[backbone_name]}
    encoder = instantiate_encoder(OmegaConf.create({"encoder": cfg_dict, "device": DEVICE}))

    query_texts = [CONCEPT_QUERY_TEXT[c] for c in GROUNDABLE_CONCEPTS]
    concept_vectors = encoder.encode_text(query_texts).to(DEVICE)
    concept_norms_sq = (concept_vectors ** 2).sum(dim=1)
    print(f"||c_i||^2 range: {concept_norms_sq.min().item():.6f} - {concept_norms_sq.max().item():.6f}", flush=True)

    print("\nLoading official CelebA metadata...", flush=True)
    all_train_paths, val_paths = load_official_partition_paths()
    official_attr_names = load_attribute_names(CELEBA_ROOT / "list_attr_celeba.txt")
    official_attr_labels = load_attribute_labels(CELEBA_ROOT / "list_attr_celeba.txt")

    rng_py = random.Random(SEED)
    train_paths = list(all_train_paths)
    rng_py.shuffle(train_paths)
    train_paths = train_paths[:N_SURROGATE_TRAIN]
    print(f"{len(train_paths)} surrogate-training images (sampled from {len(all_train_paths)} official train).", flush=True)
    print(f"{len(val_paths)} official-val images.", flush=True)

    emb_cache_path = out_dir / f"{backbone_name}_image_embeddings_cache.pt"
    if emb_cache_path.exists():
        print(f"\nLoading cached {backbone_name} image embeddings from {emb_cache_path} "
              f"(SHARED across both tasks, all 11 rates)...", flush=True)
        cached = torch.load(emb_cache_path)
        train_emb, val_emb = cached["train"].to(DEVICE), cached["val"].to(DEVICE)
    else:
        print(f"\nEncoding {len(train_paths)} train + {len(val_paths)} val images with {backbone_name} "
              f"(ONCE, shared across both tasks, all 11 rates)...", flush=True)
        train_emb = encode_images_batched(encoder, train_paths).to(DEVICE)
        val_emb = encode_images_batched(encoder, val_paths).to(DEVICE)
        torch.save({"train": train_emb.cpu(), "val": val_emb.cpu()}, emb_cache_path)
    print(f"train embeddings: {train_emb.shape}, val embeddings: {val_emb.shape}", flush=True)

    train_proj = ((train_emb @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)).detach().cpu().numpy()
    val_proj = ((val_emb @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)).detach().cpu().numpy()
    print(f"projection feature scale: mean={train_proj.mean():.4f} std={train_proj.std():.4f}", flush=True)

    from train_pcbm import run_linear_probe

    n_concepts = train_proj.shape[1]
    paper_lam = 0.01 / (2 * n_concepts)
    lam_candidates = sorted({1e-7, 1e-6, 1e-5, 1e-4, 2e-4, round(paper_lam, 10)})

    all_rows = []  # (task, rate_pct, concept_name, weight)
    scores_by_task_rate: dict[str, dict[int, dict[str, float]]] = {t: {} for t in TASKS}

    for task in TASKS:
        task_idx = official_attr_names.index(task)
        val_true = np.array([int(official_attr_labels[p.name][task_idx]) for p in val_paths])

        for rate_pct in RATES_PCT:
            print(f"\n=== task={task} rate={rate_pct}% ===", flush=True)
            model = build_model(task, rate_pct).to(DEVICE)

            train_native_logits = native_logits(train_paths, model, DEVICE)
            val_native_logits = native_logits(val_paths, model, DEVICE)
            train_surrogate = train_native_logits.argmax(axis=1)
            val_surrogate = val_native_logits.argmax(axis=1)

            best = None
            for lam in lam_candidates:
                class Args:
                    seed = SEED
                    alpha = 0.99

                Args.lam = lam
                run_info, weights, bias = run_linear_probe(Args(), (train_proj, train_surrogate), (val_proj, val_surrogate))
                nonzero_frac = float((weights != 0).mean())
                print(f"  [lam={lam:.1e}] train_fidelity={run_info['train_acc']:.2f}% val_fidelity={run_info['test_acc']:.2f}% "
                      f"nonzero_weights={nonzero_frac:.1%}", flush=True)
                if best is None or run_info["test_acc"] > best[1]["test_acc"]:
                    best = (lam, run_info, weights, bias)

            lam, run_info, weights, bias = best
            print(f"  best lam={lam:.1e}: train fidelity={run_info['train_acc']:.2f}%, "
                  f"val fidelity={run_info['test_acc']:.2f}%", flush=True)

            weight_row = weights[0, :] if weights.ndim == 2 else weights
            bias_val = bias[0] if hasattr(bias, "__len__") else bias
            pcbm_pred = ((val_proj @ weight_row + bias_val) > 0).astype(int)
            pcbm_true_label_acc = (pcbm_pred == val_true).mean()
            print(f"  PCBM surrogate's own true-{task}-label accuracy on official-val: {pcbm_true_label_acc:.4f}", flush=True)

            scores = dict(zip(GROUNDABLE_CONCEPTS, weight_row.tolist()))
            scores_by_task_rate[task][rate_pct] = scores
            for concept_name, w in scores.items():
                all_rows.append((task, rate_pct, concept_name, w))

            with open(out_dir / f"pcbm_clip_concepts_shortcut_official_train_{backbone_name}_{task.lower()}_{rate_pct}_weights.pkl", "wb") as f:
                pickle.dump({"weights": weights, "bias": bias, "lam": lam, "run_info": run_info}, f)

            ranked = sorted(scores.items(), key=lambda kv: -abs(kv[1]))
            print(f"  top-5 by |weight|: {[(c, round(s, 4)) for c, s in ranked[:5]]}", flush=True)
            print(f"  mean |weight| across all {len(GROUNDABLE_CONCEPTS)} concepts: "
                  f"{np.mean([abs(s) for s in scores.values()]):.4f}", flush=True)

    out_path = RESULTS_DIR / f"pcbm_clip_concepts_shortcut_official_train_experiment_{backbone_name}.csv"
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
