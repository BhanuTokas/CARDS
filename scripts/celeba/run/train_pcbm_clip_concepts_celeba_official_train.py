"""PCBM's "CLIP concepts" variant (concept vectors and image embeddings
are pure functions of a frozen SigLIP or CLIP-RN50 backbone, no
gradient/CAV-fit through the classifier at all -- mirrors CUB v54 and
`run_pcbm_clip_concepts_shortcut_experiment_official_train.py`'s own
design) for the MAIN (non-shortcut) official-train pipeline, prompted
directly ("The full pipeline with PCBM variants" for ViT-B/16 and
ConvNeXt-Tiny). This variant never existed for the main pipeline before
-- only `fit_celeba_official_train_cavs.py` + `train_pcbm_surrogate_
celeba_official_train.py`'s "conventional" (native-activation CAV-
based) PCBM did, and only the shortcut experiment ever got the CLIP-
concepts treatment.

Two SEPARATE axes, both needed: `CARDS_BACKBONE_NAME` (env var, which
CLASSIFIER's own predictions the surrogate is trained to mimic --
celeba_official_train_attractive_male / _vit / _convnext) and the CLIP
encoder for concept representation (CLI arg, siglip|clip_rn50) -- the
concept vectors/image embeddings never depend on which classifier is
being explained (frozen backbone, computed ONCE and reused across all
3 classifiers if this is re-run for each), only each classifier's own
argmax predictions (the surrogate's training labels) differ.

Surrogate TRAINING data is official CelebA TRAIN (162,770 images,
matching `train_pcbm_surrogate_celeba_official_train.py`'s own
convention for the main pipeline -- NOT the shortcut experiment's own
N=25,500 subsample, since this isn't a per-rate sweep needing compute
parity against anything). VALIDATION is official CelebA val (19,867
images), for the fidelity diagnostic only.

Output: a plain (task, concept_name, weight) CSV per (backbone, clip
encoder) combination -- simpler than mimicking the conventional PCBM's
own PosthocLinearCBM checkpoint save/load, since nothing downstream
needs this as a loadable nn.Module, just the per-concept weight vector
`score_all_methods_against_official_train_faithfulness.py` reads.

Usage: train_pcbm_clip_concepts_celeba_official_train.py <siglip|clip_rn50>
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder

CELEBA_ROOT = Path(os.environ.get("CELEBA_ROOT", r"C:\Users\btokas\Projects\Datasets\CelebA\celeba"))
RESULTS_DIR = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
OUT_DIR = Path(os.environ.get("CARDS_TRAINED_MODELS_DIR", "trained_models_new/celeba_full"))
BACKBONE_NAME = os.environ.get("CARDS_BACKBONE_NAME", "celeba_official_train_attractive_male")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 128
TASKS = ["Attractive", "Male"]
TASK_SLICES: dict[str, slice] = {"Attractive": slice(0, 2), "Male": slice(2, 4)}

BACKBONE_CFGS = {
    "siglip": {"model_name": "ViT-B-16-SigLIP", "pretrained": "webli"},
    "clip_rn50": {"model_name": "RN50", "pretrained": "openai"},
}


def load_attr_names(path: Path) -> list[str]:
    return path.read_text().splitlines()[1].split()


def load_attr_labels(path: Path) -> dict[str, np.ndarray]:
    lines = path.read_text().splitlines()
    result: dict[str, np.ndarray] = {}
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split()
        result[parts[0]] = np.array([v == "1" for v in parts[1:]], dtype=bool)
    return result


def encode_images_batched(encoder, filenames: list[str]) -> torch.Tensor:
    chunks = []
    for start in range(0, len(filenames), BATCH_SIZE):
        batch = [Image.open(CELEBA_ROOT / "img_align_celeba" / f).convert("RGB") for f in filenames[start:start + BATCH_SIZE]]
        chunks.append(encoder.encode_images(batch))
        if (start // BATCH_SIZE) % 50 == 0:
            print(f"  {start + len(batch)}/{len(filenames)}", flush=True)
    return torch.cat(chunks, dim=0)


def native_task_logits(filenames: list[str], native_model, preprocess, device: str) -> np.ndarray:
    logits = []
    for start in range(0, len(filenames), BATCH_SIZE):
        batch_files = filenames[start:start + BATCH_SIZE]
        batch = torch.stack([preprocess(Image.open(CELEBA_ROOT / "img_align_celeba" / f).convert("RGB")) for f in batch_files]).to(device)
        with torch.no_grad():
            logits.append(native_model(batch).cpu().numpy())
    return np.concatenate(logits, axis=0)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in BACKBONE_CFGS:
        raise SystemExit(f"Usage: {sys.argv[0]} <{'|'.join(BACKBONE_CFGS)}>")
    clip_backbone_name = sys.argv[1]

    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir = OUT_DIR / BACKBONE_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"BACKBONE_NAME={BACKBONE_NAME}  clip_backbone={clip_backbone_name}", flush=True)

    print(f"Loading {clip_backbone_name} -- PCBM's 'CLIP concepts' backbone...", flush=True)
    cfg_dict = {"name": clip_backbone_name, "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                "device": DEVICE, **BACKBONE_CFGS[clip_backbone_name]}
    encoder = instantiate_encoder(OmegaConf.create({"encoder": cfg_dict, "device": DEVICE}))

    query_texts = [CONCEPT_QUERY_TEXT[c] for c in GROUNDABLE_CONCEPTS]
    concept_vectors = encoder.encode_text(query_texts).to(DEVICE)
    concept_norms_sq = (concept_vectors ** 2).sum(dim=1)

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
    print(f"{len(train_files)} train, {len(val_files)} val images.", flush=True)

    spec = BACKBONES[BACKBONE_NAME]
    native_model = spec.load_native().to(DEVICE).eval()

    emb_cache_path = out_dir / f"{clip_backbone_name}_image_embeddings_cache.pt"
    if emb_cache_path.exists():
        print(f"Loading cached {clip_backbone_name} image embeddings from {emb_cache_path}...", flush=True)
        cached = torch.load(emb_cache_path)
        train_emb, val_emb = cached["train"].to(DEVICE), cached["val"].to(DEVICE)
    else:
        print(f"Encoding {len(train_files)} train + {len(val_files)} val images with {clip_backbone_name}...", flush=True)
        train_emb = encode_images_batched(encoder, train_files).to(DEVICE)
        val_emb = encode_images_batched(encoder, val_files).to(DEVICE)
        torch.save({"train": train_emb.cpu(), "val": val_emb.cpu()}, emb_cache_path)
    print(f"train embeddings: {train_emb.shape}, val embeddings: {val_emb.shape}", flush=True)

    train_proj = ((train_emb @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)).detach().cpu().numpy()
    val_proj = ((val_emb @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)).detach().cpu().numpy()

    train_native_logits = native_task_logits(train_files, native_model, spec.preprocess, DEVICE)
    val_native_logits = native_task_logits(val_files, native_model, spec.preprocess, DEVICE)

    from train_pcbm import run_linear_probe

    n_concepts = train_proj.shape[1]
    paper_lam = 0.01 / (2 * n_concepts)
    lam_candidates = sorted({1e-7, 1e-6, 1e-5, 1e-4, 2e-4, round(paper_lam, 10)})

    all_rows = []  # (task, concept_name, weight)
    for task_name in TASKS:
        task_slice = TASK_SLICES[task_name]
        task_attr_idx = attr_names.index(task_name)
        train_surrogate = train_native_logits[:, task_slice].argmax(axis=1)
        val_surrogate = val_native_logits[:, task_slice].argmax(axis=1)
        val_true = np.array([int(attr_labels[f][task_attr_idx]) for f in val_files])

        print(f"\n########## target task: {task_name} ##########", flush=True)
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
        print(f"  PCBM surrogate's own true-{task_name}-label accuracy on val: {pcbm_true_label_acc:.4f}", flush=True)

        for concept_name, w in zip(GROUNDABLE_CONCEPTS, weight_row.tolist()):
            all_rows.append((task_name, concept_name, w))

        ranked = sorted(zip(GROUNDABLE_CONCEPTS, weight_row.tolist()), key=lambda kv: -abs(kv[1]))
        print(f"  top-5 by |weight|: {[(c, round(w, 4)) for c, w in ranked[:5]]}", flush=True)

    out_path = RESULTS_DIR / f"pcbm_clip_concepts_celeba_official_train{BACKBONE_NAME.replace('celeba_official_train_attractive_male', '')}_{clip_backbone_name}_weights.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "concept_name", "weight"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
