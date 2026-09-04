"""PCBM's "CLIP concepts" variant (see `train_pcbm_clip_concepts_celeba_
full.py`'s own docstring for the full design rationale -- mirrors CUB
v54) against all 4 shortcut-injected classifiers (0%/33%/67%/100%), a
THIRD masking-independent attribution method on the same controlled
shortcut experiment as `run_attribution_shortcut_experiment.py`,
`run_tcav_shortcut_experiment.py`, and the CAV-based `run_pcbm_shortcut_
experiment.py`.

Real efficiency consequence of using text-embedding concepts here, unlike
the CAV-based PCBM shortcut script: concept vectors and image embeddings
are BOTH pure functions of the frozen backbone (SigLIP or CLIP-RN50, see
BACKBONE_CFGS) and the images themselves -- neither depends on which of
the 4 shortcut classifiers is being explained. So both are computed ONCE
and reused across all 4 rates; only each rate's own argmax predictions
(the surrogate's training LABELS) differ per rate, and only the cheap
`run_linear_probe` fit itself is repeated 4x -- no per-rate CAV refit, no
per-rate re-embedding, unlike the CAV-based variant's necessarily-
per-rate-refit design (CAVs there ARE tied to that specific model's own
activation space; here there's no per-model activation space involved at
all).

Surrogate TRAINING data stays CelebA-HQ's own train split (a fitting
resource). VALIDATION is official CelebA val (the standing default for
CelebA evaluation pools), matching `run_pcbm_shortcut_experiment.py`'s
own convention exactly for comparability.

Usage: run_pcbm_clip_concepts_shortcut_experiment.py <siglip|clip_rn50>
"""

from __future__ import annotations

import csv
import pickle
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
from run_cards_celeba_masking_hybrid_official_val_zscore import build_clean_official_val_paths

from cards.data.celeba import load_celebamask_hq_image_paths, split_celebamask_hq
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS, TARGET_CLASSES, load_attribute_labels, load_attribute_names
from cards.pipeline import instantiate_encoder

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
CELEBA_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebA\celeba")
RESULTS_DIR = Path("results")
CKPT_DIR = Path("trained_models_new/celeba")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
RATES_PCT = [0, 33, 67, 100]
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}

BACKBONE_CFGS = {
    "siglip": {"model_name": "ViT-B-16-SigLIP", "pretrained": "webli"},
    "clip_rn50": {"model_name": "RN50", "pretrained": "openai"},
}

# Native-model preprocessing (resnet18, 2-way head) -- matches
# run_pcbm_shortcut_experiment.py's own BACKBONES["celeba_attractive_
# young"].preprocess exactly (architecture-generic, not this
# experiment's own checkpoint).
from cards.models.backbones import BACKBONES

NATIVE_PREPROCESS = BACKBONES["celeba_attractive_young"].preprocess


def build_model(rate_pct: int) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(CKPT_DIR / f"resnet18_attractive_shortcut_{rate_pct}.pt", map_location="cpu")
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

    out_dir = Path(f"trained_models_new/celeba_shortcut_clip_concepts/{backbone_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {backbone_name} -- PCBM's 'CLIP concepts' backbone...", flush=True)
    cfg_dict = {"name": backbone_name, "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                "device": DEVICE, **BACKBONE_CFGS[backbone_name]}
    encoder = instantiate_encoder(OmegaConf.create({"encoder": cfg_dict, "device": DEVICE}))

    query_texts = [CONCEPT_QUERY_TEXT[c] for c in GROUNDABLE_CONCEPTS]
    concept_vectors = encoder.encode_text(query_texts).to(DEVICE)
    concept_norms_sq = (concept_vectors ** 2).sum(dim=1)
    print(f"||c_i||^2 range: {concept_norms_sq.min().item():.6f} - {concept_norms_sq.max().item():.6f}", flush=True)

    print("\nLoading CelebAMask-HQ metadata (surrogate TRAINING data)...", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]
    train_hq, _val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices)
    train_paths = [image_paths_by_idx[i] for i in train_hq]
    print(f"{len(train_paths)} train images (FULL scale).", flush=True)

    print("\nBuilding official-val validation sample...", flush=True)
    val_paths = build_clean_official_val_paths()
    print(f"{len(val_paths)} official-val images.", flush=True)

    emb_cache_path = out_dir / f"{backbone_name}_image_embeddings_cache.pt"
    if emb_cache_path.exists():
        print(f"\nLoading cached {backbone_name} image embeddings from {emb_cache_path} "
              f"(SHARED across all 4 rates -- backbone is frozen, model-independent)...", flush=True)
        cached = torch.load(emb_cache_path)
        train_emb, val_emb = cached["train"].to(DEVICE), cached["val"].to(DEVICE)
    else:
        print(f"\nEncoding {len(train_paths)} train + {len(val_paths)} val images with {backbone_name} "
              f"(ONCE, shared across all 4 rates)...", flush=True)
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
    print(f"\npaper's own elastic-net formula 0.01/(K*Nc) = {paper_lam:.3e} (K=2, Nc={n_concepts})", flush=True)
    lam_candidates = sorted({1e-7, 1e-6, 1e-5, 1e-4, 2e-4, round(paper_lam, 10)})

    all_rows = []  # (rate_pct, concept_name, weight)
    scores_by_rate: dict[int, dict[str, float]] = {}

    for rate_pct in RATES_PCT:
        print(f"\n=== rate={rate_pct}% ===", flush=True)
        model = build_model(rate_pct).to(DEVICE)

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
        scores = dict(zip(GROUNDABLE_CONCEPTS, weight_row.tolist()))
        scores_by_rate[rate_pct] = scores
        for concept_name, w in scores.items():
            all_rows.append((rate_pct, concept_name, w))

        with open(out_dir / f"pcbm_clip_concepts_shortcut_{backbone_name}_{rate_pct}_weights.pkl", "wb") as f:
            pickle.dump({"weights": weights, "bias": bias, "lam": lam, "run_info": run_info}, f)

        ranked = sorted(scores.items(), key=lambda kv: -abs(kv[1]))
        print(f"  top-5 by |weight|: {[(c, round(s, 4)) for c, s in ranked[:5]]}", flush=True)
        print(f"  mean |weight| across all 26 concepts: {np.mean([abs(s) for s in scores.values()]):.4f}", flush=True)

    with open(RESULTS_DIR / f"pcbm_clip_concepts_shortcut_experiment_{backbone_name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rate_pct", "concept_name", "weight"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to results/pcbm_clip_concepts_shortcut_experiment_{backbone_name}.csv")

    print("\n=== mean |weight| across all 26 concepts, per rate (the headline decline check) ===")
    for rate_pct in RATES_PCT:
        mean_abs = np.mean([abs(s) for s in scores_by_rate[rate_pct].values()])
        print(f"  rate={rate_pct:>3d}%: mean |weight| = {mean_abs:.4f}")


if __name__ == "__main__":
    main()
