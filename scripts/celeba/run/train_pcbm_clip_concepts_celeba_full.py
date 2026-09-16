"""PCBM's own "CLIP concepts" variant (Yuksekgonul et al. 2023, Section 3
/ Table 2) on CelebA -- mirrors `scripts/cub/run/train_pcbm_clip_concepts_
cub.py` (CUB v54) exactly, see that script's own docstring for the full
design rationale. Each concept vector c_i is the encoder's own TEXT
embedding of a natural-language description (CARDS' own
CONCEPT_QUERY_TEXT, the same 26 queries used everywhere else in this
track) -- no CAV fitting, no positive/negative image crops per concept
at all, unlike every other PCBM variant here.

Two backbones supported (prompted directly, "can we run the SigLIP
trained and CLIP-Resnet version of PCBM"): SigLIP (ViT-B-16-SigLIP/webli)
and CLIP-ResNet50 (RN50/openai, the paper's own literal setup). Both are
genuine joint image+text encoders (unlike the native `celeba_attractive_
young` ResNet18, which has no text tower) -- required for the projection
<f(x),c_i> to mean anything at all.

Projection formula (paper's own Eq., Section 2): f_C^(i)(x) =
<f(x),c_i> / ||c_i||_2^2. Since `encode_text` already L2-normalizes every
c_i, ||c_i||^2 = 1 (checked, not assumed), so this is a plain dot product
(cosine similarity, f(x) also L2-normalized).

Surrogate-modeling framing matches every other PCBM variant in this
track: fit against `celeba_attractive_young`'s own argmax predictions
per task (Attractive/Young via TASK_SLICES), not ground-truth labels,
then score the resulting weights against the real `celeba_full_
faithfulness.csv` ground truth.

lam is RE-SWEPT, not inherited from the CAV-based default (0.0002) --
CUB v54 found that default catastrophically crushes weights on this
different, tightly-banded cosine-similarity feature scale.

Usage: train_pcbm_clip_concepts_celeba_full.py <siglip|clip_rn50>
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

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.data.celeba import load_celebamask_hq_image_paths, split_celebamask_hq
from cards.data.celeba_attributes import (
    GROUNDABLE_CONCEPTS,
    TARGET_CLASSES,
    load_attribute_labels,
    load_attribute_names,
)
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128

TASK_SLICES: dict[str, slice] = {"Attractive": slice(0, 2), "Young": slice(2, 4)}
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}

BACKBONE_CFGS = {
    "siglip": {"model_name": "ViT-B-16-SigLIP", "pretrained": "webli"},
    "clip_rn50": {"model_name": "RN50", "pretrained": "openai"},
}


def encode_images_batched(encoder, paths: list[Path]) -> torch.Tensor:
    chunks = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch = [Image.open(p).convert("RGB") for p in paths[start : start + BATCH_SIZE]]
        chunks.append(encoder.encode_images(batch))
        if (start // BATCH_SIZE) % 20 == 0:
            print(f"  {start + len(batch)}/{len(paths)}", flush=True)
    return torch.cat(chunks, dim=0)


def load_records_by_task() -> dict[str, list[FaithfulnessResult]]:
    by_task: dict[str, list[FaithfulnessResult]] = {t: [] for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "celeba_full_faithfulness.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]].append(FaithfulnessResult(
                image=row["image"], concept_number=CONCEPT_TO_IDX[row["concept_name"]], category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return by_task


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in BACKBONE_CFGS:
        raise SystemExit(f"Usage: {sys.argv[0]} <{'|'.join(BACKBONE_CFGS)}>")
    backbone_name = sys.argv[1]

    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir = Path(f"trained_models_new/celeba_clip_concepts/{backbone_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {backbone_name} -- PCBM's 'CLIP concepts' backbone (no image concept dataset needed)...", flush=True)
    cfg_dict = {"name": backbone_name, "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                "device": DEVICE, **BACKBONE_CFGS[backbone_name]}
    encoder = instantiate_encoder(OmegaConf.create({"encoder": cfg_dict, "device": DEVICE}))

    query_texts = [CONCEPT_QUERY_TEXT[c] for c in GROUNDABLE_CONCEPTS]
    print(f"Building {len(query_texts)} concept vectors from {backbone_name}'s own text encoder...", flush=True)
    concept_vectors = encoder.encode_text(query_texts).to(DEVICE)  # (26, dim), L2-normalized already
    concept_norms_sq = (concept_vectors ** 2).sum(dim=1)
    print(f"||c_i||^2 range: {concept_norms_sq.min().item():.6f} - {concept_norms_sq.max().item():.6f} "
          f"(should be ~1.0, confirming encode_text's own L2-normalization)", flush=True)

    print("\nLoading CelebAMask-HQ metadata...", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]
    train_hq, val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices)
    train_paths = [image_paths_by_idx[i] for i in train_hq]
    val_paths = [image_paths_by_idx[i] for i in val_hq]
    print(f"{len(train_paths)} train images, {len(val_paths)} val images", flush=True)

    emb_cache_path = out_dir / f"{backbone_name}_image_embeddings_cache.pt"
    if emb_cache_path.exists():
        print(f"\nLoading cached {backbone_name} image embeddings from {emb_cache_path}...", flush=True)
        cached = torch.load(emb_cache_path)
        train_emb, val_emb = cached["train"].to(DEVICE), cached["val"].to(DEVICE)
    else:
        print(f"\nEncoding {len(train_paths)} train + {len(val_paths)} val images with {backbone_name}...", flush=True)
        train_emb = encode_images_batched(encoder, train_paths).to(DEVICE)
        val_emb = encode_images_batched(encoder, val_paths).to(DEVICE)
        torch.save({"train": train_emb.cpu(), "val": val_emb.cpu()}, emb_cache_path)
    print(f"train embeddings: {train_emb.shape}, val embeddings: {val_emb.shape}", flush=True)

    # Projection: f_C(x)[i] = <f(x), c_i> / ||c_i||^2 -- the paper's own formula.
    train_proj = ((train_emb @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)).detach().cpu().numpy()
    val_proj = ((val_emb @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)).detach().cpu().numpy()
    print(f"projection feature scale: mean={train_proj.mean():.4f} std={train_proj.std():.4f} "
          f"range=[{train_proj.min():.4f}, {train_proj.max():.4f}]", flush=True)

    print("\nLoading native celeba_attractive_young's own task logits (surrogate labels + true labels)...", flush=True)
    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()

    def native_logits(paths: list[Path]) -> np.ndarray:
        logits = []
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start : start + BATCH_SIZE]
            batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(DEVICE)
            with torch.no_grad():
                logits.append(native_model(batch).cpu().numpy())
        return np.concatenate(logits, axis=0)

    train_native_logits = native_logits(train_paths)
    val_native_logits = native_logits(val_paths)

    from train_pcbm import run_linear_probe

    records_by_task = load_records_by_task()
    n_concepts = train_proj.shape[1]
    paper_lam = 0.01 / (2 * n_concepts)  # K=2 (binary task), Nc=26
    print(f"\npaper's own elastic-net formula 0.01/(K*Nc) = {paper_lam:.3e} (K=2, Nc={n_concepts})", flush=True)
    lam_candidates = sorted({1e-7, 1e-6, 1e-5, 1e-4, 2e-4, round(paper_lam, 10)})

    all_rows = []  # (task_name, concept_name, weight)
    for task_name in TARGET_CLASSES:
        task_slice = TASK_SLICES[task_name]
        print(f"\n########## target task: {task_name} ##########", flush=True)
        train_surrogate = train_native_logits[:, task_slice].argmax(axis=1)
        val_surrogate = val_native_logits[:, task_slice].argmax(axis=1)

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
        print(f"  best lam={lam:.1e}: train fidelity={run_info['train_acc']:.2f}%, val fidelity={run_info['test_acc']:.2f}%", flush=True)

        weight_row = weights[0, :] if weights.ndim == 2 else weights
        scores = {(CONCEPT_TO_IDX[c], 1): float(weight_row[i]) for i, c in enumerate(GROUNDABLE_CONCEPTS)}
        for c, w in zip(GROUNDABLE_CONCEPTS, weight_row.tolist()):
            all_rows.append((task_name, c, w))

        rho_r = score_method_agreement(records_by_task[task_name], scores, min_samples_per_pair=3)
        sign_r = score_sign_agreement(records_by_task[task_name], scores, min_samples_per_pair=3, method_threshold=0.0)
        print(f"  vs real faithfulness ground truth: n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} "
              f"(p={rho_r.spearman_p:.4g})  sign={sign_r.agreement_frac:.1%} "
              f"({sign_r.n_agree}/{sign_r.n_pairs}, p={sign_r.binom_p:.4g})", flush=True)

        with open(out_dir / f"pcbm_clip_concepts_celeba_{backbone_name}_{task_name.lower()}_weights.pkl", "wb") as f:
            pickle.dump({"weights": weights, "bias": bias, "lam": lam, "run_info": run_info}, f)

    with open(RESULTS_DIR / f"pcbm_clip_concepts_celeba_{backbone_name}_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_task", "concept_name", "weight"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to results/pcbm_clip_concepts_celeba_{backbone_name}_scores.csv")


if __name__ == "__main__":
    main()
