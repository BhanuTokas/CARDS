"""PCBM's "CLIP concepts" variant (Yuksekgonul et al. 2023, Section 3 /
Table 2) fit against the NEW official-train classifier
(`celeba_official_train_attractive_male`), prompted directly ("Can you
run the PCBM with CLIP and SigLIP variants"). Mirrors
`train_pcbm_clip_concepts_celeba_full.py`'s own design (same projection
formula, same lam sweep) but sources images from standard CelebA's
OFFICIAL train/val partitions (162,770 / 19,867 images) instead of
CelebAMask-HQ's train_hq/val_hq -- matching what this classifier
actually trained on (same data-source switch as
`train_pcbm_surrogate_celeba_official_train.py`) -- and scores against
BOTH `celeba_official_train_faithfulness_{non_overlapping,
previously_used}.csv` ground-truth versions.

Usage: train_pcbm_clip_concepts_celeba_official_train.py <siglip|clip_rn50>
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

from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

CELEBA_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebA\celeba")
RESULTS_DIR = Path("results")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128

TASKS = ["Attractive", "Male"]
TASK_SLICES: dict[str, slice] = {"Attractive": slice(0, 2), "Male": slice(2, 4)}
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
GT_VERSIONS = {
    "non_overlapping": "celeba_official_train_faithfulness_non_overlapping.csv",
    "previously_used": "celeba_official_train_faithfulness_previously_used.csv",
}

BACKBONE_CFGS = {
    "siglip": {"model_name": "ViT-B-16-SigLIP", "pretrained": "webli"},
    "clip_rn50": {"model_name": "RN50", "pretrained": "openai"},
}


def load_attr_names(path: Path) -> list[str]:
    return Path(path).read_text().splitlines()[1].split()


def load_partition(path: Path) -> dict[str, str]:
    partition = {}
    with open(path) as f:
        for line in f:
            fname, part = line.split()
            partition[fname] = part
    return partition


def encode_images_batched(encoder, filenames: list[str]) -> torch.Tensor:
    chunks = []
    for start in range(0, len(filenames), BATCH_SIZE):
        batch_files = filenames[start : start + BATCH_SIZE]
        batch = [Image.open(CELEBA_ROOT / "img_align_celeba" / f).convert("RGB") for f in batch_files]
        chunks.append(encoder.encode_images(batch))
        if (start // BATCH_SIZE) % 200 == 0:
            print(f"  {start + len(batch_files)}/{len(filenames)}", flush=True)
    return torch.cat(chunks, dim=0)


def load_records(gt_filename: str, task_name: str) -> list[FaithfulnessResult]:
    records = []
    with open(RESULTS_DIR / gt_filename, newline="") as f:
        for row in csv.DictReader(f):
            if row["target_task"] != task_name:
                continue
            records.append(FaithfulnessResult(
                image=row["image"], concept_number=CONCEPT_TO_IDX[row["concept_name"]], category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return records


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in BACKBONE_CFGS:
        raise SystemExit(f"Usage: {sys.argv[0]} <{'|'.join(BACKBONE_CFGS)}>")
    backbone_name = sys.argv[1]

    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir = Path(f"trained_models_new/celeba_clip_concepts_official_train/{backbone_name}")
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

    print("\nLoading official CelebA metadata...", flush=True)
    attr_names = load_attr_names(CELEBA_ROOT / "list_attr_celeba.txt")
    partition = load_partition(CELEBA_ROOT / "list_eval_partition.txt")
    train_files = sorted(f for f, p in partition.items() if p == "0")
    val_files = sorted(f for f, p in partition.items() if p == "1")
    print(f"{len(train_files)} train images, {len(val_files)} val images", flush=True)

    emb_cache_path = out_dir / f"{backbone_name}_image_embeddings_cache.pt"
    if emb_cache_path.exists():
        print(f"\nLoading cached {backbone_name} image embeddings from {emb_cache_path}...", flush=True)
        cached = torch.load(emb_cache_path)
        train_emb, val_emb = cached["train"].to(DEVICE), cached["val"].to(DEVICE)
    else:
        print(f"\nEncoding {len(train_files)} train + {len(val_files)} val images with {backbone_name}...", flush=True)
        train_emb = encode_images_batched(encoder, train_files).to(DEVICE)
        val_emb = encode_images_batched(encoder, val_files).to(DEVICE)
        torch.save({"train": train_emb.cpu(), "val": val_emb.cpu()}, emb_cache_path)
    print(f"train embeddings: {train_emb.shape}, val embeddings: {val_emb.shape}", flush=True)

    train_proj = ((train_emb @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)).detach().cpu().numpy()
    val_proj = ((val_emb @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)).detach().cpu().numpy()
    print(f"projection feature scale: mean={train_proj.mean():.4f} std={train_proj.std():.4f} "
          f"range=[{train_proj.min():.4f}, {train_proj.max():.4f}]", flush=True)

    print("\nLoading celeba_official_train_attractive_male's own task logits (surrogate labels)...", flush=True)
    spec = BACKBONES["celeba_official_train_attractive_male"]
    native_model = spec.load_native().to(DEVICE).eval()

    def native_logits(filenames: list[str]) -> np.ndarray:
        logits = []
        for start in range(0, len(filenames), BATCH_SIZE):
            batch_files = filenames[start : start + BATCH_SIZE]
            batch = torch.stack([
                spec.preprocess(Image.open(CELEBA_ROOT / "img_align_celeba" / f).convert("RGB")) for f in batch_files
            ]).to(DEVICE)
            with torch.no_grad():
                logits.append(native_model(batch).cpu().numpy())
        return np.concatenate(logits, axis=0)

    train_native_logits = native_logits(train_files)
    val_native_logits = native_logits(val_files)

    from train_pcbm import run_linear_probe

    n_concepts = train_proj.shape[1]
    paper_lam = 0.01 / (2 * n_concepts)  # K=2 (binary task), Nc=26
    print(f"\npaper's own elastic-net formula 0.01/(K*Nc) = {paper_lam:.3e} (K=2, Nc={n_concepts})", flush=True)
    lam_candidates = sorted({1e-7, 1e-6, 1e-5, 1e-4, 2e-4, round(paper_lam, 10)})

    all_rows = []  # (task_name, concept_name, weight)
    for task_name in TASKS:
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

        for gt_name, gt_filename in GT_VERSIONS.items():
            records = load_records(gt_filename, task_name)
            rho_r = score_method_agreement(records, scores, min_samples_per_pair=3)
            sign_r = score_sign_agreement(records, scores, min_samples_per_pair=3, method_threshold=0.0)
            if rho_r is None:
                print(f"  [{gt_name}] too few pairs", flush=True)
                continue
            print(f"  [{gt_name}] n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} (p={rho_r.spearman_p:.4g})  "
                  f"sign={sign_r.agreement_frac:.1%} ({sign_r.n_agree}/{sign_r.n_pairs}, p={sign_r.binom_p:.4g})", flush=True)

        with open(out_dir / f"pcbm_clip_concepts_celeba_official_train_{backbone_name}_{task_name.lower()}_weights.pkl", "wb") as f:
            pickle.dump({"weights": weights, "bias": bias, "lam": lam, "run_info": run_info}, f)

    with open(RESULTS_DIR / f"pcbm_clip_concepts_celeba_official_train_{backbone_name}_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_task", "concept_name", "weight"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to results/pcbm_clip_concepts_celeba_official_train_{backbone_name}_scores.csv")


if __name__ == "__main__":
    main()
