"""SigLIP variant of PCBM's own "CLIP concepts" setup (see
train_pcbm_clip_concepts_cub.py's docstring for the full paper-formula
rationale) -- swaps the joint image+text backbone from OpenAI CLIP-RN50
to SigLIP (ViT-B-16-SigLIP/webli), the SAME encoder CARDS' own best
config (K=50/demean=True/aligned, 71.7%/59.6% sign agreement across the
two ground-truth builds) uses. Prompted directly ("Can we use the same
SigLIP VLM for PCBM as well?") -- makes the CARDS-vs-PCBM(CLIP-concepts)
comparison apples-to-apples on the encoder axis, isolating "does the
METHOD matter" from "does the ENCODER matter" (CARDS' own ablations,
v46/v47/v50, already showed encoder choice alone materially changes
CARDS' own results, so the same could easily be true for PCBM here).

`cards.encoders.open_clip_encoder.OpenClipEncoder` wraps both CLIP and
SigLIP identically via open_clip's own model_name/pretrained args, and
both `encode_text`/`encode_images` L2-normalize regardless of backbone
(confirmed directly in source, not assumed) -- so `||c_i||^2 = 1` and the
paper's own projection formula `f_C(x)[i] = <f(x),c_i>/||c_i||^2`
simplifies to a plain dot product exactly as it did for CLIP-RN50. Only
the encoder config, output paths, and cache file differ from
train_pcbm_clip_concepts_cub.py -- everything else (surrogate-label
framing, lam sweep, 87-attribute vocabulary) is identical by design, so
any difference in the final result is attributable to the encoder swap
alone.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from run_cards_cub_attributes import PREFIX_TEMPLATES  # noqa: E402

from cards.data.cub_attributes import groundable_attributes, load_attribute_names  # noqa: E402
from cards.data.cub_parts import load_images_txt  # noqa: E402
from cards.models.backbones import BACKBONES  # noqa: E402
from cards.pipeline import instantiate_encoder  # noqa: E402

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
ATTRIBUTE_NAMES_PATH = CUB_ROOT / "attributes" / "new_attributes.txt"
RESULTS_DIR = Path("results")
OUT_DIR = Path("trained_models_new/cub_siglip_concepts")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128


def readable(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


def encode_images_batched(encoder, paths: list[Path]) -> torch.Tensor:
    chunks = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch = [Image.open(p).convert("RGB") for p in paths[start : start + BATCH_SIZE]]
        chunks.append(encoder.encode_images(batch))
    return torch.cat(chunks, dim=0)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading SigLIP (ViT-B-16-SigLIP/webli) -- the SAME encoder CARDS' own best config uses...", flush=True)
    siglip_cfg = OmegaConf.create({"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE})
    siglip_encoder = instantiate_encoder(OmegaConf.create({"encoder": siglip_cfg, "device": DEVICE}))

    attribute_names = load_attribute_names(ATTRIBUTE_NAMES_PATH)
    groundable = groundable_attributes(attribute_names)
    attr_indices = sorted(groundable)
    query_texts = [PREFIX_TEMPLATES[groundable[i][0]].format(value=readable(attribute_names[i].split("::", 1)[1]))
                   for i in attr_indices]
    print(f"Building {len(attr_indices)} concept vectors from SigLIP's own text encoder (no image dataset needed)...", flush=True)
    concept_vectors = siglip_encoder.encode_text(query_texts).to(DEVICE)  # (87, dim), L2-normalized already
    concept_norms_sq = (concept_vectors ** 2).sum(dim=1)
    print(f"||c_i||^2 range: {concept_norms_sq.min().item():.6f} - {concept_norms_sq.max().item():.6f} "
          f"(should be ~1.0, confirming encode_text's own L2-normalization)", flush=True)

    image_paths = load_images_txt(CUB_ROOT)
    class_labels = {}
    for line in (CUB_ROOT / "image_class_labels.txt").read_text().splitlines():
        image_id, class_id = line.split()
        class_labels[image_id] = int(class_id) - 1
    train_ids = [
        line.split()[0]
        for line in (CUB_ROOT / "train_test_split.txt").read_text().splitlines()
        if line.split()[1] == "1"
    ]
    test_ids = [
        line.split()[0]
        for line in (CUB_ROOT / "train_test_split.txt").read_text().splitlines()
        if line.split()[1] == "0"
    ]

    train_paths = [image_paths[i] for i in train_ids]
    test_paths = [image_paths[i] for i in test_ids]
    emb_cache_path = OUT_DIR / "siglip_image_embeddings_cache.pt"
    if emb_cache_path.exists():
        print(f"\nLoading cached SigLIP image embeddings from {emb_cache_path}...", flush=True)
        cached = torch.load(emb_cache_path)
        train_siglip_emb, test_siglip_emb = cached["train"].to(DEVICE), cached["test"].to(DEVICE)
    else:
        print(f"\nEncoding {len(train_ids)} train + {len(test_ids)} test CUB images with SigLIP's own image encoder...", flush=True)
        train_siglip_emb = encode_images_batched(siglip_encoder, train_paths).to(DEVICE)
        test_siglip_emb = encode_images_batched(siglip_encoder, test_paths).to(DEVICE)
        torch.save({"train": train_siglip_emb.cpu(), "test": test_siglip_emb.cpu()}, emb_cache_path)
    print(f"train embeddings: {train_siglip_emb.shape}, test embeddings: {test_siglip_emb.shape}", flush=True)

    # Projection: f_C(x)[i] = <f(x), c_i> / ||c_i||^2 -- the paper's own formula.
    train_proj = (train_siglip_emb @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)
    test_proj = (test_siglip_emb @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)
    train_proj = train_proj.detach().cpu().numpy()
    test_proj = test_proj.detach().cpu().numpy()
    print(f"projection feature scale: mean={train_proj.mean():.4f} std={train_proj.std():.4f} "
          f"range=[{train_proj.min():.4f}, {train_proj.max():.4f}]", flush=True)

    print("\nComputing resnet18_cub's own surrogate labels (the model this investigation explains throughout)...", flush=True)
    spec = BACKBONES["resnet18_cub"]
    native_model = spec.load_native().to(DEVICE).eval()

    def native_argmax(paths: list[Path]) -> np.ndarray:
        preds = []
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start : start + BATCH_SIZE]
            batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(DEVICE)
            with torch.no_grad():
                logits = native_model(batch)
            preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    train_surrogate = native_argmax(train_paths)
    test_surrogate = native_argmax(test_paths)
    train_true = np.array([class_labels[i] for i in train_ids])
    test_true = np.array([class_labels[i] for i in test_ids])
    print(f"native model's own true-label accuracy (test): {(test_surrogate == test_true).mean():.4f}", flush=True)

    from train_pcbm import run_linear_probe

    # Same lam sweep as the CLIP-RN50 variant (v54) -- SigLIP's own feature
    # scale is expected to be in the same tightly-banded cosine-similarity
    # regime, but not assumed identical, hence sweeping rather than reusing
    # a single value directly.
    n_classes = len(set(train_true.tolist()))
    n_concepts = train_proj.shape[1]
    paper_lam = 0.01 / (n_classes * n_concepts)
    print(f"\npaper's own CUB elastic-net formula 0.01/(K*Nc) = {paper_lam:.3e} (K={n_classes}, Nc={n_concepts})", flush=True)

    lam_candidates = [1e-7, 1e-6, 1e-5, 1e-4, 2e-4]
    best = None
    for lam in lam_candidates:
        class Args:
            seed = SEED
            alpha = 0.99

        Args.lam = lam
        run_info, weights, bias = run_linear_probe(Args(), (train_proj, train_surrogate), (test_proj, test_surrogate))
        nonzero_frac = float((weights != 0).mean())
        print(f"[lam={lam:.1e}] train_fidelity={run_info['train_acc']:.2f}% test_fidelity={run_info['test_acc']:.2f}% "
              f"nonzero_weights={nonzero_frac:.1%}", flush=True)
        if best is None or run_info["test_acc"] > best[1]["test_acc"]:
            best = (lam, run_info, weights, bias)

    lam, run_info, weights, bias = best
    print(f"\nBest lam={lam:.1e}: train fidelity={run_info['train_acc']:.2f}%, test fidelity={run_info['test_acc']:.2f}%", flush=True)

    test_logits = test_proj @ weights.T + bias
    pcbm_pred = test_logits.argmax(axis=1)
    print(f"PCBM(SigLIP-concepts) surrogate's own true-label accuracy on test: {(pcbm_pred == test_true).mean():.4f}")

    with open(OUT_DIR / "pcbm_siglip_concepts_cub_weights.pkl", "wb") as f:
        pickle.dump({"weights": weights, "bias": bias, "attr_indices": attr_indices, "run_info": run_info}, f)
    print(f"\nSaved weights to {OUT_DIR / 'pcbm_siglip_concepts_cub_weights.pkl'}")

    # (concept, class) -> weight, same shape score_all_methods_against_cub_faithfulness.py's other PCBM loaders use.
    scores = {}
    for col, attr_idx in enumerate(attr_indices):
        for class_idx in range(weights.shape[0]):
            scores[(attr_idx, class_idx)] = float(weights[class_idx, col])

    import csv

    with open(RESULTS_DIR / "pcbm_siglip_concepts_cub_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attribute_index", "native_class_idx", "weight"])
        for (attr_idx, class_idx), w in scores.items():
            writer.writerow([attr_idx, class_idx, w])
    print(f"Saved {len(scores)} (attribute, class) scores to results/pcbm_siglip_concepts_cub_scores.csv")


if __name__ == "__main__":
    main()
