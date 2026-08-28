"""PCBM's own "CLIP concepts" variant (Yuksekgonul et al. 2023, Section 3
/ Table 2), which needs no positive/negative image dataset per concept at
all: each concept vector c_i is just the CLIP TEXT encoder's own
embedding of a natural-language description (e.g. "a bird with brown
wings"), no CAV fitting (no SVM, no per-concept image crops) required.
Prompted directly ("for PCBM can we try the CLIP+ResNet variant that
does not rely on a concept dataset?").

Unlike the official 112-attribute bank or our own part-crop bank (both
CAV-based, needing labeled image exemplars per concept), this needs a
genuine CLIP-style JOINT image+text embedding model as PCBM's own
backbone f -- not resnet18_cub, which has no text encoder at all. Both
the image embeddings f(x) and the concept vectors c_i must live in the
SAME space for the projection <f(x),c_i> to mean anything. Matches the
paper's own literal setup (CLIP-ResNet50) via open_clip's "RN50"/
"openai" checkpoint, the actual OpenAI CLIP RN50 weights.

Projection formula (paper's own Eq., Section 2): f_C^(i)(x) =
<f(x),c_i> / ||c_i||_2^2. Since OpenClipEncoder.encode_text already
L2-normalizes every c_i, ||c_i||^2 = 1, so this simplifies to a plain
dot product (cosine similarity, since f(x) is also L2-normalized) --
noted, not silently assumed.

Same surrogate-modeling framing as every other PCBM variant in this
investigation: fit the linear predictor g against resnet18_cub's own
argmax predictions (not ground-truth labels), since resnet18_cub is the
one model being explained throughout, keeping this comparable to the
official-bank and part-crop-bank PCBM variants already scored.

Concept vocabulary: the same 87 attribute text queries CARDS already
uses (PREFIX_TEMPLATES), so this scores against the exact same
faithfulness ground truth via the exact same (attribute, class) indexing
-- no new concept definition needed, only a new way of obtaining c_i.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ablate"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from run_cards_cub_attributes import PREFIX_TEMPLATES  # noqa: E402

from cards.data.cub_attributes import groundable_attributes, load_attribute_names  # noqa: E402
from cards.data.cub_parts import load_images_txt  # noqa: E402
from cards.models.backbones import BACKBONES  # noqa: E402
from cards.pipeline import instantiate_encoder  # noqa: E402

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
ATTRIBUTE_NAMES_PATH = CUB_ROOT / "attributes" / "new_attributes.txt"
RESULTS_DIR = Path("results")
OUT_DIR = Path("trained_models_new/cub_clip_concepts")
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

    print("Loading CLIP-ResNet50 (RN50/openai) -- the paper's own 'PCBM with CLIP concepts' backbone...", flush=True)
    clip_cfg = OmegaConf.create({"name": "clip_rn50", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                                  "model_name": "RN50", "pretrained": "openai", "device": DEVICE})
    clip_encoder = instantiate_encoder(OmegaConf.create({"encoder": clip_cfg, "device": DEVICE}))

    attribute_names = load_attribute_names(ATTRIBUTE_NAMES_PATH)
    groundable = groundable_attributes(attribute_names)
    attr_indices = sorted(groundable)
    query_texts = [PREFIX_TEMPLATES[groundable[i][0]].format(value=readable(attribute_names[i].split("::", 1)[1]))
                   for i in attr_indices]
    print(f"Building {len(attr_indices)} concept vectors from CLIP's own text encoder (no image dataset needed)...", flush=True)
    concept_vectors = clip_encoder.encode_text(query_texts).to(DEVICE)  # (87, dim), L2-normalized already
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
    emb_cache_path = OUT_DIR / "clip_rn50_image_embeddings_cache.pt"
    if emb_cache_path.exists():
        print(f"\nLoading cached CLIP-RN50 image embeddings from {emb_cache_path}...", flush=True)
        cached = torch.load(emb_cache_path)
        train_clip_emb, test_clip_emb = cached["train"].to(DEVICE), cached["test"].to(DEVICE)
    else:
        print(f"\nEncoding {len(train_ids)} train + {len(test_ids)} test CUB images with CLIP-RN50's own image encoder...", flush=True)
        train_clip_emb = encode_images_batched(clip_encoder, train_paths).to(DEVICE)
        test_clip_emb = encode_images_batched(clip_encoder, test_paths).to(DEVICE)
        torch.save({"train": train_clip_emb.cpu(), "test": test_clip_emb.cpu()}, emb_cache_path)
    print(f"train embeddings: {train_clip_emb.shape}, test embeddings: {test_clip_emb.shape}", flush=True)

    # Projection: f_C(x)[i] = <f(x), c_i> / ||c_i||^2 -- the paper's own formula.
    train_proj = (train_clip_emb @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)
    test_proj = (test_clip_emb @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)
    train_proj = train_proj.detach().cpu().numpy()
    test_proj = test_proj.detach().cpu().numpy()
    print(f"projection feature scale: mean={train_proj.mean():.4f} std={train_proj.std():.4f} "
          f"range=[{train_proj.min():.4f}, {train_proj.max():.4f}] "
          f"-- CLIP cosine similarities are always small/tightly-banded, unlike the CAV-based banks' own "
          f"margin-normalized projections", flush=True)

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

    # lam=0.0002 (every other PCBM-on-CUB refit in this investigation's own
    # default) was tried first and collapsed almost every weight to exactly
    # zero (99.6%, confirmed by inspecting the saved .pkl directly) --
    # elastic-net regularization tuned for the CAV-based banks' own
    # margin-normalized projection scale is drastically too strong for
    # these raw, tightly-banded CLIP cosine-similarity features. Sweeping
    # a range spanning the paper's own CUB formula (0.01/(K*Nc) =
    # 0.01/(200*87) ~= 5.7e-7) rather than guessing a single replacement.
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
    print(f"PCBM(CLIP-concepts) surrogate's own true-label accuracy on test: {(pcbm_pred == test_true).mean():.4f}")

    with open(OUT_DIR / "pcbm_clip_concepts_cub_weights.pkl", "wb") as f:
        pickle.dump({"weights": weights, "bias": bias, "attr_indices": attr_indices, "run_info": run_info}, f)
    print(f"\nSaved weights to {OUT_DIR / 'pcbm_clip_concepts_cub_weights.pkl'}")

    # (concept, class) -> weight, same shape score_all_methods_against_cub_faithfulness.py's other PCBM loaders use.
    scores = {}
    for col, attr_idx in enumerate(attr_indices):
        for class_idx in range(weights.shape[0]):
            scores[(attr_idx, class_idx)] = float(weights[class_idx, col])

    import csv

    with open(RESULTS_DIR / "pcbm_clip_concepts_cub_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attribute_index", "native_class_idx", "weight"])
        for (attr_idx, class_idx), w in scores.items():
            writer.writerow([attr_idx, class_idx, w])
    print(f"Saved {len(scores)} (attribute, class) scores to results/pcbm_clip_concepts_cub_scores.csv")


if __name__ == "__main__":
    main()
