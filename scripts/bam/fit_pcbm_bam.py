"""PCBM on BAM's obj and scene models, in the paper's three variants.

  own:       CAVs on the model's own pooled embedding (post_hoc_cbm's get_cavs,
             linear SVC, C=0.1, 50+50 fit and 50+50 held out), concept score =
             margin to each CAV, elastic net lam=2e-4, alpha=0.99. Matches
             fit_celeba_official_train_cavs.py + train_pcbm_surrogate_celeba_official_train.py.
  siglip,
  clip_rn50: text concepts on the VLM's image embedding, concept score =
             projection emb.c/|c|^2, elastic net with lam picked by val
             fidelity. Matches train_pcbm_clip_concepts_celeba_official_train.py.

Every variant fits the model's own predictions (argmax logits) on all 90,000
BAM training composites, and reports fidelity on the 10,000 held-out ones.
The score for (class, concept) is that class's weight on that concept.

Inputs under <bam-root>: train/ and val/ manifests and composites,
baselines/pcbm_cav_sets.csv, and baselines/tf/<model>_{train,val,pcbm_cav_sets}.npz
(logits and embeddings from store_bam_outputs.py).
Writes results/bam_pcbm_scores.csv and bam_pcbm_fits.csv, holding only the variants
computed in that run.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, os.environ.get("POST_HOC_CBM", "../post_hoc_cbm"))

from concepts.concept_utils import get_cavs
from run_hns_bam_masks import CONCEPT_QUERY_TEXT, CONCEPTS
from sample_bam_concept_sets import OBJECTS, SCENES
from train_pcbm import run_linear_probe

from cards.pipeline import instantiate_encoder

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = {"obj": OBJECTS, "scene": SCENES}
VLM_CFGS = {
    "siglip": {"model_name": "ViT-B-16-SigLIP", "pretrained": "webli"},
    "clip_rn50": {"model_name": "RN50", "pretrained": "openai"},
}
OWN_C, OWN_N_SAMPLES, OWN_LAM = 0.1, 50, 2e-4
ALPHA = 0.99
BATCH_SIZE = 256


class Args:
    seed = SEED
    alpha = ALPHA
    lam = OWN_LAM


def read_files(path: Path) -> list:
    with open(path, newline="") as handle:
        return [row["file"] for row in csv.DictReader(handle)]


def own_concept_bank(tf_dir: Path, model: str, cav_rows: list) -> tuple:
    """CAV, intercept and norm per concept, from the model's pooled embedding."""
    data = np.load(tf_dir / f"{model}_pcbm_cav_sets.npz")
    index = {name: i for i, name in enumerate(data["files"])}
    emb = data["embeddings"]
    cavs, intercepts = [], []
    for concept in CONCEPTS:
        pos = emb[[index[r["file"]] for r in cav_rows if r["concept"] == concept and r["label"] == "1"]]
        neg = emb[[index[r["file"]] for r in cav_rows if r["concept"] == concept and r["label"] == "0"]]
        n = OWN_N_SAMPLES
        x_train, x_val = np.concatenate([pos[:n], neg[:n]]), np.concatenate([pos[n:], neg[n:]])
        y_train = np.concatenate([np.ones(len(pos[:n])), np.zeros(len(neg[:n]))])
        y_val = np.concatenate([np.ones(len(pos[n:])), np.zeros(len(neg[n:]))])
        coef, train_acc, test_acc, intercept, _ = get_cavs(x_train, y_train, x_val, y_val, OWN_C)
        print(f"  {model} CAV {concept}: train {train_acc:.2f}, held-out {test_acc:.2f}", flush=True)
        cavs.append(coef[0])
        intercepts.append(intercept[0])
    cavs = np.stack(cavs)
    return cavs, np.array(intercepts), np.linalg.norm(cavs, axis=1)


def vlm_embeddings(encoder, image_dir: Path, files: list) -> np.ndarray:
    chunks = []
    for start in range(0, len(files), BATCH_SIZE):
        batch = [Image.open(image_dir / f).convert("RGB") for f in files[start:start + BATCH_SIZE]]
        chunks.append(encoder.encode_images(batch).float().cpu())
        if (start // BATCH_SIZE) % 50 == 0:
            print(f"  embedded {start + len(batch)}/{len(files)}", flush=True)
    return torch.cat(chunks).numpy()


def vlm_features(bam_root: Path, variant: str, train_files: list, val_files: list) -> tuple:
    """VLM image embeddings (cached) and text concept vectors for one variant."""
    cache = bam_root / "baselines" / f"{variant}_embeddings.npz"
    concept_path = cache.with_suffix(".concepts.npy")
    if not cache.exists():
        encoder = instantiate_encoder(OmegaConf.create({"device": DEVICE, "encoder": {
            "name": variant, "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
            "device": DEVICE, **VLM_CFGS[variant]}}))
        emb = {"train": vlm_embeddings(encoder, bam_root / "train" / "composite", train_files),
               "val": vlm_embeddings(encoder, bam_root / "val" / "composite", val_files)}
        np.savez(cache, **emb)
        concepts = encoder.encode_text([CONCEPT_QUERY_TEXT[c] for c in CONCEPTS]).float().cpu().numpy()
        np.save(concept_path, concepts)
    cached = np.load(cache)
    return {"train": cached["train"], "val": cached["val"]}, np.load(concept_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam-root", type=Path, default=Path(os.environ.get("BAM_ROOT", "../Datasets/BAM")))
    parser.add_argument("--results-dir", type=Path, default=Path(os.environ.get("CARDS_RESULTS_DIR", "results")))
    parser.add_argument("--variants", nargs="+", default=["own", "siglip", "clip_rn50"])
    parser.add_argument("--embed-only", action="store_true",
                        help="Only cache the VLM embeddings (needs no TensorFlow outputs).")
    args = parser.parse_args()
    tf_dir = args.bam_root / "baselines" / "tf"
    train_files = read_files(args.bam_root / "train" / "manifest.csv")
    val_files = read_files(args.bam_root / "val" / "manifest.csv")
    if args.embed_only:
        for variant in args.variants:
            if variant in VLM_CFGS:
                vlm_features(args.bam_root, variant, train_files, val_files)
        return

    targets = {}
    for model in CLASSES:
        for split, files in (("train", train_files), ("val", val_files)):
            data = np.load(tf_dir / f"{model}_{split}.npz")
            assert list(data["files"]) == files, f"{model}_{split}.npz is out of order"
            targets[model, split] = data["logits"].argmax(axis=1)

    out_rows, fit_rows = [], []
    for variant in args.variants:
        print(f"\n=== PCBM variant: {variant} ===", flush=True)
        features = {}
        if variant == "own":
            with open(args.bam_root / "baselines" / "pcbm_cav_sets.csv", newline="") as handle:
                cav_rows = list(csv.DictReader(handle))
            for model in CLASSES:
                cavs, intercepts, norms = own_concept_bank(tf_dir, model, cav_rows)
                for split in ("train", "val"):
                    emb = np.load(tf_dir / f"{model}_{split}.npz")["embeddings"]
                    features[model, split] = (emb @ cavs.T + intercepts) / norms
            lams = [OWN_LAM]
        else:
            emb, concepts = vlm_features(args.bam_root, variant, train_files, val_files)
            norms_sq = (concepts ** 2).sum(axis=1)
            for model in CLASSES:
                for split in ("train", "val"):
                    features[model, split] = (emb[split] @ concepts.T) / norms_sq
            n_classes, n_concepts = 10, len(CONCEPTS)
            lams = sorted({1e-7, 1e-6, 1e-5, 1e-4, 2e-4, round(0.01 / (n_classes * n_concepts), 10)})

        for model, classes in CLASSES.items():
            best = None
            for lam in lams:
                Args.lam = lam
                info, weights, _bias = run_linear_probe(
                    Args(), (features[model, "train"], targets[model, "train"]),
                    (features[model, "val"], targets[model, "val"]))
                print(f"  {model} lam={lam:.1e}: train fidelity {info['train_acc']:.2f}%, "
                      f"val fidelity {info['test_acc']:.2f}%", flush=True)
                if best is None or info["test_acc"] > best[1]["test_acc"]:
                    best = (lam, info, weights)
            lam, info, weights = best
            fit_rows.append({"variant": variant, "model": model, "lam": lam,
                             "train_fidelity": info["train_acc"], "val_fidelity": info["test_acc"]})
            for k, cls in enumerate(classes):
                for j, concept in enumerate(CONCEPTS):
                    out_rows.append({"variant": variant, "model": model, "class": cls,
                                     "concept": concept, "weight": float(weights[k, j])})

    args.results_dir.mkdir(exist_ok=True)
    for name, rows in (("bam_pcbm_scores.csv", out_rows), ("bam_pcbm_fits.csv", fit_rows)):
        with open(args.results_dir / name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {args.results_dir / name}")


if __name__ == "__main__":
    main()
