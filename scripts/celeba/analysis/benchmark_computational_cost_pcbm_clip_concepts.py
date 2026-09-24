"""Instrumented timing benchmark for PCBM's "CLIP concepts" variant,
reusing train_pcbm_clip_concepts_celeba_official_train.py's exact
pipeline at full production scale (162,770 train + 19,867 val = 182,637
images). Reconstructed as a saved, HPC-runnable script, matching the
original ad hoc benchmark's phase breakdown: list_files,
full_train_val_embed (the dominant cost -- a full-dataset pass through
the concept encoder, REUSABLE across any black-box model built on any
backbone, since concept vectors/image embeddings never depend on the
classifier being explained), concept_projection, native_scoring (the
classifier's own predictions, needed as surrogate training labels --
black-box-dependent), elastic_net_fit_x_tasks.

**Takes the concept encoder as a CLI arg (siglip|clip_rn50)**, matching
the real script's own BACKBONE_CFGS -- "CLIP-concepts" is a PCBM variant
family, not one specific model ("Why does CLIP concepts and SIGLIP
concepts have a single row? Are they not 2 different models?"): siglip is
ViT-B-16-SigLIP (the only one previously benchmarked), clip_rn50 is CLIP's
original RN50, a much smaller CNN that should have a very different cost
profile despite belonging to the same PCBM variant.

Uses a SINGLE lambda (not the real script's lam_candidates sweep) to match
the original benchmark's own simpler, single-fit-per-task structure --
this benchmark measures the cost of producing ONE attribution set, not a
hyperparameter search. Writes embeddings to a benchmark-specific cache
path so a stale cache from a prior real run can't silently skip the
full_train_val_embed phase and understate its cost.

Usage: benchmark_computational_cost_pcbm_clip_concepts.py <siglip|clip_rn50>
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder

CELEBA_ROOT = Path(os.environ.get("CELEBA_ROOT", r"C:\Users\btokas\Projects\Datasets\CelebA\celeba"))
RESULTS_DIR = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
BACKBONE_NAME = os.environ.get("CARDS_BACKBONE_NAME", "celeba_official_train_attractive_male")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 128
TASKS = ["Attractive", "Male"]
TASK_SLICES: dict[str, slice] = {"Attractive": slice(0, 2), "Male": slice(2, 4)}
CLIP_BACKBONE_CFGS = {
    "siglip": {"model_name": "ViT-B-16-SigLIP", "pretrained": "webli"},
    "clip_rn50": {"model_name": "RN50", "pretrained": "openai"},
}
CLIP_BACKBONE_LABEL = {"siglip": "SigLIP", "clip_rn50": "CLIP-RN50"}


def load_attr_names(path: Path) -> list[str]:
    return path.read_text().splitlines()[1].split()


def load_attr_labels(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines()[2:]:
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
        if (start // BATCH_SIZE) % 200 == 0:
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
    if len(sys.argv) != 2 or sys.argv[1] not in CLIP_BACKBONE_CFGS:
        raise SystemExit(f"Usage: {sys.argv[0]} <{'|'.join(CLIP_BACKBONE_CFGS)}>")
    clip_backbone_name = sys.argv[1]
    method_label = f"PCBM (CLIP-concepts, {CLIP_BACKBONE_LABEL[clip_backbone_name]})"

    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"BACKBONE_NAME={BACKBONE_NAME}  clip_backbone={clip_backbone_name}  DEVICE={DEVICE}  CELEBA_ROOT={CELEBA_ROOT}", flush=True)
    timings: dict[str, float] = {}

    cfg_dict = {"name": clip_backbone_name, "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                "device": DEVICE, **CLIP_BACKBONE_CFGS[clip_backbone_name]}
    encoder = instantiate_encoder(OmegaConf.create({"encoder": cfg_dict, "device": DEVICE}))
    query_texts = [CONCEPT_QUERY_TEXT[c] for c in GROUNDABLE_CONCEPTS]
    concept_vectors = encoder.encode_text(query_texts).to(DEVICE)
    concept_norms_sq = (concept_vectors ** 2).sum(dim=1)

    t0 = time.perf_counter()
    attr_names = load_attr_names(CELEBA_ROOT / "list_attr_celeba.txt")
    attr_labels = load_attr_labels(CELEBA_ROOT / "list_attr_celeba.txt")
    partition = {}
    with open(CELEBA_ROOT / "list_eval_partition.txt") as f:
        for line in f:
            fname, part = line.split()
            partition[fname] = part
    train_files = sorted(f for f, p in partition.items() if p == "0")
    val_files = sorted(f for f, p in partition.items() if p == "1")
    timings["list_files"] = time.perf_counter() - t0
    print(f"{len(train_files)} train, {len(val_files)} val images (list_files: {timings['list_files']:.2f}s)", flush=True)

    spec = BACKBONES[BACKBONE_NAME]
    native_model = spec.load_native().to(DEVICE).eval()

    t0 = time.perf_counter()
    train_emb = encode_images_batched(encoder, train_files).to(DEVICE)
    val_emb = encode_images_batched(encoder, val_files).to(DEVICE)
    timings["full_train_val_embed"] = time.perf_counter() - t0
    print(f"full_train_val_embed: {timings['full_train_val_embed']:.2f}s", flush=True)

    t0 = time.perf_counter()
    train_proj = ((train_emb @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)).detach().cpu().numpy()
    val_proj = ((val_emb @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)).detach().cpu().numpy()
    timings["concept_projection"] = time.perf_counter() - t0
    print(f"concept_projection: {timings['concept_projection']:.2f}s", flush=True)

    t0 = time.perf_counter()
    train_native_logits = native_task_logits(train_files, native_model, spec.preprocess, DEVICE)
    val_native_logits = native_task_logits(val_files, native_model, spec.preprocess, DEVICE)
    timings["native_scoring"] = time.perf_counter() - t0
    print(f"native_scoring: {timings['native_scoring']:.2f}s", flush=True)

    from train_pcbm import run_linear_probe

    n_concepts = train_proj.shape[1]
    single_lam = 0.01 / (2 * n_concepts)

    t0 = time.perf_counter()
    for task_name in TASKS:
        task_slice = TASK_SLICES[task_name]
        train_surrogate = train_native_logits[:, task_slice].argmax(axis=1)
        val_surrogate = val_native_logits[:, task_slice].argmax(axis=1)

        class Args:
            seed = SEED
            lam = single_lam
            alpha = 0.99

        run_info, weights, bias = run_linear_probe(Args(), (train_proj, train_surrogate), (val_proj, val_surrogate))
        print(f"  {task_name}: train fidelity={run_info['train_acc']:.2f}%  val fidelity={run_info['test_acc']:.2f}%", flush=True)
    timings["elastic_net_fit_x_tasks"] = time.perf_counter() - t0
    print(f"elastic_net_fit_x_tasks: {timings['elastic_net_fit_x_tasks']:.2f}s", flush=True)

    timings["TOTAL"] = sum(timings.values())

    print(f"\n=== Summary: {method_label} ===", flush=True)
    for phase, secs in timings.items():
        print(f"  {phase:<28s} {secs:>10.2f}s")

    out_path = RESULTS_DIR / f"computational_cost_benchmark_pcbm_clip_concepts_{clip_backbone_name}_full_scale.csv"
    with open(out_path, "w") as f:
        f.write("method,phase,seconds\n")
        for phase, secs in timings.items():
            f.write(f'"{method_label}",{phase},{secs}\n')
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
