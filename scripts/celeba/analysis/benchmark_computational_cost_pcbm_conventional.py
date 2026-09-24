"""Instrumented timing benchmark for the "conventional" PCBM variant (CAVs
fit directly in the black-box model's OWN ResNet-18 activation space, no
separate concept encoder), added as a fourth row to the computational cost
comparison ("Shouldn't it be present set size?" ... "should we not have a
normal resnet-18 based baseline as well? I mean for the PCBM encoder").

The existing results/computational_cost_benchmark_full_scale.csv only
covers "PCBM (CLIP-concepts, SigLIP)", which needs a full-dataset pass
through a heavy, separately-loaded SigLIP encoder (1185.7s of its 1648.5s
total). This conventional variant reuses the SAME ResNet-18 backbone the
black-box classifier itself uses (mirrors fit_celeba_official_train_cavs.py
+ train_pcbm_surrogate_celeba_official_train.py, at full scale: 26
GROUNDABLE_CONCEPTS, 150 pos/150 neg whole-image candidates per concept for
the CAV fit, then the full official-train (162,770) + val (19,867) =
182,637-image embed pass shared across both tasks, matching the other
methods' full-scale benchmark).

Unlike the SigLIP variant's encoder cost (shared across any black-box model
built on the same backbone family, so mostly REUSABLE), this variant's
"full_train_val_embed" and "native_scoring" phases are computed through the
SPECIFIC black-box model's own backbone -- if you swap in a different
black-box model, both need to be redone from scratch, same as TCAV. So
where the SigLIP variant's cost is mostly reusable-across-black-box-models,
this variant's cost is mostly black-box-dependent, much like TCAV -- a
real, worth-reporting structural difference between the two PCBM
configurations, not just a speed difference.

Runs identically locally and on Sol HPC (CELEBA_ROOT is the only
environment-dependent path). Requires post_hoc_cbm as a sibling directory
(../post_hoc_cbm relative to CARDS root) and the `pcbm-training` extra
(pandas, imported transitively by train_pcbm.py's own data/__init__.py).
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from torch.utils.data import DataLoader

from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.models.backbones import BACKBONES

CELEBA_ROOT = Path(os.environ.get("CELEBA_ROOT", r"C:\Users\btokas\Projects\Datasets\CelebA\celeba"))
BACKBONE_NAME = os.environ.get("CARDS_BACKBONE_NAME", "celeba_official_train_attractive_male")
RESULTS_DIR = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 50
MAX_PER_CLASS = 150
C_VALUE = 0.1  # matches CONCEPT_BANK_PATH's own "_0.1_100.pkl" choice in the surrogate-training script
BATCH_SIZE_CAV = 25
BATCH_SIZE_EMBED = 64
SEED = 42
TASKS = ["Attractive", "Male"]
TASK_SLICES: dict[str, slice] = {"Attractive": slice(0, 2), "Male": slice(2, 4)}


class FlattenedFeatureExtractor(nn.Module):
    def __init__(self, feature_extractor: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, x):
        return torch.flatten(self.feature_extractor(x), 1)


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


def embed_files(filenames, feature_extractor, preprocess, device, label, batch_size=BATCH_SIZE_EMBED):
    from PIL import Image
    embeddings = []
    for start in range(0, len(filenames), batch_size):
        batch_files = filenames[start : start + batch_size]
        batch = torch.stack([
            preprocess(Image.open(CELEBA_ROOT / "img_align_celeba" / f).convert("RGB")) for f in batch_files
        ]).to(device)
        with torch.no_grad():
            emb = torch.flatten(feature_extractor(batch), 1).cpu().numpy()
        embeddings.append(emb)
        if (start // batch_size) % 500 == 0:
            print(f"  [{label}] {start + len(batch_files)}/{len(filenames)}", flush=True)
    return np.concatenate(embeddings, axis=0)


def native_task_logits(filenames, native_model, preprocess, device, batch_size=BATCH_SIZE_EMBED):
    from PIL import Image
    logits = []
    for start in range(0, len(filenames), batch_size):
        batch_files = filenames[start : start + batch_size]
        batch = torch.stack([
            preprocess(Image.open(CELEBA_ROOT / "img_align_celeba" / f).convert("RGB")) for f in batch_files
        ]).to(device)
        with torch.no_grad():
            logits.append(native_model(batch).cpu().numpy())
    return np.concatenate(logits, axis=0)


def main():
    torch.manual_seed(SEED)
    rng_py = random.Random(SEED)
    np.random.seed(SEED)

    from concepts import ConceptBank
    from concepts.concept_utils import ListDataset, learn_concept_bank
    from models import PosthocLinearCBM
    from train_pcbm import run_linear_probe

    print(f"BACKBONE_NAME={BACKBONE_NAME}  DEVICE={DEVICE}  CELEBA_ROOT={CELEBA_ROOT}", flush=True)
    spec = BACKBONES[BACKBONE_NAME]
    native_model = spec.load_native().to(DEVICE).eval()
    backbone = FlattenedFeatureExtractor(spec.feature_extractor(native_model)).to(DEVICE).eval()
    preprocess = spec.preprocess

    timings: dict[str, float] = {}

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
    print(f"{len(train_files)} train, {len(val_files)} val official-train images "
          f"(list_files: {timings['list_files']:.2f}s)", flush=True)

    print("\n=== concept bank: feature extraction + CAV fit (all 26 concepts) ===", flush=True)
    t0 = time.perf_counter()
    concept_lib = {}
    for concept_name in GROUNDABLE_CONCEPTS:
        idx = attr_names.index(concept_name)
        pos_all = [CELEBA_ROOT / "img_align_celeba" / f for f in train_files if attr_labels[f][idx]]
        neg_all = [CELEBA_ROOT / "img_align_celeba" / f for f in train_files if not attr_labels[f][idx]]
        rng_py.shuffle(pos_all)
        rng_py.shuffle(neg_all)
        pos_paths, neg_paths = pos_all[:MAX_PER_CLASS], neg_all[:MAX_PER_CLASS]

        pos_loader = DataLoader(ListDataset(pos_paths, preprocess), batch_size=BATCH_SIZE_CAV, shuffle=False)
        neg_loader = DataLoader(ListDataset(neg_paths, preprocess), batch_size=BATCH_SIZE_CAV, shuffle=False)
        cav_info = learn_concept_bank(pos_loader, neg_loader, backbone, N_SAMPLES, [C_VALUE], device=DEVICE)
        concept_lib[concept_name] = cav_info[C_VALUE]
        print(f"  {concept_name}: train_acc={cav_info[C_VALUE][1]:.3f} test_acc={cav_info[C_VALUE][2]:.3f}", flush=True)
    timings["concept_bank_extraction_and_fit"] = time.perf_counter() - t0
    print(f"concept_bank_extraction_and_fit: {timings['concept_bank_extraction_and_fit']:.2f}s", flush=True)

    concept_bank = ConceptBank(concept_lib, DEVICE)

    print("\n=== full official train+val embed (shared across both tasks) ===", flush=True)
    t0 = time.perf_counter()
    train_emb = embed_files(train_files, backbone, preprocess, DEVICE, "train")
    val_emb = embed_files(val_files, backbone, preprocess, DEVICE, "val")
    timings["full_train_val_embed"] = time.perf_counter() - t0
    print(f"full_train_val_embed: {timings['full_train_val_embed']:.2f}s", flush=True)

    print("\n=== concept-margin projection ===", flush=True)
    t0 = time.perf_counter()
    probe_layer = PosthocLinearCBM(concept_bank, backbone_name=BACKBONE_NAME, n_classes=2).to(DEVICE)
    train_proj = probe_layer.compute_dist(torch.tensor(train_emb, device=DEVICE).float()).detach().cpu().numpy()
    val_proj = probe_layer.compute_dist(torch.tensor(val_emb, device=DEVICE).float()).detach().cpu().numpy()
    timings["concept_projection"] = time.perf_counter() - t0
    print(f"concept_projection: {timings['concept_projection']:.2f}s", flush=True)

    print("\n=== native model task logits (train+val, shared across both tasks) ===", flush=True)
    t0 = time.perf_counter()
    train_native_logits = native_task_logits(train_files, native_model, preprocess, DEVICE)
    val_native_logits = native_task_logits(val_files, native_model, preprocess, DEVICE)
    timings["native_scoring"] = time.perf_counter() - t0
    print(f"native_scoring: {timings['native_scoring']:.2f}s", flush=True)

    print("\n=== elastic net fit x tasks ===", flush=True)
    t0 = time.perf_counter()
    for task_name in TASKS:
        task_slice = TASK_SLICES[task_name]
        train_surrogate = train_native_logits[:, task_slice].argmax(axis=1)
        val_surrogate = val_native_logits[:, task_slice].argmax(axis=1)

        class Args:
            seed = SEED
            lam = 0.0002
            alpha = 0.99

        run_info, weights, bias = run_linear_probe(Args(), (train_proj, train_surrogate), (val_proj, val_surrogate))
        print(f"  {task_name}: train fidelity={run_info['train_acc']:.2f}%  val fidelity={run_info['test_acc']:.2f}%", flush=True)
    timings["elastic_net_fit_x_tasks"] = time.perf_counter() - t0
    print(f"elastic_net_fit_x_tasks: {timings['elastic_net_fit_x_tasks']:.2f}s", flush=True)

    timings["TOTAL"] = sum(timings.values())

    print("\n=== Summary: PCBM (conventional, ResNet-18) ===", flush=True)
    for phase, secs in timings.items():
        print(f"  {phase:<32s} {secs:>10.2f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "computational_cost_benchmark_pcbm_conventional_full_scale.csv"
    with open(out_path, "w") as f:
        f.write("method,phase,seconds\n")
        for phase, secs in timings.items():
            f.write(f'"PCBM (conventional, ResNet-18)",{phase},{secs}\n')
    print(f"\nSaved to {out_path}", flush=True)
    print("Append this to results/computational_cost_benchmark_full_scale.csv as the fourth method.", flush=True)


if __name__ == "__main__":
    main()
