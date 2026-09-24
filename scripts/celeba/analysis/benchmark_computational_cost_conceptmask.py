"""Instrumented timing benchmark for ConceptMask, reusing run_cards_celeba_
masking_hybrid_official_train.py's exact pipeline (SigLIP, orthogonalize=
True, demean_query=True, K=50, z-score alpha=1.0, official-val retrieval
pool) up through scoring, at full production scale. Reconstructed as a
saved, HPC-runnable script ("I also wanted to run the same comparison on
HPC as well" -- the original 3-method benchmark that produced results/
computational_cost_benchmark_full_scale.csv was only ever run ad hoc,
never saved).

Phases match the original ad hoc benchmark's own breakdown: encoder_load,
load_images_from_disk (building the official-val retrieval pool's file
list), pool_embed (SigLIP-encoding every pool image once, reusable across
concepts/tasks), query_build (text-query encode + demean + orthogonalize
for all 26 concepts), scoring_all_concepts_x_tasks (retrieval +
localization + masking + attribution, all 26 concepts x 2 tasks).

Does NOT score against ground truth (that's a separate, cheap, and
already-covered analysis step elsewhere) -- this benchmark only measures
the cost of PRODUCING attributions, matching what the other three methods'
benchmarks measure.

**Uses the RAW official-val pool, not the "clean" HQ-overlap-excluded one**
("Why does it need CelebAMask-HQ?" -- run_cards_celeba_masking_hybrid_
official_train.py's own build_clean_official_val_paths() excludes ~2,993
images that also appear in CelebAMask-HQ's train/val splits, to avoid
retrieval-pool/ground-truth leakage -- but that exclusion requires
CelebAMask-HQ's own annotation files, which aren't available on Sol at
all, making the reconstructed benchmark non-portable there. For a pure
wall-clock cost benchmark this distinction doesn't matter: it only
shrinks the pool by ~18% out of ~16,874 images, and cost is driven by
pool SIZE, not which specific images are in it. Using the raw pool here
removes the CelebAMask-HQ dependency entirely.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.models.backbones import BACKBONES
from cards.pipeline import (
    instantiate_encoder,
    orthogonalize_queries,
    process_concept,
    score_masking_hybrid_concepts,
)
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool

CELEBA_ROOT = Path(os.environ.get("CELEBA_ROOT", r"C:\Users\btokas\Projects\Datasets\CelebA\celeba"))
RESULTS_DIR = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
CACHE_DIR = Path(os.environ.get("CARDS_COST_BENCH_CACHE_DIR", "cost_bench_cache_conceptmask"))
BACKBONE_NAME = os.environ.get("CARDS_BACKBONE_NAME", "celeba_official_train_attractive_male")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
ALPHA = 1.0
SEED = 0
TASKS = ["Attractive", "Male"]
TASK_SLICE = {"Attractive": 1, "Male": 3}


class TaskBlackBox:
    """Wraps __call__ with an accumulating timer so scoring_all_concepts_x_tasks
    can be split into encoder-side (retrieval, localization, perturbation
    selection -- reusable across black-box models) vs black-box-dependent
    (time actually spent inside the black box's own forward pass) without
    touching masking_score/masking_mode internals ("What would be the
    reusable vs. black box specific split here?").
    """

    total_call_seconds = 0.0

    def __init__(self, native_model, task_name: str, preprocess, device: str):
        self.model = native_model
        self.task_idx = TASK_SLICE[task_name]
        self._preprocess = preprocess
        self.device = device

    def preprocess(self, image):
        return self._preprocess(image.convert("RGB"))

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        t0 = time.perf_counter()
        out = self.model(batch.to(self.device))[:, self.task_idx].detach().cpu()
        TaskBlackBox.total_call_seconds += time.perf_counter() - t0
        return out


def build_raw_official_val_paths() -> list[Path]:
    paths = []
    with open(CELEBA_ROOT / "list_eval_partition.txt") as f:
        for line in f:
            fname, part = line.split()
            if part == "1":
                paths.append(CELEBA_ROOT / "img_align_celeba" / fname)
    return paths


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"BACKBONE_NAME={BACKBONE_NAME}  DEVICE={DEVICE}  CELEBA_ROOT={CELEBA_ROOT}", flush=True)
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    cfg = OmegaConf.create({
        "seed": SEED, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": str(CACHE_DIR),
        "retrieval": {"strategy": "naive"}, "k": K,
        "masking_hybrid": {"threshold_method": "zscore", "alpha": ALPHA},
    })
    encoder = instantiate_encoder(cfg)
    spec = BACKBONES[BACKBONE_NAME]
    native_model = spec.load_native().to(DEVICE).eval()
    timings["encoder_load"] = time.perf_counter() - t0
    print(f"encoder_load: {timings['encoder_load']:.2f}s", flush=True)

    t0 = time.perf_counter()
    official_paths = build_raw_official_val_paths()
    pairs = [(p, 0) for p in official_paths]
    timings["load_images_from_disk"] = time.perf_counter() - t0
    print(f"{len(pairs)} pool images (load_images_from_disk: {timings['load_images_from_disk']:.2f}s)", flush=True)

    t0 = time.perf_counter()
    pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": cfg.encoder, "cache_dir": str(CACHE_DIR)})
    pool_cfg.dataset = {"name": "celeba_official_val_raw", "root": str(CELEBA_ROOT)}
    pool_cfg.pool_source = "val"
    pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
    timings["pool_embed"] = time.perf_counter() - t0
    print(f"pool: {len(pool.paths)} images (pool_embed: {timings['pool_embed']:.2f}s)", flush=True)

    t0 = time.perf_counter()
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)
    raw_queries = {c: build_concept_query(CONCEPT_QUERY_TEXT[c], encoder) for c in GROUNDABLE_CONCEPTS}
    demeaned = {c: demean_query(q, text_center) for c, q in raw_queries.items()}
    queries = orthogonalize_queries(demeaned)
    timings["query_build"] = time.perf_counter() - t0
    print(f"query_build: {timings['query_build']:.2f}s", flush=True)

    t0 = time.perf_counter()
    results = []
    for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
        result = process_concept(cfg, encoder, pool, concept_name, query=queries[concept_name])
        results.append(result)
        if (concept_idx + 1) % 10 == 0:
            print(f"  retrieval [{concept_idx + 1}/{len(GROUNDABLE_CONCEPTS)}]", flush=True)

    for task_name in TASKS:
        black_box = TaskBlackBox(native_model, task_name, spec.preprocess, DEVICE)
        hybrid_results = score_masking_hybrid_concepts(cfg, encoder, black_box, pool, GROUNDABLE_CONCEPTS, results)
        print(f"  scored {len(hybrid_results)} concepts for task={task_name}", flush=True)
    timings["scoring_all_concepts_x_tasks"] = time.perf_counter() - t0
    print(f"scoring_all_concepts_x_tasks: {timings['scoring_all_concepts_x_tasks']:.2f}s", flush=True)

    timings["TOTAL"] = sum(timings.values())

    black_box_seconds = TaskBlackBox.total_call_seconds
    reusable_seconds = timings["TOTAL"] - black_box_seconds

    print("\n=== Summary: ConceptMask ===", flush=True)
    for phase, secs in timings.items():
        print(f"  {phase:<32s} {secs:>10.2f}s")
    print(f"\n  black_box_calls (within scoring_all_concepts_x_tasks) {black_box_seconds:>10.2f}s", flush=True)
    print(f"  reusable (encoder-side)                               {reusable_seconds:>10.2f}s", flush=True)
    print(f"  black-box-dependent                                   {black_box_seconds:>10.2f}s", flush=True)

    out_path = RESULTS_DIR / "computational_cost_benchmark_conceptmask_full_scale.csv"
    with open(out_path, "w") as f:
        f.write("method,phase,seconds\n")
        for phase, secs in timings.items():
            f.write(f"ConceptMask,{phase},{secs}\n")
        f.write(f"ConceptMask,black_box_calls,{black_box_seconds}\n")
        f.write(f"ConceptMask,reusable_encoder_side,{reusable_seconds}\n")
        f.write(f"ConceptMask,black_box_dependent,{black_box_seconds}\n")
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
