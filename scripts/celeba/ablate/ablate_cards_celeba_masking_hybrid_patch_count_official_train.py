"""Patch-count (L) ablation for the masking hybrid on the official-train
CelebA classifier (ResNet18). ICLR 2027 Reviewer 1 asked which L was used
and how much it matters, comparing it to K.

L is not a free parameter of the pipeline: it is the encoder's patch grid
(image_size / patch_size)^2, so it is varied two different ways. Only the
first isolates L; keep the two apart when reporting.

- L=49, 100: same ViT-B-16-SigLIP weights and the same retrieval as the
  headline run, with the 14x14 patch embeddings average-pooled to a coarser
  grid before localization. Only mask granularity changes.
- L=256, 576, 1024: the published B/16 SigLIP checkpoints at 256, 384 and
  512 px. These are different weights, so retrieval, queries and fill
  selection all change along with L.

L=196 is recomputed rather than reused and must reproduce the joint encoder
ablation's (siglip, orth=True, demean=True) cell before anything else runs.

Fixed at the settings behind the paper's tables: orthogonalize=True,
demean_query=True, z-score alpha=1.0 (not the 1.5 the paper's text states),
K=50, ensembled prompts.

Set L_VALUES (comma-separated) to run a subset. L=196 always runs first.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from ablate_cards_celeba_masking_hybrid_k_demean_true_official_train import (
    ALPHA,
    BACKBONE_NAME,
    CACHE_DIR,
    CELEBA_ROOT,
    CONCEPT_TO_IDX,
    DEVICE,
    GT_VERSIONS,
    K_REUSED,
    RESULTS_DIR,
    SEED,
    TASKS,
    TaskBlackBox,
    load_records,
    load_reused_k50_scores,
)
from run_cards_celeba_full import CONCEPT_QUERY_TEXT
from run_cards_celeba_masking_hybrid_official_val_zscore import build_clean_official_val_paths

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
from cards.validation.broden_faithfulness import score_method_agreement, score_sign_agreement

BASELINE_L = 196
BASELINE_RHO_TOLERANCE = 1e-3
BASELINE_GT = "previously_used"
CSV_PATH = RESULTS_DIR / "cards_celeba_masking_hybrid_patch_count_official_train_ablation.csv"
RAW_CSV_PATH = RESULTS_DIR / "cards_celeba_masking_hybrid_patch_count_official_train_ablation_raw_scores.csv"


@dataclass(frozen=True)
class Variant:
    L: int
    model_name: str
    pooled_grid: int | None  # coarser grid side, or None for the encoder's native grid
    batch_size: int | None  # None keeps the cache key of the existing 224 px pool


VARIANTS = {
    196: Variant(196, "ViT-B-16-SigLIP", None, None),
    49: Variant(49, "ViT-B-16-SigLIP", 7, None),
    100: Variant(100, "ViT-B-16-SigLIP", 10, None),
    256: Variant(256, "ViT-B-16-SigLIP-256", None, 192),
    576: Variant(576, "ViT-B-16-SigLIP-384", None, 96),
    1024: Variant(1024, "ViT-B-16-SigLIP-512", None, 48),
}


class CoarsePatchEncoder:
    """Delegates to a real encoder, average-pooling its patch grid. Image and
    text embeddings are untouched, so retrieval and fill selection match the
    native-grid run exactly."""

    def __init__(self, encoder, grid_side: int):
        self._encoder = encoder
        self._grid_side = grid_side

    def __getattr__(self, name):
        return getattr(self._encoder, name)

    def encode_patches(self, image: Image.Image) -> torch.Tensor:
        patches = self._encoder.encode_patches(image)
        side = round(patches.shape[0] ** 0.5)
        grid = patches.T.reshape(1, patches.shape[1], side, side)
        pooled = F.adaptive_avg_pool2d(grid, self._grid_side)
        return F.normalize(pooled[0].flatten(1).T, dim=-1)


def build_encoder(variant: Variant):
    encoder_cfg = {
        "name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
        "model_name": variant.model_name, "pretrained": "webli", "device": DEVICE,
    }
    if variant.batch_size is not None:
        encoder_cfg["batch_size"] = variant.batch_size
    cfg = OmegaConf.create({
        "seed": SEED, "device": DEVICE, "encoder": encoder_cfg, "cache_dir": str(CACHE_DIR),
        "retrieval": {"strategy": "naive"}, "k": K_REUSED,
        "masking_hybrid": {"threshold_method": "zscore", "alpha": ALPHA},
    })
    encoder = instantiate_encoder(cfg)
    if variant.pooled_grid is not None:
        encoder = CoarsePatchEncoder(encoder, variant.pooled_grid)

    n_patches = encoder.encode_patches(Image.new("RGB", (64, 64))).shape[0]
    if n_patches != variant.L:
        raise RuntimeError(f"{variant.model_name} produced L={n_patches}, expected {variant.L}")
    return cfg, encoder


def run_variant(variant: Variant, native_model, spec, pairs) -> tuple[list[tuple], list[tuple]]:
    print(f"\n{'=' * 20} L={variant.L} ({variant.model_name}, pooled_grid={variant.pooled_grid}) {'=' * 20}", flush=True)
    cfg, encoder = build_encoder(variant)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)
    raw_queries = {
        c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], encoder), text_center)
        for c in GROUNDABLE_CONCEPTS
    }
    queries = orthogonalize_queries(raw_queries)

    pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": cfg.encoder, "cache_dir": str(CACHE_DIR)})
    pool_cfg.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}
    pool_cfg.pool_source = "val"
    start = time.perf_counter()
    pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
    pool_seconds = time.perf_counter() - start
    print(f"pool: {len(pool.paths)} images ({pool_seconds:.0f}s)", flush=True)

    results = []
    for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
        results.append(process_concept(cfg, encoder, pool, concept_name, query=queries[concept_name]))
        if (concept_idx + 1) % 10 == 0:
            print(f"  retrieval [{concept_idx + 1}/{len(GROUNDABLE_CONCEPTS)}]", flush=True)

    rows, raw_rows = [], []
    for task_name in TASKS:
        start = time.perf_counter()
        print(f"  scoring {task_name}", flush=True)
        black_box = TaskBlackBox(native_model, task_name, spec.preprocess, DEVICE)
        hybrid = score_masking_hybrid_concepts(cfg, encoder, black_box, pool, GROUNDABLE_CONCEPTS, results)
        scoring_seconds = time.perf_counter() - start
        scores = {(CONCEPT_TO_IDX[c], 1): r.raw_score for c, r in hybrid.items()}

        for c, r in hybrid.items():
            deltas = np.asarray(r.delta_scores)
            delta_std = float(deltas.std(ddof=1)) if len(deltas) > 1 else float("nan")
            fills = Counter(r.selected_strategies).most_common()
            raw_rows.append((variant.L, variant.model_name, variant.pooled_grid, task_name, c, r.raw_score, delta_std,
                             len(deltas), r.n_skipped_degenerate, ";".join(f"{s}={n}" for s, n in fills)))

        for gt_name, gt_filename in GT_VERSIONS.items():
            records = load_records(gt_filename, task_name)
            rho = score_method_agreement(records, scores)
            sign = score_sign_agreement(records, scores)
            print(f"  [{task_name}/{gt_name}] n={rho.n_pairs} rho={rho.spearman_rho:+.4f} "
                  f"sign={sign.agreement_frac:.1%} scoring={scoring_seconds:.0f}s", flush=True)
            rows.append((variant.L, variant.model_name, variant.pooled_grid, task_name, gt_name, rho.n_pairs,
                         rho.spearman_rho, rho.spearman_p, sign.agreement_frac, pool_seconds, scoring_seconds))
    return rows, raw_rows


def pooled_rho(rows: list[tuple], L: int, gt_name: str) -> float:
    return float(np.mean([r[6] for r in rows if r[0] == L and r[4] == gt_name]))


def reused_baseline_rho() -> float:
    reused = load_reused_k50_scores()
    rhos = []
    for task_name in TASKS:
        records = load_records(GT_VERSIONS[BASELINE_GT], task_name)
        rhos.append(score_method_agreement(records, reused[task_name]).spearman_rho)
    return float(np.mean(rhos))


def write_csvs(rows: list[tuple], raw_rows: list[tuple]) -> None:
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["L", "model_name", "pooled_grid", "target_task", "gt_version", "n_pairs", "spearman_rho",
                         "spearman_p", "sign_agreement", "pool_seconds", "scoring_seconds"])
        writer.writerows(rows)
    with open(RAW_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["L", "model_name", "pooled_grid", "target_task", "concept_name", "hybrid_raw_score",
                         "delta_std", "n_scored", "n_skipped_degenerate", "selected_fills"])
        writer.writerows(raw_rows)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    requested = [int(v) for v in os.environ.get("L_VALUES", ",".join(map(str, VARIANTS))).split(",")]
    order = [BASELINE_L] + [L for L in requested if L != BASELINE_L]

    spec = BACKBONES[BACKBONE_NAME]
    native_model = spec.load_native().to(DEVICE).eval()
    pairs = [(p, 0) for p in build_clean_official_val_paths()]

    rows, raw_rows = [], []
    for L in order:
        new_rows, new_raw = run_variant(VARIANTS[L], native_model, spec, pairs)
        rows += new_rows
        raw_rows += new_raw
        write_csvs(rows, raw_rows)

        if L == BASELINE_L:
            expected, got = reused_baseline_rho(), pooled_rho(rows, L, BASELINE_GT)
            print(f"\nbaseline check: pooled rho {got:.4f}, expected {expected:.4f}", flush=True)
            if abs(got - expected) > BASELINE_RHO_TOLERANCE:
                raise RuntimeError("L=196 did not reproduce the reused baseline; stopping before the sweep")

    print("\n=== pooled rho (Attractive + Male, naive mean) ===")
    for L in sorted({r[0] for r in rows}):
        cost = next(r for r in rows if r[0] == L)
        print(f"  L={L:<5d} {cost[1]:<22s} previously_used={pooled_rho(rows, L, 'previously_used'):.3f} "
              f"non_overlapping={pooled_rho(rows, L, 'non_overlapping'):.3f}")


if __name__ == "__main__":
    main()
