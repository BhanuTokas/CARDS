"""Patch-count (L) ablation for COCO-Stuff CARDS; port of
scripts/celeba/ablate/ablate_cards_celeba_masking_hybrid_patch_count_official_train.py.

L is the encoder's patch grid, (image_size / patch_size)^2, so it is varied two ways. Only the first isolates L:
  - L=49, 100: the same ViT-B-16-SigLIP weights and retrieval as the headline run, with the 14x14 patch embeddings
    average-pooled to a coarser grid before localization. Only mask granularity changes.
  - L=256, 576, 1024: the published B/16 SigLIP checkpoints at 256, 384, 512 px. Different weights, so retrieval, queries
    and fill selection change along with L.
L=196 is recomputed and must reproduce the headline cell (alpha=1.5, K=50) from alpha_ablation_scores.csv before anything
else runs.

Fixed as in ablate_k.py: SigLIP, alpha=1.5 (z-score), K=50, demean=True, orth=True, best-of-7 fill, ResNet18 black box.
env L_VALUES=49,100 runs a subset (196 always runs first); SKIP_BASELINE_CHECK=1 disables the stop-on-mismatch.
Threshold: the z-score threshold ablation on COCO is alpha_ablation_*.csv (alpha 0.25-2.5); no separate script.

Outputs (results/coco_stuff/): patch_count_ablation_scores.csv (L, concept_name, class_name, raw_score) and
patch_count_ablation_summary.csv (L, model_name, strategy, class_name, ..., pooled_rho, pooled_p, pooled_n)
Usage: cd /scratch/rnair21/CARDS && python scripts/coco_stuff/ablations/ablate_patch_count.py"""

from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from ablate_k import (  # noqa: E402
    ALPHA, ALPHA_CSV, BRODEN_DROPPED_CONCEPTS, BRODEN_ROOT, CKPT_PATH, COCO_ROOT, DEVICE, FAITH_CSV, FILL_STRATEGIES,
    K_BASELINE, RESULTS_DIR, SEED, STRATEGIES, evaluate, load_faithfulness, load_k50_scores,
)

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.coco_stuff import COCO_STUFF_20_CLASSES
from cards.data.datasets import list_broden_concepts, load_coco_stuff
from cards.models.coco_stuff_black_box import _coco_stuff_preprocess
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import mask_region

BASELINE_L = 196
TOL = 1e-3
SCORES_CSV = RESULTS_DIR / "patch_count_ablation_scores.csv"
SUMMARY_CSV = RESULTS_DIR / "patch_count_ablation_summary.csv"


@dataclass(frozen=True)
class Variant:
    L: int
    model_name: str
    pooled_grid: int | None
    batch_size: int | None  # None keeps the 224 px cache key


VARIANTS = {
    196: Variant(196, "ViT-B-16-SigLIP", None, None),
    49: Variant(49, "ViT-B-16-SigLIP", 7, None),
    100: Variant(100, "ViT-B-16-SigLIP", 10, None),
    256: Variant(256, "ViT-B-16-SigLIP-256", None, 192),
    576: Variant(576, "ViT-B-16-SigLIP-384", None, 96),
    1024: Variant(1024, "ViT-B-16-SigLIP-512", None, 48),
}


class CoarsePatchEncoder:
    """Delegates to a real encoder, average-pooling its patch grid. Image/text embeddings untouched, so retrieval and
    fill selection match the native-grid run exactly."""

    def __init__(self, encoder, grid_side: int):
        self._encoder, self._grid_side = encoder, grid_side

    def __getattr__(self, name):
        return getattr(self._encoder, name)

    def encode_patches(self, image):
        p = self._encoder.encode_patches(image)
        side = round(p.shape[0] ** 0.5)
        grid = F.adaptive_avg_pool2d(p.T.reshape(1, p.shape[1], side, side), self._grid_side)
        return F.normalize(grid[0].flatten(1).T, dim=-1)


def build(variant: Variant):
    enc_cfg = {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
               "model_name": variant.model_name, "pretrained": "webli", "device": DEVICE}
    if variant.batch_size is not None:
        enc_cfg["batch_size"] = variant.batch_size
    cfg = OmegaConf.create({"seed": SEED, "device": DEVICE, "encoder": enc_cfg, "cache_dir": "embedding_cache"})
    encoder = instantiate_encoder(cfg)
    cfg.dataset = {"name": "coco_stuff", "root": str(COCO_ROOT)}
    cfg.pool_source = "val"
    pairs = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="val")
    t0 = time.perf_counter()
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)  # cached on the real encoder
    print(f"pool: {len(pool.paths)} images ({time.perf_counter() - t0:.0f}s)", flush=True)
    if variant.pooled_grid is not None:
        encoder = CoarsePatchEncoder(encoder, variant.pooled_grid)
    n = encoder.encode_patches(Image.new("RGB", (64, 64))).shape[0]
    if n != variant.L:
        raise RuntimeError(f"{variant.model_name} produced L={n}, expected {variant.L}")
    return encoder, pool


def run_variant(variant: Variant, raw_model, preprocess, concepts) -> dict[tuple[str, str], float]:
    print(f"\n{'=' * 20} L={variant.L} ({variant.model_name}, pooled_grid={variant.pooled_grid}) {'=' * 20}", flush=True)
    encoder, pool = build(variant)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)
    queries = orthogonalize_queries({c: demean_query(build_concept_query(c, encoder), text_center) for c in concepts})
    cidx = {c: i for i, c in enumerate(concepts)}
    scores: dict[tuple[str, str], float] = {}
    for c_i, concept in enumerate(concepts):
        t_c = queries[concept]; t_dev = t_c.to(DEVICE)
        present, _ = retrieve_top_bottom_k(pool, t_c, K_BASELINE)
        cached = []
        for idx in present:
            image = Image.open(pool.paths[idx]).convert("RGB")
            cached.append((idx, image, localize_concept(encoder, image, t_c, (image.height, image.width))))
        cutoff = concept_zscore_cutoff([sm for _, _, sm in cached], ALPHA)
        deltas = {cls: [] for cls in COCO_STUFF_20_CLASSES}
        for idx, image, sim_map in cached:
            mask = threshold_mask(sim_map, method="fixed", cutoff=cutoff)
            if not mask.any() or mask.all():
                continue
            rng = np.random.default_rng(SEED + cidx[concept] * 10_000 + int(idx))
            cands = [mask_region(image, mask, strategy=s, rng=rng) for s in FILL_STRATEGIES]
            with torch.no_grad():
                emb = encoder.encode_images([image] + cands).to(DEVICE)
            best_i, best_angle = 0, None
            for i in range(len(FILL_STRATEGIES)):
                d = emb[0] - emb[1 + i]; d = d / (d.norm() + 1e-8)
                ang = float(np.degrees(np.arccos(float(torch.clamp(d @ t_dev, -1.0, 1.0)))))
                if best_angle is None or ang < best_angle:
                    best_angle, best_i = ang, i
            px = torch.cat([preprocess(image).unsqueeze(0), preprocess(cands[best_i]).unsqueeze(0)]).to(DEVICE)
            with torch.no_grad():
                lg = raw_model(px)
            for k, cls in enumerate(COCO_STUFF_20_CLASSES):
                deltas[cls].append((lg[0, k] - lg[1, k]).item())
        for cls in COCO_STUFF_20_CLASSES:
            scores[(concept, cls)] = float(np.mean(deltas[cls])) if deltas[cls] else 0.0
        if (c_i + 1) % 10 == 0:
            print(f"  [{c_i + 1}/{len(concepts)}] {concept}", flush=True)
    return scores


def main() -> None:
    import torch.nn as nn
    from torchvision.models import resnet18

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    requested = [int(v) for v in os.environ.get("L_VALUES", ",".join(map(str, VARIANTS))).split(",")]
    order = [BASELINE_L] + [L for L in requested if L != BASELINE_L]
    concepts = [c for c in list_broden_concepts(BRODEN_ROOT) if c not in BRODEN_DROPPED_CONCEPTS]

    raw_model = resnet18(weights=None)
    raw_model.fc = nn.Linear(raw_model.fc.in_features, len(COCO_STUFF_20_CLASSES))
    raw_model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    raw_model = raw_model.to(DEVICE).eval()
    preprocess = _coco_stuff_preprocess()

    faiths = {s: load_faithfulness(FAITH_CSV, s) for s in STRATEGIES}
    reference = load_k50_scores(ALPHA_CSV, alpha=ALPHA)
    all_scores: dict[int, dict] = {}
    summary: list[dict] = []
    for L in order:
        all_scores[L] = run_variant(VARIANTS[L], raw_model, preprocess, concepts)
        for strategy in STRATEGIES:
            faith, counts = faiths[strategy]
            rows, mean_rho, mean_sign, pr, pp, pn = evaluate(all_scores[L], faith, counts)
            print(f"  L={L:<5d} {strategy:<11s} mean_rho={mean_rho:+.4f} mean_sign={mean_sign:.4f} pooled_rho={pr:+.4f} n={pn}",
                  flush=True)
            summary = [r for r in summary if not (r["L"] == L and r["strategy"] == strategy)]
            summary += [{"L": L, "model_name": VARIANTS[L].model_name, "strategy": strategy, **r,
                         "pooled_rho": pr, "pooled_p": pp, "pooled_n": pn} for r in rows]
            if L == BASELINE_L:
                ref_pr = evaluate(reference, faith, counts)[3]
                print(f"  baseline check ({strategy}): pooled {pr:.4f}, alpha_ablation reference {ref_pr:.4f}", flush=True)
                if abs(pr - ref_pr) > TOL and not os.environ.get("SKIP_BASELINE_CHECK"):
                    raise RuntimeError("L=196 did not reproduce the headline cell; stopping before the sweep")
        with open(SCORES_CSV, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["L", "concept_name", "class_name", "raw_score"])
            for l_, sc in all_scores.items():
                for (c, cls), v in sc.items():
                    w.writerow([l_, c, cls, v])
        fields = ["L", "model_name", "strategy", "class_name", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement",
                  "sign_p", "pooled_rho", "pooled_p", "pooled_n"]
        with open(SUMMARY_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summary)
    print(f"Saved -> {SCORES_CSV}, {SUMMARY_CSV}", flush=True)


if __name__ == "__main__":
    main()
