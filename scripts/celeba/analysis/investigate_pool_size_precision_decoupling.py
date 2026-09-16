"""Investigates the open puzzle from v115d/v115e: at a 169-image (1%)
official-val pool, mean retrieval precision@50 is only ~0.35 (collapses
badly for rare concepts specifically), yet the aggregate 26-concept
rank correlation (rho) against faithfulness ground truth stays stable
at rho~0.64-0.74 across 8 independent seeds. Does that mean the
low-precision concepts are just ALONG FOR THE RIDE (their own rank gets
scrambled, but not enough of them to move the aggregate rho much), or
are they somehow still landing in roughly the right rank DESPITE poor
retrieval precision?

Two complementary within-run diagnostics, computed per (seed, concept)
across the SAME 8 seeds as `ablate_cards_celeba_masking_hybrid_pool_
size_1pct_seeds.py` (independent random 169-image draws, not nested):
  1. Rank error: |rank(raw_score) - rank(GT delta_p)| among that seed's
     26 concepts -- does higher rank error co-occur with lower
     precision@50, across all 8x26=208 (seed, concept) observations?
  2. Leave-one-out delta-rho: for each seed, recompute the 25-concept
     rho with one concept removed at a time; delta_rho = rho_without -
     rho_full. A concept whose removal INCREASES rho was dragging the
     correlation down; a concept whose removal DECREASES rho was
     helping it. Averaged per concept across the 8 seeds, then
     correlated against that concept's own mean precision@50.

Ground truth (delta_p per concept, Attractive task) is POOL-INDEPENDENT
(it's the real classifier's masking-based ground truth over its own
held-out sample, unrelated to the retrieval pool) -- computed ONCE, not
per seed.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT, TASK_POSITIVE_LOGIT_INDEX
from run_cards_celeba_masking_hybrid_official_val_zscore import build_clean_official_val_paths

from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba_attributes import (
    GROUNDABLE_CONCEPTS,
    load_attribute_labels,
    load_attribute_names,
)
from cards.models.backbones import BACKBONES
from cards.pipeline import (
    instantiate_encoder,
    orthogonalize_queries,
    process_concept,
    score_masking_hybrid_concepts,
)
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool

CELEBA_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebA\celeba")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
ALPHA = 1.0
FRACTION = 0.01
TASK = "Attractive"
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


def load_gt_delta_p() -> dict[str, float]:
    """mean delta_p per concept for (task=Attractive, predicted_class=1),
    the SAME aggregation `score_method_agreement` uses internally --
    pool-independent, computed once."""
    grouped: dict[int, list[float]] = {i: [] for i in range(len(GROUNDABLE_CONCEPTS))}
    with open(RESULTS_DIR / "celeba_full_faithfulness_attribute_conditioned.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["target_task"] != TASK or int(row["predicted_class"]) != 1:
                continue
            grouped[CONCEPT_TO_IDX[row["concept_name"]]].append(float(row["delta_p"]))
    return {GROUNDABLE_CONCEPTS[i]: float(np.mean(v)) for i, v in grouped.items() if v}


class TaskBlackBox:
    def __init__(self, native_model, task_name: str, preprocess, device: str):
        self.model = native_model
        self.task_idx = TASK_POSITIVE_LOGIT_INDEX[task_name]
        self._preprocess = preprocess
        self.device = device

    def preprocess(self, image):
        return self._preprocess(image.convert("RGB"))

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch.to(self.device))[:, self.task_idx].detach().cpu()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    gt_delta_p = load_gt_delta_p()
    concepts_with_gt = [c for c in GROUNDABLE_CONCEPTS if c in gt_delta_p]
    print(f"{len(concepts_with_gt)}/{len(GROUNDABLE_CONCEPTS)} concepts have GT for {TASK}", flush=True)

    attr_names = load_attribute_names(CELEBA_ROOT / "list_attr_celeba.txt")
    attr_labels = load_attribute_labels(CELEBA_ROOT / "list_attr_celeba.txt")
    concept_col = {c: attr_names.index(c) for c in GROUNDABLE_CONCEPTS}

    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
        "retrieval": {"strategy": "naive"}, "k": K,
        "masking_hybrid": {"threshold_method": "zscore", "alpha": ALPHA},
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()
    black_box = TaskBlackBox(native_model, TASK, spec.preprocess, DEVICE)

    official_paths = build_clean_official_val_paths()
    pairs = [(p, 0) for p in official_paths]

    pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": cfg.encoder, "cache_dir": "embedding_cache"})
    pool_cfg.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}
    pool_cfg.pool_source = "val"
    full_pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
    n_full = len(full_pool.paths)
    n_sub = round(n_full * FRACTION)
    print(f"full pool: {n_full} images -- 1% subsample size: {n_sub}", flush=True)

    labels = np.zeros((n_full, len(GROUNDABLE_CONCEPTS)), dtype=bool)
    for i, p in enumerate(full_pool.paths):
        row = attr_labels.get(p.name)
        if row is not None:
            labels[i] = [row[concept_col[c]] for c in GROUNDABLE_CONCEPTS]

    raw_queries = {c: build_concept_query(CONCEPT_QUERY_TEXT[c], encoder) for c in GROUNDABLE_CONCEPTS}
    demeaned = {c: demean_query(q, text_center) for c, q in raw_queries.items()}
    queries = orthogonalize_queries(demeaned)

    per_concept_rows = []  # (seed, concept, precision_at_50, raw_score, rank_pred, rank_gt, rank_error)
    loo_rows = []  # (seed, concept, rho_full, rho_without, delta_rho)

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        sub_idx = rng.choice(n_full, size=n_sub, replace=False)
        pool = replace(full_pool,
                        paths=[full_pool.paths[i] for i in sub_idx],
                        embeddings=full_pool.embeddings[sub_idx],
                        labels=[full_pool.labels[i] for i in sub_idx] if full_pool.labels is not None else None)
        sub_labels = labels[sub_idx]
        print(f"\n{'=' * 20} seed={seed} (n={n_sub}) {'=' * 20}", flush=True)

        results = []
        precision_by_concept = {}
        for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
            t_c = queries[concept_name]
            sims = (pool.embeddings @ t_c.to(pool.embeddings.dtype)).numpy()
            top_k = np.argsort(-sims)[:K]
            precision_by_concept[concept_name] = int(sub_labels[top_k, concept_idx].sum()) / K

            result = process_concept(cfg, encoder, pool, concept_name, query=t_c)
            results.append(result)
            if (concept_idx + 1) % 10 == 0:
                print(f"  retrieval [{concept_idx + 1}/{len(GROUNDABLE_CONCEPTS)}]", flush=True)

        hybrid_results = score_masking_hybrid_concepts(cfg, encoder, black_box, pool, GROUNDABLE_CONCEPTS, results)
        raw_score_by_concept = {c: r.raw_score for c, r in hybrid_results.items()}

        gt_vals = [gt_delta_p[c] for c in concepts_with_gt]
        pred_vals = [raw_score_by_concept[c] for c in concepts_with_gt]
        rank_gt = {c: r for c, r in zip(concepts_with_gt, np.argsort(np.argsort(gt_vals)))}
        rank_pred = {c: r for c, r in zip(concepts_with_gt, np.argsort(np.argsort(pred_vals)))}

        rho_full, p_full = spearmanr(gt_vals, pred_vals)
        print(f"  full rho={rho_full:+.4f} (p={p_full:.4g})", flush=True)

        for c in concepts_with_gt:
            rank_error = abs(rank_pred[c] - rank_gt[c])
            per_concept_rows.append((seed, c, precision_by_concept[c], raw_score_by_concept[c],
                                      int(rank_pred[c]), int(rank_gt[c]), rank_error))

        for held_out in concepts_with_gt:
            remaining = [c for c in concepts_with_gt if c != held_out]
            g = [gt_delta_p[c] for c in remaining]
            p = [raw_score_by_concept[c] for c in remaining]
            rho_without, _ = spearmanr(g, p)
            loo_rows.append((seed, held_out, rho_full, rho_without, rho_without - rho_full))

    with open(RESULTS_DIR / "celeba_pool_size_1pct_per_concept_precision_vs_rank.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "concept_name", "precision_at_50", "raw_score", "rank_pred", "rank_gt", "rank_error"])
        writer.writerows(per_concept_rows)

    with open(RESULTS_DIR / "celeba_pool_size_1pct_leave_one_out.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "held_out_concept", "rho_full", "rho_without", "delta_rho"])
        writer.writerows(loo_rows)

    print(f"\nSaved {len(per_concept_rows)} per-concept rows and {len(loo_rows)} leave-one-out rows", flush=True)

    # --- Diagnostic 1: precision vs rank error, pooled across all (seed, concept) ---
    precisions_all = [r[2] for r in per_concept_rows]
    rank_errors_all = [r[6] for r in per_concept_rows]
    rho1, p1 = spearmanr(precisions_all, rank_errors_all)
    print(f"\n=== Diagnostic 1: precision@50 vs rank_error, pooled n={len(precisions_all)} "
          f"(8 seeds x {len(concepts_with_gt)} concepts) ===")
    print(f"  Spearman rho(precision, rank_error) = {rho1:+.4f} (p={p1:.4g})")
    print("  (negative rho = higher precision associated with LOWER rank error, as expected if precision matters)")

    # --- Diagnostic 2: mean precision vs mean leave-one-out delta_rho, per concept ---
    mean_precision_by_concept = {
        c: float(np.mean([r[2] for r in per_concept_rows if r[1] == c])) for c in concepts_with_gt
    }
    mean_delta_rho_by_concept = {
        c: float(np.mean([r[4] for r in loo_rows if r[1] == c])) for c in concepts_with_gt
    }
    prec_vec = [mean_precision_by_concept[c] for c in concepts_with_gt]
    delta_rho_vec = [mean_delta_rho_by_concept[c] for c in concepts_with_gt]
    rho2, p2 = spearmanr(prec_vec, delta_rho_vec)
    print(f"\n=== Diagnostic 2: mean precision@50 vs mean leave-one-out delta_rho, per concept, n={len(concepts_with_gt)} ===")
    print(f"  Spearman rho(precision, delta_rho) = {rho2:+.4f} (p={p2:.4g})")
    print("  (positive rho = LOW-precision concepts tend to have delta_rho>0, i.e. removing them HELPS rho --")
    print("   confirms they're dragging it down; near-zero/negative = they're not the ones responsible)")

    print("\n  concepts sorted by mean precision@50 (ascending), with mean delta_rho:")
    for c in sorted(concepts_with_gt, key=lambda c: mean_precision_by_concept[c]):
        print(f"    {c:<20s} mean_precision@50={mean_precision_by_concept[c]:.3f}  "
              f"mean_delta_rho={mean_delta_rho_by_concept[c]:+.4f}")


if __name__ == "__main__":
    main()
