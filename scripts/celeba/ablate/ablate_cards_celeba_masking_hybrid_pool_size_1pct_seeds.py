"""Seed-stability check for the 1%-pool cell of v115c/v115d's pool-size
ablation, prompted directly ("Can we do the 1% experiment at different
seeds to see if the value is stable?") -- v115c found rho=+0.6390 at a
single 1%-of-pool (n=169) draw (seed=0), but v115d's precision@50 check
showed retrieval precision collapses badly for rare concepts at that
pool size (e.g. `Bald` has only 2 true positives in the WHOLE 169-image
pool). A single seed can't distinguish "rho is genuinely robust at this
pool size" from "this particular seed happened to draw a lucky 169
images" -- this script draws several INDEPENDENT random 169-image
subsamples (not nested prefixes of one permutation, unlike v115c) and
reports rho/sign per seed to see the actual spread.

Same winning config as v113-v115 (SigLIP, orthogonalize=True,
demean_query=True, K=50, z-score alpha=1.0), `baseline` phrasing,
official-val pool, corrected attribute-conditioned ground truth. Also
reports mean precision@50 (against REAL attribute labels, same
methodology as `check_pool_size_retrieval_precision.py`) per seed
alongside rho, to see whether low-precision draws correspond to
low-rho draws or the two are decoupled.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
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
    TARGET_CLASSES,
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
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

CELEBA_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebA\celeba")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
ALPHA = 1.0
FRACTION = 0.01
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


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


def load_records_by_task() -> dict[str, list[FaithfulnessResult]]:
    by_task: dict[str, list[FaithfulnessResult]] = {t: [] for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "celeba_full_faithfulness_attribute_conditioned.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]].append(FaithfulnessResult(
                image=row["image"], concept_number=CONCEPT_TO_IDX[row["concept_name"]], category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return by_task


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    records_by_task = load_records_by_task()

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

    all_rows = []  # (seed, target_task, n_pairs, rho, rho_p, sign_frac, n_agree, sign_p, mean_precision_at_50)

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        sub_idx = rng.choice(n_full, size=n_sub, replace=False)
        pool = replace(full_pool,
                        paths=[full_pool.paths[i] for i in sub_idx],
                        embeddings=full_pool.embeddings[sub_idx],
                        labels=[full_pool.labels[i] for i in sub_idx] if full_pool.labels is not None else None)
        sub_labels = labels[sub_idx]
        print(f"\n{'=' * 20} seed={seed} (n={n_sub}) {'=' * 20}", flush=True)

        precisions = []
        results = []
        for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
            t_c = queries[concept_name]
            sims = (pool.embeddings @ t_c.to(pool.embeddings.dtype)).numpy()
            top_k = np.argsort(-sims)[:K]
            n_true = int(sub_labels[top_k, concept_idx].sum())
            precisions.append(n_true / K)

            result = process_concept(cfg, encoder, pool, concept_name, query=t_c)
            results.append(result)
            if (concept_idx + 1) % 10 == 0:
                print(f"  retrieval [{concept_idx + 1}/{len(GROUNDABLE_CONCEPTS)}]", flush=True)

        mean_precision = float(np.mean(precisions))

        for task_name in TARGET_CLASSES:
            black_box = TaskBlackBox(native_model, task_name, spec.preprocess, DEVICE)
            hybrid_results = score_masking_hybrid_concepts(cfg, encoder, black_box, pool, GROUNDABLE_CONCEPTS, results)
            scores = {(CONCEPT_TO_IDX[c], 1): r.raw_score for c, r in hybrid_results.items()}

            rho_r = score_method_agreement(records_by_task[task_name], scores)
            sign_r = score_sign_agreement(records_by_task[task_name], scores)
            if rho_r is None:
                print(f"  [{task_name}] too few pairs", flush=True)
                continue
            print(f"  [{task_name}] n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} (p={rho_r.spearman_p:.4g})  "
                  f"sign={sign_r.agreement_frac:.1%} (p={sign_r.binom_p:.4g})  mean_precision@50={mean_precision:.3f}",
                  flush=True)
            all_rows.append((seed, task_name, rho_r.n_pairs, rho_r.spearman_rho, rho_r.spearman_p,
                              sign_r.agreement_frac, sign_r.n_agree, sign_r.binom_p, mean_precision))

    out_path = RESULTS_DIR / "cards_celeba_masking_hybrid_pool_size_1pct_seed_stability.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "target_task", "n_pairs", "spearman_rho", "spearman_p",
                          "sign_agreement", "n_agree", "binom_p", "mean_precision_at_50"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")

    print("\n=== summary, Attractive task, across seeds ===")
    attractive = [r for r in all_rows if r[1] == "Attractive"]
    for r in attractive:
        print(f"  seed={r[0]} rho={r[3]:+.4f} (p={r[4]:.4g})  sign={r[5]:.1%} (p={r[7]:.4g})  precision@50={r[8]:.3f}")
    rhos = [r[3] for r in attractive]
    print(f"\n  rho across {len(rhos)} seeds: mean={np.mean(rhos):+.4f} std={np.std(rhos):.4f} "
          f"min={min(rhos):+.4f} max={max(rhos):+.4f}")


if __name__ == "__main__":
    main()
