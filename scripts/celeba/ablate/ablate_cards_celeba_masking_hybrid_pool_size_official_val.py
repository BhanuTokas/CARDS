"""Retrieval-dataset-size ablation for the masking hybrid on CelebA
official-val, at the now-confirmed winning config (SigLIP,
orthogonalize=True, demean_query=True, K=50, z-score threshold
alpha=1.0 -- `ablate_cards_celeba_masking_hybrid_joint_orth_demean_
encoder.py`'s own v113 result, rho=+0.6704, the best cell of that
16-cell grid).

Scope, decided directly with the user (first "10, 25, 50 and 100%",
then widened to even 10%-steps -- "Can we do it at intervals of 10%,
i.e., 10%, 20%, 30%, 40%, 50%... and so on?" -- then extended further
down, "Can we also do a 1%, 5% version?"): 12 fractions (1%, 5%, then
10% through 100% in 10%-steps) of the 16,874-image official-val pool,
fixed seed (0), sampled WITHOUT replacement, each a strict subset of
the next larger fraction (build the random permutation ONCE, take
prefixes -- 1% images are a subset of 5%'s, a subset of 10%'s, etc., so
results form a genuine nested growth curve rather than independent
unrelated draws). At 1% (n=169) and 5% (n=844) the pool is still well
above K=50, so naive top-K retrieval is unaffected structurally, but
this is the sparsest end of the curve tested in this track so far.

Built on the REAL cards.pipeline integration (process_concept +
score_masking_hybrid_concepts) and the CORRECTED attribute-conditioned
ground truth, matching v113. The full-pool embeddings are computed/
cached ONCE (via the same official-val cache key every other v98+
official-val script already uses) and then sliced in-memory per
fraction -- no re-embedding per fraction, only retrieval+masking+
scoring differ.
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT, TASK_POSITIVE_LOGIT_INDEX
from run_cards_celeba_masking_hybrid_official_val_zscore import build_clean_official_val_paths

from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS, TARGET_CLASSES
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
SEED = 0
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
FRACTIONS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


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

    cfg = OmegaConf.create({
        "seed": SEED, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
        "retrieval": {"strategy": "naive"}, "k": K,
        "masking_hybrid": {"threshold_method": "zscore", "alpha": ALPHA},
    })
    encoder = instantiate_encoder(cfg)

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()

    official_paths = build_clean_official_val_paths()
    pairs = [(p, 0) for p in official_paths]

    pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": cfg.encoder, "cache_dir": "embedding_cache"})
    pool_cfg.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}
    pool_cfg.pool_source = "val"
    full_pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
    n_full = len(full_pool.paths)
    print(f"full pool: {n_full} images", flush=True)

    # Build queries ONCE (baseline phrasing, orth=True, demean=True is
    # handled inside process_concept via cfg -- not passed a custom
    # `query` here, unlike the joint/prompt ablation scripts, since this
    # axis holds phrasing/orth/demean fixed at their established values
    # and only varies the pool).
    cfg.orthogonalize = True
    cfg.demean_query = True
    raw_queries = {c: build_concept_query(CONCEPT_QUERY_TEXT[c], encoder) for c in GROUNDABLE_CONCEPTS}

    # One nested random permutation -- 10% images subset of 25%'s, etc.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_full)

    all_rows = []  # (fraction, target_task, n_pool, n_pairs, rho, rho_p, sign_frac, n_agree, sign_p)

    for fraction in FRACTIONS:
        n_sub = n_full if fraction == 1.0 else round(n_full * fraction)
        sub_idx = perm[:n_sub]
        pool = replace(full_pool,
                        paths=[full_pool.paths[i] for i in sub_idx],
                        embeddings=full_pool.embeddings[sub_idx],
                        labels=[full_pool.labels[i] for i in sub_idx] if full_pool.labels is not None else None)
        print(f"\n{'=' * 20} fraction={fraction:.0%} (n={n_sub}) {'=' * 20}", flush=True)

        # demean uses the SAME production text-center regardless of pool
        # size (a text-only quantity) -- process_concept computes this
        # internally from cfg, consistent with every other run script.
        text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)
        demeaned = {c: demean_query(q, text_center) for c, q in raw_queries.items()}
        queries = orthogonalize_queries(demeaned)

        results = []
        for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
            result = process_concept(cfg, encoder, pool, concept_name, query=queries[concept_name])
            results.append(result)
            if (concept_idx + 1) % 10 == 0:
                print(f"  retrieval [{concept_idx + 1}/{len(GROUNDABLE_CONCEPTS)}]", flush=True)

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
                  f"sign={sign_r.agreement_frac:.1%} (p={sign_r.binom_p:.4g})", flush=True)
            all_rows.append((fraction, task_name, n_sub, rho_r.n_pairs, rho_r.spearman_rho, rho_r.spearman_p,
                              sign_r.agreement_frac, sign_r.n_agree, sign_r.binom_p))

    out_path = RESULTS_DIR / "cards_celeba_masking_hybrid_pool_size_ablation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fraction", "target_task", "n_pool", "n_pairs", "spearman_rho", "spearman_p",
                          "sign_agreement", "n_agree", "binom_p"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")

    print("\n=== summary, Attractive task ===")
    for r in [r for r in all_rows if r[1] == "Attractive"]:
        print(f"  fraction={r[0]:.0%} n_pool={r[2]:<6d} rho={r[4]:+.4f} (p={r[5]:.4g})  sign={r[6]:.1%} (p={r[8]:.4g})")


if __name__ == "__main__":
    main()
