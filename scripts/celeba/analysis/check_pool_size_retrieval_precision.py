"""Direct check of the skepticism raised about v115/v115b/v115c ("a pool
of 169 having sufficient diversity to represent all concepts seems
unlikely"): does a tiny official-val pool (1%, n=169) actually retrieve
GENUINE concept-positive images for K=50, or does naive top-K retrieval
just return the "least bad" available images regardless of true
prevalence, with the hybrid's rho staying stable for some OTHER reason?

Mirrors `check_retrieval_dilution_celeba.py`'s own precision@K
methodology (rank by cosine similarity to t_c, check against REAL
attribute labels, not a proxy) but compares the SAME K=50 present set
across pool sizes rather than across K -- the same nested 1%/5%/10%/
25%/50%/100% subsamples v115c actually used (same seed=0 permutation),
same `baseline`-phrasing/demean/orthogonalize query construction.

For each concept: precision@50 (fraction of the top-50 retrieved images
whose REAL attribute label is True) at each pool fraction, plus the raw
count of TRUE positives available in that fraction's pool at all (a
concept with a 3% base rate has ~5 true positives in a 169-image pool --
if K=50 must return 50 images regardless, the other ~45 are necessarily
false matches).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT
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
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool

CELEBA_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebA\celeba")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
SEED = 0
FRACTIONS = [0.01, 0.05, 0.10, 0.50, 1.00]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    attr_names = load_attribute_names(CELEBA_ROOT / "list_attr_celeba.txt")
    attr_labels = load_attribute_labels(CELEBA_ROOT / "list_attr_celeba.txt")
    concept_col = {c: attr_names.index(c) for c in GROUNDABLE_CONCEPTS}

    cfg = OmegaConf.create({
        "seed": SEED, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    official_paths = build_clean_official_val_paths()
    pairs = [(p, 0) for p in official_paths]

    pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": cfg.encoder, "cache_dir": "embedding_cache"})
    pool_cfg.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}
    pool_cfg.pool_source = "val"
    full_pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
    n_full = len(full_pool.paths)
    print(f"full pool: {n_full} images", flush=True)

    # real label per image in the full pool, in pool order
    labels = np.zeros((n_full, len(GROUNDABLE_CONCEPTS)), dtype=bool)
    for i, p in enumerate(full_pool.paths):
        row = attr_labels.get(p.name)
        if row is not None:
            labels[i] = [row[concept_col[c]] for c in GROUNDABLE_CONCEPTS]

    raw_queries = {c: build_concept_query(CONCEPT_QUERY_TEXT[c], encoder) for c in GROUNDABLE_CONCEPTS}
    demeaned = {c: demean_query(q, text_center) for c, q in raw_queries.items()}
    queries = orthogonalize_queries(demeaned)

    # SAME nested permutation as ablate_cards_celeba_masking_hybrid_pool_size_official_val.py
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_full)

    all_rows = []  # (fraction, n_pool, concept, base_rate_pct, n_true_in_pool, precision_at_50, n_true_in_top50)

    for fraction in FRACTIONS:
        n_sub = n_full if fraction == 1.0 else round(n_full * fraction)
        sub_idx = perm[:n_sub]
        sub_embeds = full_pool.embeddings[sub_idx]
        sub_labels = labels[sub_idx]
        print(f"\n{'=' * 20} fraction={fraction:.0%} (n={n_sub}) {'=' * 20}", flush=True)

        precisions = []
        for concept_name in GROUNDABLE_CONCEPTS:
            t_c = queries[concept_name]
            sims = (sub_embeds @ t_c.to(sub_embeds.dtype)).numpy()
            top_k = np.argsort(-sims)[:K]
            n_true_in_top50 = int(sub_labels[top_k, GROUNDABLE_CONCEPTS.index(concept_name)].sum())
            precision = n_true_in_top50 / K
            n_true_in_pool = int(sub_labels[:, GROUNDABLE_CONCEPTS.index(concept_name)].sum())
            base_rate_pct = 100.0 * n_true_in_pool / n_sub
            precisions.append(precision)
            all_rows.append((fraction, n_sub, concept_name, base_rate_pct, n_true_in_pool, precision, n_true_in_top50))

        print(f"  mean precision@50 across 26 concepts: {np.mean(precisions):.3f}  "
              f"(min={min(precisions):.3f}, max={max(precisions):.3f})", flush=True)

    out_path = RESULTS_DIR / "celeba_pool_size_retrieval_precision.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fraction", "n_pool", "concept_name", "base_rate_pct", "n_true_in_pool",
                          "precision_at_50", "n_true_in_top50"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")

    print("\n=== rarest concepts at 1% pool: base rate, true-positive count, precision@50 ===")
    frac1 = [r for r in all_rows if r[0] == 0.01]
    for r in sorted(frac1, key=lambda r: r[3])[:10]:
        print(f"  {r[2]:<20s} base_rate={r[3]:5.1f}%  n_true_in_pool={r[4]:>4d}  "
              f"precision@50={r[5]:.3f} ({r[6]}/50 genuinely true)")


if __name__ == "__main__":
    main()
