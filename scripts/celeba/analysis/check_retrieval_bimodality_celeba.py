"""Checks the bimodal-similarity-distribution assumption behind a proposed
retrieval-set-size fix, prompted directly ("if we assume bimodal
distribution, we can identify a positive and negative set... we take
min(l,K) images in the positive set"). Before building an adaptive-K
retrieval function, this checks whether pool-wide cosine-similarity
distributions (t_c vs. every pool image) actually LOOK bimodal for real
CelebA concepts -- the same "assume bimodal" idea already failed once in
this codebase at the PIXEL-similarity level (CUB v69/v70, Otsu
underperformed a fixed top-pct cutoff there) using the exact same
`otsu_threshold` function reused here at the retrieval level instead.

Six concepts, chosen directly from v90's own dilution-check findings
(`check_retrieval_dilution_celeba.py`) to span the full range: 3 "clean"
concepts (precision@K flat at ~1.00 for every K -- Eyeglasses, Smiling,
Wavy_Hair), 1 "severe dilution" concept (starts high, degrades badly by
K=50 -- Big_Nose), and 2 "broken at every K" concepts (precision 0.00-0.30
even at K=5 -- Narrow_Eyes, Pale_Skin) -- the critical case, since a
retrieval FAILURE (bad embedding-space separation) should look different
from genuine DILUTION (too few true positives) if the bimodal-split idea
is going to distinguish them.

For each concept: Otsu threshold on the full pool's similarity array
(`otsu_threshold`, already generic/reusable, not pixel-specific despite
living in cards.attribution.localization); how many images land above it
(l); what fraction of those l images are REAL true positives per CelebA's
own attribute label (precision of the Otsu-selected set); and a coarse
20-bin text histogram of the similarity distribution for a visual
bimodality check without needing to open a plot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.attribution.localization import otsu_threshold
from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.celeba_attributes import load_attribute_labels, load_attribute_names
from cards.data.datasets import load_celeba
from cards.pipeline import instantiate_encoder
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50

CONCEPTS_TO_CHECK = [
    ("Eyeglasses", "clean (v90: precision~1.00 at every K)"),
    ("Smiling", "clean (v90: precision~1.00 at every K)"),
    ("Wavy_Hair", "clean (v90: precision~1.00 at every K)"),
    ("Big_Nose", "severe dilution (v90: 0.60 at K=5 -> 0.32 at K=50)"),
    ("Narrow_Eyes", "broken at every K (v90: 0.00-0.30 throughout)"),
    ("Pale_Skin", "broken at every K (v90: 0.00-0.30 throughout)"),
]


def text_histogram(values: np.ndarray, n_bins: int = 20, width: int = 50) -> str:
    counts, edges = np.histogram(values, bins=n_bins)
    max_count = counts.max()
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * max(0, round(c / max_count * width)) if max_count else ""
        lines.append(f"  [{edges[i]:+.3f}, {edges[i+1]:+.3f}) {c:>5d} {bar}")
    return "\n".join(lines)


def main():
    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    cfg.dataset = {"name": "celeba", "root": str(CELEBA_HQ_ROOT)}
    cfg.pool_source = "val"
    pairs = load_celeba(CELEBA_HQ_ROOT, split="val")
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images (same as v90's dilution check)\n", flush=True)

    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    pool_labels = np.array([attr_labels_by_file[f"{Path(p).stem}.jpg"] for p in pool.paths])  # (N, 40)

    for concept_name, tag in CONCEPTS_TO_CHECK:
        t_c = build_concept_query(CONCEPT_QUERY_TEXT[concept_name], encoder)
        t_c = demean_query(t_c, text_center)
        sims = (pool.embeddings @ t_c).numpy()

        attr_idx = attr_names.index(concept_name)
        true_positive_mask = pool_labels[:, attr_idx].astype(bool)
        n_true_total = int(true_positive_mask.sum())

        cutoff = otsu_threshold(sims)
        above = sims > cutoff
        l = int(above.sum())
        precision_otsu = float(true_positive_mask[above].mean()) if l else float("nan")

        # top-min(l,K) by similarity (the proposed fix), vs. plain top-K (today's behavior)
        order = np.argsort(-sims)
        n_select = min(l, K) if l else 0
        precision_adaptive = float(true_positive_mask[order[:n_select]].mean()) if n_select else float("nan")
        precision_topk = float(true_positive_mask[order[:K]].mean())

        print(f"=== {concept_name}  ({tag}) ===")
        print(f"  n_true_in_pool={n_true_total}/{len(pool.paths)}  sim range=[{sims.min():+.3f}, {sims.max():+.3f}] "
              f"mean={sims.mean():+.3f} std={sims.std():.3f}")
        print(f"  Otsu cutoff={cutoff:+.3f}  l (images above cutoff)={l}  "
              f"precision of Otsu-selected set={precision_otsu:.3f}")
        print(f"  proposed fix: min(l,K)={n_select} selected, precision={precision_adaptive:.3f}   "
              f"vs. today's flat top-K={K}, precision={precision_topk:.3f}")
        print(text_histogram(sims))
        print()


if __name__ == "__main__":
    main()
