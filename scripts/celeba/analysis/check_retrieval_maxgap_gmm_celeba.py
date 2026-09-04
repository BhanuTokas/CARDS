"""Two alternative "identify a positive cluster" methods, tested against
the same 6 concepts/pool as `check_retrieval_bimodality_celeba.py`, after
that script confirmed Otsu fails because the similarity distributions are
genuinely unimodal (prompted directly, "if not Otsu's method, is there
any way to identify a positive cluster?"):

1. Max-gap on the SORTED ranking, not the histogram: restricted to ranks
   [MIN_RANK, MAX_RANK], find the single largest drop between consecutive
   similarity values. Unlike Otsu (which looks at the WHOLE distribution's
   shape), this only looks locally near the top of the ranking, so it can
   in principle find a real boundary even inside a globally smooth/
   unimodal distribution -- a histogram bins away exactly this kind of
   local structure.
2. 1-component vs. 2-component GMM (BIC comparison): a formal statistical
   test of "is there real evidence of 2 clusters" per concept, rather
   than eyeballing a histogram. If 2 components genuinely fit better,
   also reports the two components' means/stds/weights and the decision
   boundary between them (intersection of the two weighted Gaussians) as
   a candidate alternative cutoff.

Both are evaluated the same way as the Otsu check: does `l` (or the
GMM's implied set size) come out smaller than K for concepts with fewer
true positives, and does the resulting set's precision (against REAL
CelebA attribute labels) actually improve on flat top-K?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.celeba_attributes import load_attribute_labels, load_attribute_names
from cards.data.datasets import load_celeba
from cards.pipeline import instantiate_encoder
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
MIN_RANK, MAX_RANK = 5, 300  # search window for the max-gap method

CONCEPTS_TO_CHECK = [
    ("Eyeglasses", "clean"),
    ("Smiling", "clean"),
    ("Wavy_Hair", "clean"),
    ("Big_Nose", "severe dilution"),
    ("Narrow_Eyes", "broken at every K"),
    ("Pale_Skin", "broken at every K"),
]


def max_gap_l(sims_sorted_desc: np.ndarray) -> int:
    """Largest consecutive drop within [MIN_RANK, MAX_RANK]; returns the
    rank (1-indexed count) at which that drop occurs, i.e. l = the number
    of images on the high side of the biggest local jump."""
    window = sims_sorted_desc[MIN_RANK : MAX_RANK + 1]
    gaps = window[:-1] - window[1:]
    best = int(np.argmax(gaps))
    return MIN_RANK + best + 1


def gmm_1_vs_2(sims: np.ndarray) -> tuple[float, float, GaussianMixture | None]:
    """Returns (bic_1, bic_2, gmm_2_or_None). Lower BIC is better; gmm_2
    is returned only so its component stats can be printed."""
    x = sims.reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, random_state=42).fit(x)
    gmm2 = GaussianMixture(n_components=2, random_state=42).fit(x)
    return gmm1.bic(x), gmm2.bic(x), gmm2


def gmm_decision_boundary(gmm2: GaussianMixture, sims: np.ndarray) -> float | None:
    """Scans the observed similarity range for the point where the two
    weighted Gaussian densities cross, on whichever side has the higher
    mean (the 'positive' component). None if components aren't ordered
    the way expected or don't cross in-range."""
    means = gmm2.means_.ravel()
    order = np.argsort(means)  # low, high
    lo, hi = order
    xs = np.linspace(sims.min(), sims.max(), 2000)

    def weighted_density(idx, x):
        mean, var, weight = gmm2.means_[idx, 0], gmm2.covariances_[idx].ravel()[0], gmm2.weights_[idx]
        return weight * np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)

    d_lo = weighted_density(lo, xs)
    d_hi = weighted_density(hi, xs)
    sign = np.sign(d_hi - d_lo)
    crossings = np.where(np.diff(sign) != 0)[0]
    if len(crossings) == 0:
        return None
    return float(xs[crossings[-1]])  # rightmost crossing = boundary nearest the high component


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
    print(f"pool: {len(pool.paths)} images\n", flush=True)

    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    pool_labels = np.array([attr_labels_by_file[f"{Path(p).stem}.jpg"] for p in pool.paths])

    for concept_name, tag in CONCEPTS_TO_CHECK:
        t_c = build_concept_query(CONCEPT_QUERY_TEXT[concept_name], encoder)
        t_c = demean_query(t_c, text_center)
        sims = (pool.embeddings @ t_c).numpy()

        attr_idx = attr_names.index(concept_name)
        true_positive_mask = pool_labels[:, attr_idx].astype(bool)
        n_true_total = int(true_positive_mask.sum())

        order = np.argsort(-sims)
        sims_sorted = sims[order]
        precision_topk = float(true_positive_mask[order[:K]].mean())

        print(f"=== {concept_name}  ({tag}, n_true={n_true_total}/{len(pool.paths)}) ===")

        # --- Method 1: max-gap on the sorted ranking ---
        l_gap = max_gap_l(sims_sorted)
        n_select_gap = min(l_gap, K)
        precision_gap = float(true_positive_mask[order[:n_select_gap]].mean())
        print(f"  [max-gap] l={l_gap}  min(l,K)={n_select_gap}  precision={precision_gap:.3f}  "
              f"(vs. flat top-K precision={precision_topk:.3f})")

        # --- Method 2: GMM(1) vs GMM(2) BIC ---
        bic1, bic2, gmm2 = gmm_1_vs_2(sims)
        prefers_2 = bic2 < bic1
        print(f"  [GMM] BIC(1 comp)={bic1:.1f}  BIC(2 comp)={bic2:.1f}  "
              f"{'2-component preferred' if prefers_2 else '1-component preferred (no real 2nd cluster)'}")
        if prefers_2:
            means = gmm2.means_.ravel()
            stds = np.sqrt(gmm2.covariances_.ravel())
            weights = gmm2.weights_.ravel()
            order_comp = np.argsort(means)
            print(f"    components (low->high): mean={means[order_comp]}  std={stds[order_comp]}  "
                  f"weight={weights[order_comp]}")
            boundary = gmm_decision_boundary(gmm2, sims)
            if boundary is not None:
                l_gmm = int((sims > boundary).sum())
                n_select_gmm = min(l_gmm, K)
                precision_gmm = float(true_positive_mask[order[:n_select_gmm]].mean())
                print(f"    decision boundary={boundary:+.4f}  l={l_gmm}  min(l,K)={n_select_gmm}  "
                      f"precision={precision_gmm:.3f}")
            else:
                print("    components don't cross in-range -- no usable boundary")
        print()


if __name__ == "__main__":
    main()
