"""Text-query-driven, mask-free spatial localization within a single
image, used by cards.attribution.masking_mode's same-image counterfactual
scoring (cfg.scoring_mode == "masking_hybrid"). No manual masks, no
external segmenter (SAM etc.) -- just the encoder's own per-patch
embeddings (cards.encoders.base.PatchLocalizableEncoder) dotted against
the concept's own text query.

Ported from scripts/celeba/analysis/localize_concept_patches_celeba.py
(notes/celeba_correlation_investigation.md v78 onward), where this was
validated: AUROC=0.855-0.878 against real segmentation masks on CelebA
and CUB, though a negative control there found part of that is
positional-prior confound for centrally-located concepts, not pure
text-specificity -- see that file's own v78/v69 write-ups for the full
caveat.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cards.encoders.base import PatchLocalizableEncoder


def localize_concept(
    encoder: PatchLocalizableEncoder,
    image: Image.Image,
    query: torch.Tensor,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Per-pixel similarity map to `query`, upsampled to `target_hw`.

    `query` is used AS-IS, deliberately NOT renormalized to unit length
    first: cards.concepts.prompts.demean_query intentionally leaves a
    de-meaned query non-unit-norm (retrieval is scale-invariant so it's
    safe there), and every validated masking-hybrid number in notes/
    celeba_correlation_investigation.md v78-v93 was produced against that
    as-is vector. Renormalizing here would quietly change results, not
    just "clean up" the math -- if a caller wants a unit-norm query, it's
    their job to pass one in.
    """
    patches = encoder.encode_patches(image)  # (n_patches, dim)
    n_patches = patches.shape[0]
    grid_side = round(n_patches**0.5)
    if grid_side * grid_side != n_patches:
        raise ValueError(f"encode_patches returned a non-square patch count: {n_patches}")

    sims = (patches @ query.to(patches.device)).cpu().numpy()
    sim_grid = sims.reshape(grid_side, grid_side)

    sim_t = torch.from_numpy(sim_grid).float()[None, None]
    upsampled = F.interpolate(sim_t, size=target_hw, mode="bilinear", align_corners=False)
    return upsampled[0, 0].numpy()


def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Per-image adaptive threshold -- picks the cutoff maximizing
    between-class variance of the histogram, i.e. the split that best
    separates the score distribution into two clusters. NOT the default
    (see threshold_mask): notes/cub_correlation_investigation.md v69/v70
    found this makes Dice WORSE than a fixed top-K% cutoff for this
    specific use case -- these similarity maps aren't bimodal enough
    (closer to a smooth gradient than a tight foreground/background
    split), so Otsu tends to roughly bisect the image rather than isolate
    a small region. Kept available since it's a real, differently-shaped
    alternative, not because it's recommended.
    """
    hist, bin_edges = np.histogram(values, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_centers) / np.where(weight1 == 0, 1, weight1)
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.where(weight2[::-1] == 0, 1, weight2[::-1]))[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = int(np.argmax(variance12))
    return float(bin_centers[idx])


def concept_zscore_cutoff(sim_maps: list[np.ndarray], k: float) -> float:
    """Pooled mean + k*std across multiple similarity maps -- typically
    every present-set image for ONE concept, giving a per-concept-
    calibrated cutoff rather than top_pct's fixed area budget or Otsu's
    bimodal-split assumption (see threshold_mask's own docstring for why
    Otsu underperforms here; this is a third alternative, prompted
    directly: "What if, alternatively we computed a threshold for each
    concept based on some statistics?" -> "I am inclined towards per
    concept statistics"). Unlike a single global absolute cutoff, this
    adapts automatically to each concept's own raw similarity scale --
    load-bearing since demean_query deliberately leaves the query
    non-unit-norm, so different concepts' raw dot-product magnitudes
    aren't directly comparable to begin with.

    Pools ALL pixels from ALL maps into one distribution before taking
    mean/std (not a mean-of-per-image-means), so images with sharper
    peaks contribute proportionally rather than being averaged away.
    """
    pooled = np.concatenate([m.flatten() for m in sim_maps])
    return float(pooled.mean() + k * pooled.std())


def threshold_mask(
    sim_map: np.ndarray, top_pct: float = 15, method: str = "top_pct", cutoff: float | None = None
) -> np.ndarray:
    """Binarizes a similarity map (from localize_concept) into a boolean
    mask, `True` = the localized concept region.

    "top_pct" (default, validated throughout notes/celeba_correlation_
    investigation.md v78-v93 and notes/cub_correlation_investigation.md
    v67-v72 at top_pct=15): a fixed area budget, the top `top_pct`
    percent of pixels by similarity score.
    "otsu": see otsu_threshold's own docstring for why this isn't the
    default.
    "fixed": an explicit, externally-computed `cutoff` (e.g. from
    concept_zscore_cutoff) applied as-is -- this method doesn't compute
    its own cutoff from `sim_map`, since the whole point of a
    per-concept-calibrated threshold is that it's derived from MULTIPLE
    images pooled together, not any single map being thresholded here.
    """
    if method == "top_pct":
        cutoff = np.percentile(sim_map, 100 - top_pct)
    elif method == "otsu":
        cutoff = otsu_threshold(sim_map.flatten())
    elif method == "fixed":
        if cutoff is None:
            raise ValueError("method='fixed' requires an explicit cutoff")
    else:
        raise ValueError(f"unknown threshold method {method!r}")
    return sim_map >= cutoff
