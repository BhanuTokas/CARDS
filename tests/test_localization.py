"""Tests for cards.attribution.localization."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from cards.attribution.localization import (
    concept_zscore_cutoff,
    localize_concept,
    otsu_threshold,
    threshold_mask,
)


class _FakePatchEncoder:
    """encode_patches returns a fixed 3x3 grid (9 patches) where patch
    index `hot_idx` has a much higher dot product against any query
    aligned with `hot_direction` than the rest -- lets tests assert
    localize_concept ranks the right spatial location highest without
    needing a real model."""

    def __init__(self, n_patches: int = 9, dim: int = 4, hot_idx: int = 4):
        self.n_patches = n_patches
        self.dim = dim
        self.hot_idx = hot_idx

    def encode_patches(self, image: Image.Image) -> torch.Tensor:
        # Every patch has SOME dim-0 (query-aligned) signal mixed with a
        # dim-1 (orthogonal) component, so post-normalize cosine
        # similarity to a dim-0-aligned query still varies across
        # patches -- unlike an all-zero-except-dim-0 design, where
        # normalization would trivially give every patch cos_sim=1.
        patches = torch.zeros(self.n_patches, self.dim)
        patches[:, 0] = 0.1
        patches[:, 1] = 1.0
        patches[self.hot_idx, 0] = 1.0
        patches[self.hot_idx, 1] = 0.1
        return torch.nn.functional.normalize(patches, dim=-1)


def test_localize_concept_ranks_hot_patch_highest():
    encoder = _FakePatchEncoder(n_patches=9, dim=4, hot_idx=4)  # center of a 3x3 grid
    query = torch.tensor([1.0, 0.0, 0.0, 0.0])
    image = Image.new("RGB", (30, 30))

    sim_map = localize_concept(encoder, image, query, target_hw=(30, 30))

    assert sim_map.shape == (30, 30)
    # patch 4 is the center of the 3x3 grid -> rows/cols 10:20 in a 30x30 upsample
    center_region = sim_map[10:20, 10:20]
    corner_region = sim_map[0:10, 0:10]
    assert center_region.mean() > corner_region.mean()


def test_localize_concept_rejects_non_square_patch_count():
    encoder = _FakePatchEncoder(n_patches=5, dim=4, hot_idx=0)
    query = torch.tensor([1.0, 0.0, 0.0, 0.0])
    image = Image.new("RGB", (10, 10))

    with pytest.raises(ValueError):
        localize_concept(encoder, image, query, target_hw=(10, 10))


def test_localize_concept_does_not_renormalize_query():
    # A non-unit-norm query should scale the similarity map proportionally
    # (i.e. it's used as-is in the dot product), not get silently
    # renormalized to unit length first.
    encoder = _FakePatchEncoder(n_patches=9, dim=4, hot_idx=0)
    image = Image.new("RGB", (9, 9))

    unit_query = torch.tensor([1.0, 0.0, 0.0, 0.0])
    scaled_query = unit_query * 5.0

    unit_map = localize_concept(encoder, image, unit_query, target_hw=(9, 9))
    scaled_map = localize_concept(encoder, image, scaled_query, target_hw=(9, 9))

    assert np.allclose(scaled_map, unit_map * 5.0, atol=1e-4)


# ---- threshold_mask / otsu_threshold ----


def test_threshold_mask_top_pct_selects_expected_fraction():
    sim_map = np.arange(100).reshape(10, 10).astype(float)

    mask = threshold_mask(sim_map, top_pct=10, method="top_pct")

    assert mask.sum() == 10  # top 10% of 100 pixels


def test_threshold_mask_rejects_unknown_method():
    sim_map = np.zeros((4, 4))
    with pytest.raises(ValueError):
        threshold_mask(sim_map, method="bogus")


def test_otsu_threshold_separates_bimodal_distribution():
    rng = np.random.default_rng(0)
    low = rng.normal(0, 1, 1000)
    high = rng.normal(10, 1, 200)
    data = np.concatenate([low, high])

    t = otsu_threshold(data)

    assert 3.0 < t < 7.0  # clearly between the two clusters
    frac_above = (data > t).mean()
    assert 0.10 < frac_above < 0.25  # roughly the high-cluster's own share (200/1200)


# ---- concept_zscore_cutoff / threshold_mask(method="fixed") ----


def test_concept_zscore_cutoff_matches_hand_computed_pooled_stats():
    maps = [np.array([[0.0, 2.0], [4.0, 6.0]]), np.array([[8.0, 10.0], [12.0, 14.0]])]
    pooled = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])

    cutoff = concept_zscore_cutoff(maps, alpha=1.0)

    assert cutoff == pytest.approx(pooled.mean() + pooled.std())


def test_concept_zscore_cutoff_pools_pixels_not_per_image_means():
    # One map with a sharp peak, one flat -- a mean-of-per-image-means
    # would treat both images equally regardless of pixel count/spread;
    # pooling should weight the peaked map's larger internal std more.
    peaked = np.array([0.0, 0.0, 0.0, 100.0])
    flat = np.array([5.0, 5.0, 5.0, 5.0])

    pooled_cutoff = concept_zscore_cutoff([peaked, flat], alpha=1.0)
    combined = np.concatenate([peaked, flat])

    assert pooled_cutoff == pytest.approx(combined.mean() + combined.std())
    # Sanity: this must differ from naively averaging each map's own
    # mean+std (25.0+43.3 vs. 5.0+0.0, averaged -> a different number).
    naive = np.mean([peaked.mean() + peaked.std(), flat.mean() + flat.std()])
    assert pooled_cutoff != pytest.approx(naive)


def test_threshold_mask_fixed_uses_cutoff_as_is():
    sim_map = np.array([[1.0, 2.0], [3.0, 4.0]])

    mask = threshold_mask(sim_map, method="fixed", cutoff=2.5)

    assert np.array_equal(mask, np.array([[False, False], [True, True]]))


def test_threshold_mask_fixed_requires_cutoff():
    sim_map = np.zeros((2, 2))
    with pytest.raises(ValueError):
        threshold_mask(sim_map, method="fixed")
