"""Tests for cards.validation.broden_faithfulness's mask/placement logic
using synthetic images and arrays -- no real model or Broden data needed.
compute_faithfulness's own real-model integration is verified separately
(a smoke run against BACKBONES["resnet18"] + real Broden images)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from cards.validation.broden_faithfulness import (
    _area_matched_rectangle,
    mask_region,
    random_placements,
)

# ---- mask_region ----


def test_mask_region_blur_changes_only_masked_pixels():
    image = Image.new("RGB", (20, 20), color=(200, 50, 50))
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True

    result = mask_region(image, mask, strategy="blur")
    result_arr = np.array(result)
    original_arr = np.array(image)

    # outside the mask, a uniform-color image should survive blur unchanged
    assert np.array_equal(result_arr[0, 0], original_arr[0, 0])


def test_mask_region_zero_fill_masked_pixels_are_black():
    image = Image.new("RGB", (10, 10), color=(200, 50, 50))
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True

    result = np.array(mask_region(image, mask, strategy="zero_fill"))

    assert tuple(result[4, 4]) == (0, 0, 0)
    assert tuple(result[0, 0]) == (200, 50, 50)


def test_mask_region_mean_fill_masked_pixels_are_imagenet_mean():
    image = Image.new("RGB", (10, 10), color=(200, 50, 50))
    mask = np.ones((10, 10), dtype=bool)

    result = np.array(mask_region(image, mask, strategy="mean_fill"))

    assert tuple(result[0, 0]) == (124, 116, 104)


def test_mask_region_rejects_unknown_strategy():
    image = Image.new("RGB", (10, 10))
    mask = np.zeros((10, 10), dtype=bool)

    with pytest.raises(ValueError):
        mask_region(image, mask, strategy="nonsense")


def test_mask_region_rejects_mismatched_mask_shape():
    image = Image.new("RGB", (10, 10))
    mask = np.zeros((5, 5), dtype=bool)

    with pytest.raises(ValueError):
        mask_region(image, mask)


# ---- random_placements ----


def test_random_placements_preserves_shape_and_area():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 10:15] = True  # 10x5 region
    rng = np.random.default_rng(0)

    placements, n_fallbacks = random_placements(mask, rng, n_draws=5)

    assert len(placements) == 5
    for p in placements:
        assert p.shape == mask.shape
        # exact-shape translation preserves area exactly (no fallback expected -- small mask, big image)
        assert p.sum() == mask.sum()
    assert n_fallbacks == 0


def test_random_placements_avoids_overlapping_the_true_mask():
    mask = np.zeros((50, 50), dtype=bool)
    mask[0:10, 0:10] = True
    rng = np.random.default_rng(0)

    placements, _ = random_placements(mask, rng, n_draws=10, max_overlap_frac=0.1)

    for p in placements:
        overlap_frac = (p & mask).sum() / p.sum()
        assert overlap_frac <= 0.1 + 1e-9


def test_random_placements_falls_back_when_no_valid_offset_exists():
    # mask covering the whole image -- no translation can avoid overlapping itself
    mask = np.ones((10, 10), dtype=bool)
    rng = np.random.default_rng(0)

    placements, n_fallbacks = random_placements(mask, rng, n_draws=3, max_attempts=5)

    assert n_fallbacks == 3
    assert len(placements) == 3


def test_random_placements_empty_mask_returns_nothing():
    mask = np.zeros((10, 10), dtype=bool)
    rng = np.random.default_rng(0)

    placements, n_fallbacks = random_placements(mask, rng, n_draws=5)

    assert placements == []
    assert n_fallbacks == 0


# ---- _area_matched_rectangle ----


def test_area_matched_rectangle_approximately_preserves_area():
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:15, 5:15] = True  # area 100, a 10x10 square
    rng = np.random.default_rng(0)

    rect = _area_matched_rectangle(mask, rng)

    assert rect.shape == mask.shape
    # side = round(sqrt(100)) = 10 -> area exactly 100 here
    assert rect.sum() == 100
