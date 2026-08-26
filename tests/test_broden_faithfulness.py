"""Tests for cards.validation.broden_faithfulness's mask/placement logic
using synthetic images and arrays -- no real model or Broden data needed.
compute_faithfulness's own real-model integration is verified separately
(a smoke run against BACKBONES["resnet18"] + real Broden images)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import torch

from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    _area_matched_rectangle,
    compute_faithfulness,
    mask_region,
    random_placements,
    score_method_agreement,
    score_sign_agreement,
)


def _fr(concept_number, predicted_class, delta_p) -> FaithfulnessResult:
    """Minimal FaithfulnessResult for score_method_agreement/
    score_sign_agreement tests -- only concept_number/predicted_class/
    delta_p matter to those functions, everything else is filler."""
    return FaithfulnessResult(
        image="x.jpg", concept_number=concept_number, category="test", predicted_class=predicted_class,
        p0=0.5, p_masked=0.5 - delta_p, delta_p=delta_p, delta_logit=0.0,
        random_delta_p_mean=0.0, random_delta_p_std=0.0, z_score=0.0, n_random_fallbacks=0,
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


# ---- compute_faithfulness's target_class override ----


class _FixedLogitsModel:
    """Always returns the same logits regardless of input -- argmax is
    always class 0, so passing target_class=2 is the only way to get a
    predicted_class of 2, isolating the override behavior."""

    def preprocess(self, image):
        return torch.zeros(3, 4, 4)

    def __call__(self, batch):
        return torch.tensor([[2.0, 1.0, 0.5]]).repeat(batch.shape[0], 1)


def test_compute_faithfulness_defaults_to_argmax():
    image = Image.new("RGB", (4, 4), color=(100, 100, 100))
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    result = compute_faithfulness(
        image=image, image_path="x.jpg", concept_number=1, category="test",
        mask=mask, model=_FixedLogitsModel(), rng=np.random.default_rng(0),
    )

    assert result.predicted_class == 0  # argmax of [2.0, 1.0, 0.5]


def test_compute_faithfulness_target_class_overrides_argmax():
    image = Image.new("RGB", (4, 4), color=(100, 100, 100))
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    result = compute_faithfulness(
        image=image, image_path="x.jpg", concept_number=1, category="test",
        mask=mask, model=_FixedLogitsModel(), rng=np.random.default_rng(0),
        target_class=2,
    )

    assert result.predicted_class == 2
    import torch.nn.functional as F

    expected_p0 = F.softmax(torch.tensor([2.0, 1.0, 0.5]), dim=0)[2].item()
    assert result.p0 == pytest.approx(expected_p0)


# ---- score_method_agreement / score_sign_agreement ----


def test_score_method_agreement_returns_none_below_three_pairs():
    records = [_fr(1, 10, 0.1), _fr(1, 10, 0.2), _fr(1, 10, 0.15)]  # only 1 unique pair

    result = score_method_agreement(records, {(1, 10): 5.0})

    assert result is None


def test_score_method_agreement_perfect_correlation():
    # delta_p and method score both increase together across 3 pairs
    records = [
        _fr(1, 10, 0.1), _fr(1, 10, 0.1), _fr(1, 10, 0.1),
        _fr(1, 20, 0.2), _fr(1, 20, 0.2), _fr(1, 20, 0.2),
        _fr(1, 30, 0.3), _fr(1, 30, 0.3), _fr(1, 30, 0.3),
    ]
    scores = {(1, 10): 1.0, (1, 20): 2.0, (1, 30): 3.0}

    result = score_method_agreement(records, scores)

    assert result.n_pairs == 3
    assert result.spearman_rho == pytest.approx(1.0)


def test_score_sign_agreement_returns_none_below_three_pairs():
    records = [_fr(1, 10, 0.1), _fr(1, 10, 0.2), _fr(1, 10, 0.15)]

    result = score_sign_agreement(records, {(1, 10): 5.0})

    assert result is None


def test_score_sign_agreement_perfect_agreement():
    records = [
        _fr(1, 10, 0.1), _fr(1, 10, 0.1), _fr(1, 10, 0.1),   # positive delta_p
        _fr(1, 20, -0.2), _fr(1, 20, -0.2), _fr(1, 20, -0.2),  # negative delta_p
        _fr(1, 30, 0.3), _fr(1, 30, 0.3), _fr(1, 30, 0.3),   # positive delta_p
    ]
    scores = {(1, 10): 1.0, (1, 20): -1.0, (1, 30): 2.0}  # same signs throughout

    result = score_sign_agreement(records, scores)

    assert result.n_pairs == 3
    assert result.n_agree == 3
    assert result.agreement_frac == pytest.approx(1.0)


def test_score_sign_agreement_perfect_disagreement():
    records = [
        _fr(1, 10, 0.1), _fr(1, 10, 0.1), _fr(1, 10, 0.1),
        _fr(1, 20, -0.2), _fr(1, 20, -0.2), _fr(1, 20, -0.2),
        _fr(1, 30, 0.3), _fr(1, 30, 0.3), _fr(1, 30, 0.3),
    ]
    scores = {(1, 10): -1.0, (1, 20): 1.0, (1, 30): -2.0}  # every sign flipped

    result = score_sign_agreement(records, scores)

    assert result.n_agree == 0
    assert result.agreement_frac == pytest.approx(0.0)


def test_score_sign_agreement_respects_method_threshold():
    # TCAV-style score in [0, 1], centered at 0.5 rather than 0
    records = [
        _fr(1, 10, 0.1), _fr(1, 10, 0.1), _fr(1, 10, 0.1),   # positive delta_p
        _fr(1, 20, -0.2), _fr(1, 20, -0.2), _fr(1, 20, -0.2),  # negative delta_p
        _fr(1, 30, 0.3), _fr(1, 30, 0.3), _fr(1, 30, 0.3),   # positive delta_p
    ]
    scores = {(1, 10): 0.9, (1, 20): 0.1, (1, 30): 0.6}  # above/below/above 0.5 respectively

    result = score_sign_agreement(records, scores, method_threshold=0.5)

    assert result.n_agree == 3
    assert result.agreement_frac == pytest.approx(1.0)
