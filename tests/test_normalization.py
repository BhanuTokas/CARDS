"""Step 7 tests: score normalization."""

from __future__ import annotations

import pytest
import torch

from cards.attribution.normalization import (
    angular_distance,
    embedding_distance_normalize,
    variance_normalize,
)


def test_variance_normalize_matches_manual_computation():
    present_outputs = torch.tensor([2.0, 4.0])
    absent_outputs = torch.tensor([0.0, 0.0])

    result = variance_normalize(present_outputs, absent_outputs)

    combined = torch.tensor([2.0, 4.0, 0.0, 0.0])
    expected = (3.0 - 0.0) / combined.std(unbiased=True).item()
    assert result == pytest.approx(expected)


def test_variance_normalize_rejects_zero_pooled_std():
    present_outputs = torch.tensor([1.0, 1.0])
    absent_outputs = torch.tensor([1.0, 1.0])

    with pytest.raises(ValueError):
        variance_normalize(present_outputs, absent_outputs)


def test_embedding_distance_normalize():
    assert embedding_distance_normalize(raw_score=4.0, delta_c=2.0) == pytest.approx(2.0)


def test_embedding_distance_normalize_rejects_zero_delta():
    with pytest.raises(ValueError):
        embedding_distance_normalize(raw_score=4.0, delta_c=0.0)


def test_variance_normalize_rejects_near_zero_pooled_std():
    # not exactly zero, but small enough that dividing by it would blow up
    present_outputs = torch.tensor([1.0, 1.0 + 1e-10])
    absent_outputs = torch.tensor([1.0, 1.0])

    with pytest.raises(ValueError):
        variance_normalize(present_outputs, absent_outputs)


def test_embedding_distance_normalize_rejects_near_zero_delta():
    with pytest.raises(ValueError):
        embedding_distance_normalize(raw_score=4.0, delta_c=1e-10)


def test_angular_distance_identical_vectors_is_zero():
    vector = torch.tensor([1.0, 2.0, 3.0])
    assert angular_distance(vector, vector) == pytest.approx(0.0, abs=1e-6)


def test_angular_distance_orthogonal_vectors_is_one():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    assert angular_distance(a, b) == pytest.approx(1.0)


def test_angular_distance_opposite_vectors_is_two():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([-1.0, 0.0])
    assert angular_distance(a, b) == pytest.approx(2.0)
