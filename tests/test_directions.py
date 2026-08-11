"""Step 4 tests: diff-of-means direction estimation."""

from __future__ import annotations

import pytest
import torch

from cards.directions.estimate import estimate_direction


def test_estimate_direction_matches_diff_of_means():
    present = torch.tensor([[1.0, 0.0], [3.0, 0.0]])  # mean [2, 0]
    absent = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # mean [0, 0]

    direction = estimate_direction("concept", present, absent)

    assert direction.concept == "concept"
    assert direction.magnitude == pytest.approx(2.0)
    assert torch.allclose(direction.unit_vector, torch.tensor([1.0, 0.0]))


def test_estimate_direction_unit_vector_is_unit_norm():
    present = torch.tensor([[3.0, 4.0]])
    absent = torch.tensor([[0.0, 0.0]])

    direction = estimate_direction("concept", present, absent)

    assert direction.magnitude == pytest.approx(5.0)
    assert torch.isclose(direction.unit_vector.norm(), torch.tensor(1.0), atol=1e-6)


def test_estimate_direction_rejects_zero_magnitude():
    present = torch.tensor([[1.0, 1.0], [2.0, 2.0]])  # mean [1.5, 1.5]
    absent = torch.tensor([[1.5, 1.5]])  # same mean

    with pytest.raises(ValueError):
        estimate_direction("concept", present, absent)


def test_estimate_direction_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        estimate_direction("concept", torch.tensor([1.0, 2.0]), torch.tensor([[1.0, 2.0]]))
