"""Step 6 tests: global-mode attribution scoring."""

from __future__ import annotations

import pytest
import torch

from cards.attribution.global_mode import global_score


def _black_box(images: torch.Tensor) -> torch.Tensor:
    return images.sum(dim=1)


def test_global_score_matches_manual_computation():
    present_images = torch.tensor([[1.0, 1.0], [3.0, 1.0]])  # outputs 2, 4 -> mean 3
    absent_images = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # outputs 0, 0 -> mean 0

    raw_score, present_outputs, absent_outputs = global_score(_black_box, present_images, absent_images)

    assert raw_score == pytest.approx(3.0)
    assert torch.allclose(present_outputs, torch.tensor([2.0, 4.0]))
    assert torch.allclose(absent_outputs, torch.tensor([0.0, 0.0]))
