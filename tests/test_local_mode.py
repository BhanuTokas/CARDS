"""Step 6 tests: local-mode attribution scoring."""

from __future__ import annotations

import pytest
import torch

from cards.attribution.local_mode import local_score, local_score_matched


def _black_box(images: torch.Tensor) -> torch.Tensor:
    return images.sum(dim=1)


def test_local_score_matches_manual_computation():
    x = torch.tensor([5.0, 5.0])  # b(x) = 10
    absent_images = torch.tensor([[1.0, 1.0], [3.0, 1.0]])  # outputs 2, 4 -> mean 3

    score = local_score(_black_box, x, absent_images)

    assert score == pytest.approx(7.0)  # 10 - 3


def test_local_score_matched_restricts_to_nearest_neighbors():
    x = torch.tensor([5.0, 5.0])  # b(x) = 10
    x_embedding = torch.tensor([1.0, 0.0])
    pool_embeddings = torch.tensor(
        [
            [1.0, 0.0],  # nearest to x_embedding
            [-1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    pool_images = torch.tensor(
        [
            [10.0, 10.0],  # output 20 -- this is the only one that should be used
            [0.0, 0.0],
            [100.0, 0.0],
        ]
    )

    score = local_score_matched(_black_box, x, x_embedding, pool_embeddings, pool_images, n_neighbors=1)

    assert score == pytest.approx(10.0 - 20.0)


def test_local_score_matched_rejects_n_neighbors_too_large():
    x = torch.tensor([1.0, 1.0])
    x_embedding = torch.tensor([1.0, 0.0])
    pool_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pool_images = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

    with pytest.raises(ValueError):
        local_score_matched(_black_box, x, x_embedding, pool_embeddings, pool_images, n_neighbors=5)


def test_local_score_matched_rejects_nonpositive_n_neighbors():
    x = torch.tensor([1.0, 1.0])
    x_embedding = torch.tensor([1.0, 0.0])
    pool_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pool_images = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

    with pytest.raises(ValueError):
        local_score_matched(_black_box, x, x_embedding, pool_embeddings, pool_images, n_neighbors=0)


def test_local_score_matched_rejects_mismatched_pool_lengths():
    x = torch.tensor([1.0, 1.0])
    x_embedding = torch.tensor([1.0, 0.0])
    pool_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])  # 3 embeddings
    pool_images = torch.tensor([[1.0, 1.0], [2.0, 2.0]])  # only 2 images

    with pytest.raises(ValueError):
        local_score_matched(_black_box, x, x_embedding, pool_embeddings, pool_images, n_neighbors=1)
