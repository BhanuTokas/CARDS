"""Step 2 tests: naive top-k / bottom-k retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from cards.retrieval.pool import CandidatePool
from cards.retrieval.retrieve import retrieve_top_bottom_k


def _make_pool(n: int, dim: int = 4, seed: int = 0) -> CandidatePool:
    generator = torch.Generator().manual_seed(seed)
    embeddings = F.normalize(torch.randn(n, dim, generator=generator), dim=-1)
    paths = [Path(f"img_{i}.jpg") for i in range(n)]
    return CandidatePool(paths=paths, embeddings=embeddings)


def test_retrieve_top_bottom_k_ranks_by_cosine_similarity():
    pool = _make_pool(n=10)
    t_c = pool.embeddings[0]

    present, absent = retrieve_top_bottom_k(pool, t_c, k=3)

    similarities = pool.embeddings @ t_c
    expected_order = torch.argsort(similarities, descending=True).tolist()
    assert present == expected_order[:3]
    assert absent == expected_order[-3:]


def test_retrieve_top_bottom_k_no_overlap():
    pool = _make_pool(n=10)
    t_c = F.normalize(torch.randn(4), dim=0)

    present, absent = retrieve_top_bottom_k(pool, t_c, k=4)

    assert set(present).isdisjoint(absent)
    assert len(present) == len(absent) == 4


def test_retrieve_top_bottom_k_rejects_k_too_large():
    pool = _make_pool(n=6)
    t_c = F.normalize(torch.randn(4), dim=0)

    with pytest.raises(ValueError):
        retrieve_top_bottom_k(pool, t_c, k=4)  # 2*4 > 6


def test_retrieve_top_bottom_k_rejects_nonpositive_k():
    pool = _make_pool(n=6)
    t_c = F.normalize(torch.randn(4), dim=0)

    with pytest.raises(ValueError):
        retrieve_top_bottom_k(pool, t_c, k=0)
