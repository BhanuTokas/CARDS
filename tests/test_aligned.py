"""Tests for cards.retrieval.aligned.aligned_retrieval: greedy negative
selection that optimizes cos(mean(P_c) - mean(N_c), t_c) directly, rather
than ranking each candidate independently by similarity to t_c."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from cards.retrieval.aligned import aligned_retrieval
from cards.retrieval.pool import CandidatePool


def _pool(embeddings: list[list[float]]) -> CandidatePool:
    tensor = torch.tensor(embeddings, dtype=torch.float32)
    paths = [Path(f"img_{i}.jpg") for i in range(len(embeddings))]
    return CandidatePool(paths=paths, embeddings=tensor)


def _cosine_align(pool: CandidatePool, present_indices, absent_indices, t_c) -> float:
    mean_present = pool.embeddings[present_indices].mean(dim=0)
    mean_absent = pool.embeddings[absent_indices].mean(dim=0)
    d_c = mean_present - mean_absent
    return F.cosine_similarity(d_c.unsqueeze(0), F.normalize(t_c, dim=0).unsqueeze(0)).item()


def test_aligned_retrieval_beats_naive_bottom_k_warm_start():
    # Engineered so naive bottom-k (ranks each candidate independently)
    # picks C1+C2 (both individually more anti-aligned with t_c than C3),
    # but C1+C2's off-axis components both point the same way and add up,
    # while C1+C3's off-axis components point opposite ways and cancel --
    # giving a strictly better aggregate cosine alignment despite C3
    # being individually the "worst" of the three candidates.
    pool = _pool(
        [
            [10.0, 0.0],  # present (index 0)
            [-9.0, 3.0],  # C1 (index 1) -- cosine ~ -0.9487, best individually
            [-9.0, 4.0],  # C2 (index 2) -- cosine ~ -0.9138, 2nd best -- naive picks this
            [-9.0, -4.5],  # C3 (index 3) -- cosine ~ -0.8944, worst -- naive skips this
        ]
    )
    t_c = torch.tensor([1.0, 0.0])
    present_indices = [0]

    naive_warm_start = [1, 2]  # what bottom-2-by-cosine picks
    naive_score = _cosine_align(pool, present_indices, naive_warm_start, t_c)

    result = aligned_retrieval(pool, present_indices, t_c, k=2)
    result_score = _cosine_align(pool, present_indices, result, t_c)

    # {C2, C3} (indices 2, 3) turns out to beat both the naive warm start
    # {C1, C2} and the also-better-than-naive {C1, C3} -- the greedy
    # search finds this via a swap even though it wasn't the first
    # alternative considered by hand.
    assert set(result) == {2, 3}
    assert result_score > naive_score
    assert result_score == pytest.approx(0.999913, abs=1e-4)
    assert naive_score == pytest.approx(0.983453, abs=1e-4)


def test_aligned_retrieval_never_worse_than_naive_warm_start():
    # General invariant, true by construction regardless of data: the
    # search starts at the naive warm start and only accepts strict
    # improvements, so it can never end up worse.
    torch.manual_seed(0)
    embeddings = F.normalize(torch.randn(30, 6), dim=-1)
    paths = [Path(f"img_{i}.jpg") for i in range(30)]
    pool = CandidatePool(paths=paths, embeddings=embeddings)
    t_c = F.normalize(torch.randn(6), dim=0)
    present_indices = [0, 1, 2]

    naive_order = torch.argsort(pool.embeddings[3:] @ t_c)
    naive_warm_start = (naive_order[:5] + 3).tolist()
    naive_score = _cosine_align(pool, present_indices, naive_warm_start, t_c)

    result = aligned_retrieval(pool, present_indices, t_c, k=5)
    result_score = _cosine_align(pool, present_indices, result, t_c)

    assert result_score >= naive_score - 1e-6


def test_aligned_retrieval_returns_k_indices_disjoint_from_present():
    pool = _pool([[float(i), 0.0] for i in range(10)])
    t_c = torch.tensor([1.0, 0.0])
    present_indices = [0, 1]

    result = aligned_retrieval(pool, present_indices, t_c, k=3)

    assert len(result) == 3
    assert set(result).isdisjoint(present_indices)


def test_aligned_retrieval_rejects_non_positive_k():
    pool = _pool([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    with pytest.raises(ValueError):
        aligned_retrieval(pool, [0], torch.tensor([1.0, 0.0]), k=0)


def test_aligned_retrieval_rejects_too_few_candidates():
    pool = _pool([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    with pytest.raises(ValueError):
        aligned_retrieval(pool, [0], torch.tensor([1.0, 0.0]), k=5)
