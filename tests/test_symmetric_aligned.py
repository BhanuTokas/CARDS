"""Tests for cards.retrieval.symmetric_aligned.symmetric_aligned_retrieval:
greedy joint selection of P_c and N_c that optimizes
cos(mean(P_c) - mean(N_c), t_c) directly, extending aligned_retrieval's
one-sided (N_c-only) search to both sides."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from cards.retrieval.pool import CandidatePool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.retrieval.symmetric_aligned import symmetric_aligned_retrieval


def _pool(embeddings: list[list[float]]) -> CandidatePool:
    tensor = torch.tensor(embeddings, dtype=torch.float32)
    paths = [Path(f"img_{i}.jpg") for i in range(len(embeddings))]
    return CandidatePool(paths=paths, embeddings=tensor)


def _cosine_align(pool: CandidatePool, present_indices, absent_indices, t_c) -> float:
    mean_present = pool.embeddings[present_indices].mean(dim=0)
    mean_absent = pool.embeddings[absent_indices].mean(dim=0)
    d_c = mean_present - mean_absent
    return F.cosine_similarity(d_c.unsqueeze(0), F.normalize(t_c, dim=0).unsqueeze(0)).item()


def test_symmetric_aligned_retrieval_beats_naive_on_present_side():
    # Mirrors aligned_retrieval's own hand-engineered test (see
    # test_aligned.py) but on the *present* side: P1+P2 are each
    # individually more aligned with t_c than P3, so naive top-k picks
    # {P1, P2} -- but P2+P3's off-axis components cancel while P1+P2's
    # both point the same way, giving a strictly better aggregate
    # alignment. N1/N2 are engineered to have no rivals on the absent
    # side, so they stay fixed and isolate the present-side behavior.
    pool = _pool(
        [
            [-9.0, 0.0],  # N1 (index 0) -- absent, no better candidate exists
            [-9.0, 0.0001],  # N2 (index 1) -- absent, no better candidate exists
            [9.0, 3.0],  # P1 (index 2) -- best individually
            [9.0, 4.0],  # P2 (index 3) -- 2nd best -- naive top-2 picks this
            [9.0, -4.5],  # P3 (index 4) -- 3rd best -- naive skips this
        ]
    )
    t_c = torch.tensor([1.0, 0.0])

    naive_present, naive_absent = retrieve_top_bottom_k(pool, t_c, k=2)
    assert set(naive_present) == {2, 3}
    assert set(naive_absent) == {0, 1}
    naive_score = _cosine_align(pool, naive_present, naive_absent, t_c)

    present, absent = symmetric_aligned_retrieval(pool, t_c, k=2)
    result_score = _cosine_align(pool, present, absent, t_c)

    # The absent side has no better option available, so it stays put;
    # the present side swaps out P1 for P3, landing on {P2, P3}.
    assert set(absent) == {0, 1}
    assert set(present) == {3, 4}
    assert result_score > naive_score
    assert result_score == pytest.approx(0.999904, abs=1e-4)
    assert naive_score == pytest.approx(0.981616, abs=1e-4)


def test_symmetric_aligned_retrieval_never_worse_than_naive_warm_start():
    # General invariant, true by construction regardless of data: the
    # search starts at the naive top-k/bottom-k warm start and only
    # accepts strict improvements on either side, so it can never end up
    # worse than that starting point.
    torch.manual_seed(0)
    embeddings = F.normalize(torch.randn(30, 6), dim=-1)
    paths = [Path(f"img_{i}.jpg") for i in range(30)]
    pool = CandidatePool(paths=paths, embeddings=embeddings)
    t_c = F.normalize(torch.randn(6), dim=0)

    naive_present, naive_absent = retrieve_top_bottom_k(pool, t_c, k=5)
    naive_score = _cosine_align(pool, naive_present, naive_absent, t_c)

    present, absent = symmetric_aligned_retrieval(pool, t_c, k=5)
    result_score = _cosine_align(pool, present, absent, t_c)

    assert result_score >= naive_score - 1e-6


def test_symmetric_aligned_retrieval_returns_disjoint_k_sized_sets():
    torch.manual_seed(1)
    embeddings = F.normalize(torch.randn(20, 4), dim=-1)
    paths = [Path(f"img_{i}.jpg") for i in range(20)]
    pool = CandidatePool(paths=paths, embeddings=embeddings)
    t_c = F.normalize(torch.randn(4), dim=0)

    present, absent = symmetric_aligned_retrieval(pool, t_c, k=4)

    assert len(present) == 4
    assert len(absent) == 4
    assert set(present).isdisjoint(absent)


def test_symmetric_aligned_retrieval_rejects_non_positive_k():
    pool = _pool([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [-1.0, 0.0]])
    with pytest.raises(ValueError):
        symmetric_aligned_retrieval(pool, torch.tensor([1.0, 0.0]), k=0)


def test_symmetric_aligned_retrieval_rejects_k_too_large_for_pool():
    pool = _pool([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    with pytest.raises(ValueError):
        symmetric_aligned_retrieval(pool, torch.tensor([1.0, 0.0]), k=2)


def test_present_candidate_pool_size_blocks_swap_to_irrelevant_image():
    # Reproduces, in miniature, the failure mode diagnosed in v12
    # (notes/pcbm_correlation_investigation.md): unconstrained, the
    # search swaps P1 out for an image that's individually almost
    # irrelevant to t_c (cosine ~0.011) purely because it improves the
    # aggregate cos(d_c, t_c) fit. Restricting present_candidate_pool_size
    # to exactly k (i.e. only naive top-k itself is eligible) should
    # block that swap entirely, leaving P_c at the naive warm start.
    pool = _pool(
        [
            [-9.0, 0.0],  # N1 (index 0) -- absent, no better candidate exists
            [-9.0, 0.0001],  # N2 (index 1) -- absent, no better candidate exists
            [9.0, 3.0],  # P1 (index 2) -- best individually
            [9.0, 4.0],  # P2 (index 3) -- 2nd best -- naive top-2 picks this
            [0.1, -9.0],  # irrelevant candidate (index 4) -- cosine ~0.011, helps the aggregate fit but not individually relevant
        ]
    )
    t_c = torch.tensor([1.0, 0.0])

    unconstrained_present, unconstrained_absent = symmetric_aligned_retrieval(pool, t_c, k=2)
    assert set(unconstrained_present) == {3, 4}  # swaps in the irrelevant candidate

    constrained_present, constrained_absent = symmetric_aligned_retrieval(
        pool, t_c, k=2, present_candidate_pool_size=2
    )
    assert set(constrained_present) == {2, 3}  # blocked -- stays at naive top-k
    assert set(constrained_absent) == {0, 1}

    unconstrained_score = _cosine_align(pool, unconstrained_present, unconstrained_absent, t_c)
    constrained_score = _cosine_align(pool, constrained_present, constrained_absent, t_c)
    assert unconstrained_score == pytest.approx(0.983401, abs=1e-4)
    assert constrained_score == pytest.approx(0.981616, abs=1e-4)


def test_present_candidate_pool_size_rejects_value_below_k():
    pool = _pool([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [-1.0, 0.0]])
    with pytest.raises(ValueError):
        symmetric_aligned_retrieval(pool, torch.tensor([1.0, 0.0]), k=2, present_candidate_pool_size=1)
