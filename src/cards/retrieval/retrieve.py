"""Step 2 — Distribution retrieval (concept-present / concept-absent).

Naive top-k / bottom-k by cosine similarity to `t_c`. This is the baseline
ablation arm; the default confound-controlled retrieval lives in confound.py.
"""

from __future__ import annotations

import torch

from cards.retrieval.pool import CandidatePool


def retrieve_top_bottom_k(
    pool: CandidatePool,
    t_c: torch.Tensor,
    k: int,
) -> tuple[list[int], list[int]]:
    """Return (present_indices, absent_indices) into `pool`, top-k / bottom-k by cosine sim to t_c."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    n = pool.embeddings.shape[0]
    if 2 * k > n:
        raise ValueError(f"2*k ({2 * k}) exceeds pool size ({n}); shrink k or grow the pool")

    similarities = pool.embeddings @ t_c  # both L2-normalized -> cosine similarity
    order = torch.argsort(similarities, descending=True)
    present_indices = order[:k].tolist()
    absent_indices = order[-k:].tolist()
    return present_indices, absent_indices
