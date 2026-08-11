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
    raise NotImplementedError
