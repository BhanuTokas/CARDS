"""Step 3 — Confound control.

Two matching strategies:
  - `matched_retrieval` (default): for each P_c image, retrieve its nearest
    neighbor by angular distance after projecting out the concept direction.
    Doesn't require naming the confound in advance.
  - `stratified_retrieval`: retrieve P_c/N_c separately within each class
    stratum. Used when the confound is known ahead of time and is
    class-conditional (e.g. the CCE MetaDataset benchmark).
"""

from __future__ import annotations

import torch

from cards.retrieval.pool import CandidatePool


def matched_retrieval(
    pool: CandidatePool,
    t_c: torch.Tensor,
    present_indices: list[int],
    concept_direction: torch.Tensor,
) -> list[int]:
    """For each present-set image, find its nearest neighbor after projecting
    `concept_direction` out of the pool embeddings. Returns absent_indices."""
    raise NotImplementedError


def stratified_retrieval(
    pool: CandidatePool,
    t_c: torch.Tensor,
    k: int,
) -> tuple[list[int], list[int]]:
    """Retrieve P_c/N_c independently within each class stratum in `pool.labels`."""
    raise NotImplementedError
