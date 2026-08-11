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
import torch.nn.functional as F

from cards.retrieval.pool import CandidatePool
from cards.retrieval.retrieve import retrieve_top_bottom_k


def _project_out(embeddings: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove the component of `embeddings` along the unit vector `direction`."""
    unit_direction = F.normalize(direction, dim=0)
    coefficients = embeddings @ unit_direction  # (N,)
    return embeddings - coefficients.unsqueeze(-1) * unit_direction


def matched_retrieval(
    pool: CandidatePool,
    present_indices: list[int],
    t_c: torch.Tensor,
) -> list[int]:
    """For each present-set image, find its nearest neighbor (by cosine
    similarity) among the rest of the pool after projecting `t_c` out of
    every embedding — i.e. matched on everything except the concept itself.

    `t_c` (Step 1's concept query), not the eventual diff-of-means direction
    `d_c`, is what's available at this point in the pipeline (Step 3 runs
    before Step 4). Returns absent_indices, same length as present_indices;
    the same neighbor may be picked more than once if it's the nearest match
    for multiple present-set images.
    """
    n = pool.embeddings.shape[0]
    present_set = set(present_indices)
    candidate_mask = torch.ones(n, dtype=torch.bool)
    candidate_mask[torch.tensor(present_indices, dtype=torch.long)] = False
    candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
    if candidate_indices.numel() == 0:
        raise ValueError("no candidates remain after excluding the present set")

    projected = _project_out(pool.embeddings, t_c)
    candidate_projected = F.normalize(projected[candidate_indices], dim=-1)
    present_projected = F.normalize(projected[torch.tensor(present_indices, dtype=torch.long)], dim=-1)

    similarities = present_projected @ candidate_projected.T  # (len(present), len(candidates))
    nearest = similarities.argmax(dim=1)
    return candidate_indices[nearest].tolist()


def stratified_retrieval(
    pool: CandidatePool,
    t_c: torch.Tensor,
    k: int,
) -> tuple[list[int], list[int]]:
    """Retrieve P_c/N_c independently within each class stratum in `pool.labels`."""
    if pool.labels is None:
        raise ValueError("stratified_retrieval requires pool.labels")

    labels = torch.tensor(pool.labels)
    present_indices: list[int] = []
    absent_indices: list[int] = []
    for class_label in sorted(set(pool.labels)):
        class_indices = (labels == class_label).nonzero(as_tuple=True)[0]
        stratum = CandidatePool(
            paths=[pool.paths[i] for i in class_indices.tolist()],
            embeddings=pool.embeddings[class_indices],
        )
        local_present, local_absent = retrieve_top_bottom_k(stratum, t_c, k)
        present_indices.extend(class_indices[torch.tensor(local_present, dtype=torch.long)].tolist())
        absent_indices.extend(class_indices[torch.tensor(local_absent, dtype=torch.long)].tolist())

    return present_indices, absent_indices
