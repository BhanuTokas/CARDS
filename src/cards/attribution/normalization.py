"""Step 7 — Score normalization.

Default: output-variance normalization (Cohen's-d-style effect size).
Ablation: embedding-distance normalization, divided by Delta_c (Step 4),
with a choice of distance function for Delta_c itself (Euclidean vs.
angular between P_c/N_c centroids — angular is favored for consistency
with the cosine-similarity retrieval in Step 2).
"""

from __future__ import annotations

import torch


def variance_normalize(
    present_outputs: torch.Tensor,
    absent_outputs: torch.Tensor,
) -> float:
    """(mean(P_c) - mean(N_c)) / pooled_std(P_c union N_c)."""
    raise NotImplementedError


def embedding_distance_normalize(raw_score: float, delta_c: float) -> float:
    """raw_score / Delta_c."""
    raise NotImplementedError


def angular_distance(centroid_a: torch.Tensor, centroid_b: torch.Tensor) -> float:
    """1 - cosine_similarity(centroid_a, centroid_b)."""
    raise NotImplementedError
