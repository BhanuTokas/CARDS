"""Step 7 — Score normalization.

Default: output-variance normalization (Cohen's-d-style effect size).
Ablation: embedding-distance normalization, divided by Delta_c (Step 4),
with a choice of distance function for Delta_c itself (Euclidean vs.
angular between P_c/N_c centroids — angular is favored for consistency
with the cosine-similarity retrieval in Step 2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def variance_normalize(
    present_outputs: torch.Tensor,
    absent_outputs: torch.Tensor,
) -> float:
    """(mean(P_c) - mean(N_c)) / pooled_std(P_c union N_c)."""
    pooled_std = torch.cat([present_outputs, absent_outputs]).std(unbiased=True)
    if pooled_std.item() == 0.0:
        raise ValueError("pooled_std is zero; present/absent outputs have no variance")
    raw_score = present_outputs.mean() - absent_outputs.mean()
    return (raw_score / pooled_std).item()


def embedding_distance_normalize(raw_score: float, delta_c: float) -> float:
    """raw_score / Delta_c."""
    if delta_c == 0.0:
        raise ValueError("delta_c is zero; cannot normalize by embedding distance")
    return raw_score / delta_c


def angular_distance(centroid_a: torch.Tensor, centroid_b: torch.Tensor) -> float:
    """1 - cosine_similarity(centroid_a, centroid_b)."""
    cosine_similarity = F.cosine_similarity(centroid_a.unsqueeze(0), centroid_b.unsqueeze(0)).item()
    return 1.0 - cosine_similarity
