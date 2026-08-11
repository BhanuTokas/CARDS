"""Step 4 — Direction estimation.

d_c = mean(phi(P_c)) - mean(phi(N_c)), the standard difference-in-means
estimator. Raw pre-normalization magnitude Delta_c = ||d_c|| is retained for
Step 7's embedding-distance normalization option.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ConceptDirection:
    concept: str
    unit_vector: torch.Tensor  # d_c, normalized
    magnitude: float  # Delta_c, pre-normalization ||mean(P_c) - mean(N_c)||


def estimate_direction(
    concept: str,
    present_embeddings: torch.Tensor,
    absent_embeddings: torch.Tensor,
) -> ConceptDirection:
    raise NotImplementedError
