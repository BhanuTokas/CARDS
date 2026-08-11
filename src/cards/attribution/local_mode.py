"""Step 6 — Local mode (black-box analogue of CCE).

score_c(x) = b(x) - mean(b(N_c)), reusing the same N_c retrieved for global
mode. Ablation: compare against a locally-matched N_c (restricted to x's
nearest neighbors) to reduce the confound exposure the global N_c mean has
for a single unmatched x.
"""

from __future__ import annotations

from typing import Callable

import torch


def local_score(
    black_box: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    absent_images: torch.Tensor,
) -> float:
    raise NotImplementedError


def local_score_matched(
    black_box: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    x_embedding: torch.Tensor,
    pool_embeddings: torch.Tensor,
    pool_images: torch.Tensor,
    n_neighbors: int,
) -> float:
    """Ablation variant: mean(b(N_c)) restricted to N_c images nearest to x
    in embedding space, instead of the full global N_c."""
    raise NotImplementedError
