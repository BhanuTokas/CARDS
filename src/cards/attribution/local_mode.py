"""Step 6 — Local mode (black-box analogue of CCE).

score_c(x) = b(x) - mean(b(N_c)), reusing the same N_c retrieved for global
mode. Ablation: compare against a locally-matched N_c (restricted to x's
nearest neighbors) to reduce the confound exposure the global N_c mean has
for a single unmatched x.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


@torch.no_grad()
def local_score(
    black_box: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    absent_images: torch.Tensor,
) -> float:
    """`x` is a single unbatched image; `absent_images` is the batched N_c
    reused from global mode."""
    x_output = black_box(x.unsqueeze(0)).squeeze(0)
    absent_outputs = black_box(absent_images)
    return (x_output - absent_outputs.mean()).item()


@torch.no_grad()
def local_score_matched(
    black_box: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    x_embedding: torch.Tensor,
    pool_embeddings: torch.Tensor,
    pool_images: torch.Tensor,
    n_neighbors: int,
) -> float:
    """Ablation variant: mean(b(N_c)) restricted to N_c images nearest to x
    in embedding space, instead of the full global N_c. `pool_embeddings`
    and `pool_images` are N_c's embeddings/images (assumed L2-normalized,
    same convention as retrieval), not the full candidate pool.
    """
    n = pool_embeddings.shape[0]
    if n_neighbors <= 0:
        raise ValueError(f"n_neighbors must be positive, got {n_neighbors}")
    if n_neighbors > n:
        raise ValueError(f"n_neighbors ({n_neighbors}) exceeds N_c size ({n})")

    similarities = pool_embeddings @ x_embedding
    nearest = torch.topk(similarities, k=n_neighbors).indices
    return local_score(black_box, x, pool_images[nearest])
