"""Step 6 — Global mode (black-box analogue of TCAV).

score_c = mean(b(P_c)) - mean(b(N_c)): compare the black-box model's outputs
directly across the retrieved positive/negative distributions. No per-image
embedding shift needed.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


@torch.no_grad()
def global_score(
    black_box: Callable[[torch.Tensor], torch.Tensor],
    present_images: torch.Tensor,
    absent_images: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Returns (raw_score, b(P_c) outputs, b(N_c) outputs) — the latter two
    are needed by Step 7's output-variance normalization."""
    present_outputs = black_box(present_images)
    absent_outputs = black_box(absent_images)
    raw_score = (present_outputs.mean() - absent_outputs.mean()).item()
    return raw_score, present_outputs, absent_outputs
