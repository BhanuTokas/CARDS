"""Candidate image pool: encode once, retrieve many times.

Pool source is validation-set-first with a training-set fallback (see the
open design decisions checklist) — kept as a constructor argument, not a
hardcoded path, so both are drop-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from cards.encoders.base import ImageEncoder


@dataclass
class CandidatePool:
    paths: list[Path]
    embeddings: torch.Tensor  # (N, dim), L2-normalized
    labels: list[int] | None = None  # only populated when needed (Step 3 stratify-by-class)

    @classmethod
    def build(
        cls,
        image_dir: Path,
        encoder: ImageEncoder,
        labels: list[int] | None = None,
        batch_size: int = 256,
    ) -> "CandidatePool":
        raise NotImplementedError
