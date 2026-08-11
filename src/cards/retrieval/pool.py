"""Candidate image pool: encode once, retrieve many times.

Pool source is validation-set-first with a training-set fallback (see the
open design decisions checklist) — kept as a constructor argument, not a
hardcoded path, so both are drop-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from cards.encoders.base import ImageEncoder

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
    ) -> CandidatePool:
        paths = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            raise ValueError(f"no images found under {image_dir}")
        if labels is not None and len(labels) != len(paths):
            raise ValueError(f"labels length ({len(labels)}) != number of images ({len(paths)})")

        chunks = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = [Image.open(p) for p in batch_paths]
            chunks.append(encoder.encode_images(images))
        embeddings = torch.cat(chunks, dim=0)

        return cls(paths=paths, embeddings=embeddings, labels=labels)
