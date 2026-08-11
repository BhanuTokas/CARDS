"""Encoder protocols shared across the pipeline.

Step 6 ablates the encoder (CLIP vs. OpenCLIP-H vs. SigLIP) without changing
any downstream code, so retrieval/direction/attribution code should depend
only on these protocols, never on a concrete backbone.
"""

from __future__ import annotations

from typing import Protocol

import torch
from PIL import Image


class TextEncoder(Protocol):
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Return L2-normalized text embeddings, shape (len(texts), dim)."""
        ...


class ImageEncoder(Protocol):
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Return L2-normalized image embeddings, shape (len(images), dim)."""
        ...


class ImageTextEncoder(TextEncoder, ImageEncoder, Protocol):
    """A joint image/text embedding space (CLIP-like)."""

    @property
    def embed_dim(self) -> int: ...
