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


class PatchLocalizableEncoder(ImageTextEncoder, Protocol):
    """An ImageTextEncoder that can also embed individual image patches in
    the SAME space as encode_images/encode_text, for text-query-driven
    spatial localization (cards.attribution.localization) with no manual
    masks or external segmenter. A narrower capability than
    ImageTextEncoder itself: not every encoder can expose this cheaply
    (see cards.encoders.open_clip_encoder/perception_encoder for the two
    architecture-specific ways of getting it), so this stays a separate
    Protocol rather than a required ImageTextEncoder method.
    """

    def encode_patches(self, image: Image.Image) -> torch.Tensor:
        """Per-patch L2-normalized embeddings for ONE image, flat shape
        (n_patches, dim) -- same space as encode_images, but one vector
        per spatial patch instead of one pooled vector for the whole
        image. n_patches is always a perfect square (the ViT's own patch
        grid, row-major); callers needing the 2D grid reshape it
        themselves (see cards.attribution.localization.localize_concept)."""
        ...
