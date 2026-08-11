"""open_clip-backed encoder.

`open_clip_torch` covers CLIP, OpenCLIP-H, and SigLIP checkpoints under one
API, which is what makes the Step 6 encoder ablation a config change rather
than a code change (see configs/encoder/*.yaml).
"""

from __future__ import annotations

import torch
from PIL import Image

from cards.encoders.base import ImageTextEncoder


class OpenClipEncoder(ImageTextEncoder):
    def __init__(self, model_name: str, pretrained: str, device: str = "cuda"):
        raise NotImplementedError

    @property
    def embed_dim(self) -> int:
        raise NotImplementedError

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        raise NotImplementedError
