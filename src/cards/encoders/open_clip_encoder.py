"""open_clip-backed encoder.

`open_clip_torch` covers CLIP, OpenCLIP-H, and SigLIP checkpoints under one
API, which is what makes the Step 6 encoder ablation a config change rather
than a code change (see configs/encoder/*.yaml).
"""

from __future__ import annotations

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

from cards.encoders.base import ImageTextEncoder


class OpenClipEncoder(ImageTextEncoder):
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: str = "cuda",
        batch_size: int = 256,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval().to(device)

    @property
    def embed_dim(self) -> int:
        return self.model.visual.output_dim

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            features = self.model.encode_text(tokens)
            chunks.append(F.normalize(features, dim=-1).cpu())
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        chunks = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            pixels = torch.stack([self.preprocess(img.convert("RGB")) for img in batch]).to(self.device)
            features = self.model.encode_image(pixels)
            chunks.append(F.normalize(features, dim=-1).cpu())
        return torch.cat(chunks, dim=0)
