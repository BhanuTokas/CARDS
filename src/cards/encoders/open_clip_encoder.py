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
        self._embed_dim: int | None = None

    @property
    def embed_dim(self) -> int:
        # open_clip's visual backbone class varies by model (plain ViT,
        # ResNet, timm-backed models like SigLIP), each exposing the output
        # dim under a different attribute name (or none at all, for
        # TimmModel) -- probing a real forward pass is the one thing that
        # works uniformly across all of them.
        if self._embed_dim is None:
            probe = Image.new("RGB", (8, 8))
            self._embed_dim = self.encode_images([probe]).shape[1]
        return self._embed_dim

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

    @torch.no_grad()
    def encode_patches(self, image: Image.Image) -> torch.Tensor:
        """Per-patch embeddings in the SAME space as encode_images, via
        one of two architecture-specific tricks (both verified by direct
        reconstruction against encode_image, cos=1.0):

        - SigLIP (a timm-backed TimmModel under open_clip, `model.visual`
          exposes `.trunk`): SigLIP pools through a learned nonlinear
          AttentionPoolLatent (MAP head), not a CLS token + linear
          projection -- a patch's raw pre-pool hidden state does NOT
          live in the final embedding space. Fix: route each patch
          through that SAME attn_pool as if it were the only token in a
          length-1 sequence -- attention over one token is trivially
          identity, so this deterministically yields a per-patch vector
          in the exact final space.
        - Plain open_clip ViT (CLIP, open_clip_h -- `attn_pool=None`,
          `pool_type='tok'`): exposes a genuinely separate LINEAR
          `visual.proj` applied after `visual.ln_post`. Since `ln_post`
          applies elementwise to every token (not just the pooled one)
          BEFORE pooling, applying `proj` directly to patch tokens is
          architecturally EXACT here -- no attn_pool trick needed.
        """
        pixels = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        visual = self.model.visual
        if hasattr(visual, "trunk"):
            feats = visual.trunk.forward_features(pixels)  # (1, N, C)
            per_patch = visual.trunk.attn_pool(feats[0].unsqueeze(1))  # (N, C)
        else:
            feats = visual.transformer(visual._embeds(pixels))
            ln = visual.ln_post(feats)  # (1, N+1, C) -- index 0 is the CLS token
            per_patch = ln[0, 1:] @ visual.proj  # (N, C)
        return F.normalize(per_patch, dim=-1).cpu()
