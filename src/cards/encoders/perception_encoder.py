"""Perception Encoder (Meta FAIR, PE-Core) adapter for CARDS'
ImageTextEncoder protocol.

Not on PyPI -- like `cards.models.posthoc_cbm`'s handling of the
post_hoc_cbm repo, this expects facebookresearch/perception_models
cloned as a sibling directory (or wherever `perception_models_path`
points) and added to `sys.path` at runtime, NOT pip-installed via
`pip install -e .`. That's deliberate: the repo's requirements.txt pulls
in an exact-pinned `numpy==2.1.2` plus a large research/training stack
(wandb, lm-eval, decord, webdataset, datatrove, viztracer, ...) meant for
their full training/eval pipeline, none of which is needed just to run
inference with the encoder -- installing all of that risks destabilizing
CARDS' own environment (the numpy pin alone could force a downgrade that
breaks other deps). The actual runtime dependencies of
`core.vision_encoder.pe`/`transforms`/`tokenizer` (traced directly from
their imports) are much smaller: einops, timm, ftfy, regex -- CARDS' own
`perception` extra (`uv sync --extra perception`) -- plus torch/
torchvision/numpy/huggingface_hub, already CARDS deps.

License note: `core.vision_encoder` (what this module imports) is under
that repo's LICENSE.PE, plain Apache 2.0. The restrictive "FAIR
Noncommercial Research License" (LICENSE.PLM) covers a different part of
that repo (their language model, not used here) -- checked directly
against both license files before adding this adapter.

PE-Core checkpoint names as of writing: PE-Core-T16-384, PE-Core-S16-384,
PE-Core-B16-224, PE-Core-L14-336, PE-Core-G14-448 (larger = better/
slower, matching CARDS' existing encoder-size ablation pattern of
clip.yaml vs. open_clip_h.yaml).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

from cards.encoders.base import ImageTextEncoder


class PerceptionEncoder(ImageTextEncoder):
    def __init__(
        self,
        model_name: str,
        perception_models_path: str,
        device: str = "cuda",
        batch_size: int = 256,
    ):
        repo_path = str(Path(perception_models_path).resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as pe_transforms

        self.device = device
        self.batch_size = batch_size
        self.model = pe.CLIP.from_config(model_name, pretrained=True)
        self.model = self.model.to(device).eval()
        self.preprocess = pe_transforms.get_image_transform(self.model.image_size)
        self.tokenizer = pe_transforms.get_text_tokenizer(self.model.context_length)
        self._embed_dim: int | None = None

    @property
    def embed_dim(self) -> int:
        # PE-Core doesn't expose one top-level embedding-dim attribute
        # (vision/text widths are tracked separately internally) -- probe
        # via a real forward pass instead, same approach as
        # OpenClipEncoder.embed_dim for the same reason (uniform across
        # backbone internals CARDS doesn't want to depend on).
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
            features = self.model.encode_text(tokens, normalize=True)
            chunks.append(features.cpu())
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        chunks = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            pixels = torch.stack([self.preprocess(img.convert("RGB")) for img in batch]).to(self.device)
            features = self.model.encode_image(pixels, normalize=True)
            chunks.append(features.cpu())
        return torch.cat(chunks, dim=0)
