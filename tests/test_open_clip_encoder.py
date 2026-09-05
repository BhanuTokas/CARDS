"""Tests for cards.encoders.open_clip_encoder's encode_patches dispatch
logic. Constructing a real OpenClipEncoder needs a real checkpoint
download -- integration-only, not tested here (same approach as
test_perception_encoder.py). These tests fake just enough of
open_clip's own `model.visual` internals to exercise the SigLIP
(attn_pool/timm) and plain-ViT (linear proj) dispatch branches and their
math, without any real model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image

from cards.encoders.open_clip_encoder import OpenClipEncoder


def _make_encoder(model) -> OpenClipEncoder:
    encoder = OpenClipEncoder.__new__(OpenClipEncoder)
    encoder.device = "cpu"
    encoder.batch_size = 8
    encoder.model = model
    encoder.preprocess = lambda img: torch.zeros(3, 4, 4)
    encoder._embed_dim = None
    return encoder


# ---- SigLIP path (model.visual has a .trunk -- timm-backed TimmModel) ----


class _FakeAttnPool:
    """A tiny deterministic stand-in for AttentionPoolLatent -- doubles
    whatever it's given, applied to (N, 1, C) input as encode_patches'
    own length-1-sequence trick does."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 1  # (N, 1, C), the length-1-sequence trick
        return x.squeeze(1) * 2.0


class _FakeTrunk:
    def __init__(self, feats: torch.Tensor):
        self._feats = feats  # (1, N, C)
        self.attn_pool = _FakeAttnPool()

    def forward_features(self, pixels: torch.Tensor) -> torch.Tensor:
        return self._feats


class _FakeSigLIPVisual:
    def __init__(self, feats: torch.Tensor):
        self.trunk = _FakeTrunk(feats)


def test_encode_patches_siglip_path_uses_attn_pool_trick():
    feats = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)  # 4 patches, dim 3
    model = type("M", (), {"visual": _FakeSigLIPVisual(feats)})()
    encoder = _make_encoder(model)

    result = encoder.encode_patches(Image.new("RGB", (16, 16)))

    expected = F.normalize(feats[0] * 2.0, dim=-1)
    assert result.shape == (4, 3)
    assert torch.allclose(result, expected, atol=1e-5)


# ---- plain open_clip ViT path (no .trunk -- CLIP/open_clip_h) ----


class _FakePlainVisual:
    """No `.trunk` attribute at all -- encode_patches' own hasattr check
    must fall through to the linear-proj branch."""

    def __init__(self, embeds: torch.Tensor, proj: torch.Tensor):
        self._embeds_out = embeds  # (1, N+1, C), index 0 = CLS token
        self.proj = proj

    def _embeds(self, pixels: torch.Tensor) -> torch.Tensor:
        return self._embeds_out

    def transformer(self, x: torch.Tensor) -> torch.Tensor:
        return x  # identity, so ln_post sees _embeds_out unchanged

    def ln_post(self, x: torch.Tensor) -> torch.Tensor:
        return x  # identity


def test_encode_patches_plain_vit_path_uses_linear_projection():
    embeds = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)  # CLS + 4 patches, dim 3
    proj = torch.eye(3) * 2.0
    model = type("M", (), {"visual": _FakePlainVisual(embeds, proj)})()
    encoder = _make_encoder(model)

    result = encoder.encode_patches(Image.new("RGB", (16, 16)))

    expected = F.normalize(embeds[0, 1:] @ proj, dim=-1)  # CLS token (index 0) dropped
    assert result.shape == (4, 3)
    assert torch.allclose(result, expected, atol=1e-5)


def test_encode_patches_dispatches_correctly_by_trunk_presence():
    # A model exposing BOTH a `.trunk` (SigLIP marker) AND the plain-ViT
    # attributes, with each path's own math producing a clearly
    # DIFFERENT result -- confirms hasattr(visual, "trunk") actually
    # selects the attn_pool branch over the linear-proj branch when both
    # are technically present, not just that either branch runs without
    # error in isolation.
    feats = torch.tensor([[[1.0, 0.0, 0.0]]]).expand(1, 4, 3).clone()  # attn_pool path -> doubles this

    class _AmbiguousVisual(_FakeSigLIPVisual, _FakePlainVisual):
        def __init__(self):
            _FakeSigLIPVisual.__init__(self, feats)
            _FakePlainVisual.__init__(self, torch.ones(1, 5, 3), torch.eye(3) * 100.0)

    model = type("M", (), {"visual": _AmbiguousVisual()})()
    encoder = _make_encoder(model)

    result = encoder.encode_patches(Image.new("RGB", (16, 16)))

    # If dispatch incorrectly took the linear-proj branch, it would use
    # the all-ones `_embeds_out` / `proj*100` data instead -- still
    # normalized to unit length, but this equality check on the PRE-
    # normalization direction (via the attn_pool branch's own doubled
    # `feats` values) confirms which branch actually ran.
    expected = F.normalize(feats[0] * 2.0, dim=-1)
    assert torch.allclose(result, expected, atol=1e-5)
