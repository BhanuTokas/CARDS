"""Tests for cards.encoders.perception_encoder's dependency-free logic:
batching/concatenation in encode_text/encode_images, encode_patches'
linear-projection math, and embed_dim's lazy-probe caching. Constructing
a real PerceptionEncoder needs facebookresearch/perception_models on
sys.path plus a real PE-Core checkpoint download -- integration-only,
verified via a real smoke run, not here (same approach as
test_posthoc_cbm.py)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image

from cards.encoders.perception_encoder import PerceptionEncoder


class _FakeTokens:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def to(self, device):
        return self


class _FakeTokenizer:
    def __call__(self, texts: list[str]):
        return _FakeTokens(len(texts))


class _FakeVisual:
    """PE-Core's own public forward_features(norm=True, strip_cls_token=
    True) helper + a linear `.proj` -- encode_patches' own dispatch
    target, bypassing PE's real attention-pool head entirely (same
    linear-bypass logic as OpenClipEncoder's plain-ViT path)."""

    def __init__(self, patch_tokens: torch.Tensor, proj: torch.Tensor):
        self._patch_tokens = patch_tokens  # (1, N, C), already CLS-stripped
        self.proj = proj

    def forward_features(self, pixels: torch.Tensor, norm: bool = False, strip_cls_token: bool = False):
        assert norm and strip_cls_token
        return self._patch_tokens


class _FakePEModel:
    """Assigns each text/image batch a distinct, size-encoded feature
    block, so tests can verify batch order/concatenation without a real
    model -- mirrors the _CountingImageEncoder pattern used elsewhere."""

    def __init__(self, visual: _FakeVisual | None = None):
        self.visual = visual

    def encode_text(self, tokens: _FakeTokens, normalize: bool = False):
        return torch.full((tokens.batch_size, 4), float(tokens.batch_size))

    def encode_image(self, pixels: torch.Tensor, normalize: bool = False):
        return torch.full((pixels.shape[0], 4), float(pixels.shape[0]))


def _make_encoder(batch_size: int = 2, visual: _FakeVisual | None = None) -> PerceptionEncoder:
    encoder = PerceptionEncoder.__new__(PerceptionEncoder)
    encoder.device = "cpu"
    encoder.batch_size = batch_size
    encoder.model = _FakePEModel(visual)
    encoder.tokenizer = _FakeTokenizer()
    encoder.preprocess = lambda img: torch.zeros(3, 4, 4)
    encoder._embed_dim = None
    return encoder


# ---- encode_text ----


def test_encode_text_batches_and_concatenates():
    encoder = _make_encoder(batch_size=2)

    result = encoder.encode_text(["a", "b", "c", "d", "e"])

    # 5 texts, batch_size=2 -> chunks of 2, 2, 1
    assert result.shape == (5, 4)
    assert result[:2].unique().item() == 2.0
    assert result[2:4].unique().item() == 2.0
    assert result[4:5].unique().item() == 1.0


def test_encode_text_single_batch():
    encoder = _make_encoder(batch_size=10)

    result = encoder.encode_text(["a", "b", "c"])

    assert result.shape == (3, 4)
    assert (result == 3.0).all()


# ---- encode_images ----


def test_encode_images_batches_and_concatenates():
    encoder = _make_encoder(batch_size=2)
    images = [Image.new("RGB", (8, 8)) for _ in range(5)]

    result = encoder.encode_images(images)

    assert result.shape == (5, 4)
    assert result[:2].unique().item() == 2.0
    assert result[4:5].unique().item() == 1.0


# ---- encode_patches ----


def test_encode_patches_applies_linear_projection():
    patch_tokens = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)  # 4 patches, dim 3
    proj = torch.eye(3) * 2.0
    encoder = _make_encoder(visual=_FakeVisual(patch_tokens, proj))

    result = encoder.encode_patches(Image.new("RGB", (16, 16)))

    expected = F.normalize(patch_tokens[0] @ proj, dim=-1)
    assert result.shape == (4, 3)
    assert torch.allclose(result, expected, atol=1e-5)


def test_encode_patches_calls_forward_features_with_norm_and_strip_cls():
    # _FakeVisual.forward_features itself asserts norm=True,
    # strip_cls_token=True are passed -- a call with the wrong kwargs
    # would raise there instead of silently using unnormalized/CLS-
    # included tokens.
    patch_tokens = torch.ones(1, 2, 2)
    encoder = _make_encoder(visual=_FakeVisual(patch_tokens, torch.eye(2)))

    encoder.encode_patches(Image.new("RGB", (16, 16)))  # no AssertionError == kwargs were correct


# ---- embed_dim ----


def test_embed_dim_probes_once_and_caches():
    encoder = _make_encoder(batch_size=8)
    calls = {"count": 0}
    original_encode_images = encoder.encode_images

    def counting_encode_images(images):
        calls["count"] += 1
        return original_encode_images(images)

    encoder.encode_images = counting_encode_images

    first = encoder.embed_dim
    second = encoder.embed_dim

    assert first == second == 4
    assert calls["count"] == 1
