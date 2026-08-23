"""Tests for cards.models.posthoc_cbm's dependency-free logic: resolving a
target class name to the checkpoint's output index, the CUB preprocessing
transform, and __call__'s target-class-column selection. Loading a real
checkpoint (needs post_hoc_cbm on sys.path + pytorchcv + the actual .ckpt
file) is integration-only -- verified via a real smoke run, not here."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from cards.models.posthoc_cbm import (
    PosthocCBMBlackBox,
    cub_preprocess,
    resnet18_preprocess,
    resolve_target_index,
)

# ---- resolve_target_index ----


def test_resolve_target_index_finds_correct_index():
    idx_to_class = {0: "Black_footed_Albatross", 1: "Laysan_Albatross", 2: "Scarlet_Tanager"}

    assert resolve_target_index(idx_to_class, "Scarlet_Tanager") == 2
    assert resolve_target_index(idx_to_class, "Black_footed_Albatross") == 0


def test_resolve_target_index_raises_on_unknown_class():
    idx_to_class = {0: "Black_footed_Albatross", 1: "Laysan_Albatross"}

    with pytest.raises(ValueError):
        resolve_target_index(idx_to_class, "Not_A_Real_Bird")


# ---- cub_preprocess ----


def test_cub_preprocess_center_crops_to_224():
    image = Image.new("RGB", (500, 300), color=(120, 60, 200))

    tensor = cub_preprocess()(image)

    assert tensor.shape == (3, 224, 224)


def test_cub_preprocess_normalizes_with_cub_mean_std():
    # a uniform mid-gray image: pixel value 0.5 everywhere pre-normalize,
    # (0.5 - 0.5) / 2.0 = 0.0 after normalize
    image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    tensor = cub_preprocess()(image)

    assert tensor.abs().max().item() < 0.02  # ~0, allowing for 128/255 rounding


# ---- resnet18_preprocess ----


def test_resnet18_preprocess_center_crops_to_224():
    image = Image.new("RGB", (500, 300), color=(120, 60, 200))

    tensor = resnet18_preprocess()(image)

    assert tensor.shape == (3, 224, 224)


def test_resnet18_preprocess_normalizes_with_imagenet_mean_std():
    image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    tensor = resnet18_preprocess()(image)

    pixel = 128 / 255
    expected_r = (pixel - 0.485) / 0.229
    expected_g = (pixel - 0.456) / 0.224
    expected_b = (pixel - 0.406) / 0.225
    assert tensor[0, 0, 0].item() == pytest.approx(expected_r, abs=1e-4)
    assert tensor[1, 0, 0].item() == pytest.approx(expected_g, abs=1e-4)
    assert tensor[2, 0, 0].item() == pytest.approx(expected_b, abs=1e-4)


# ---- PosthocCBMBlackBox backbone dispatch ----


class _FakePcbmModel:
    def __init__(self, backbone_name):
        self.backbone_name = backbone_name
        self.idx_to_class = {0: "a", 1: "b"}

    def eval(self):
        return self

    def to(self, device):
        return self


def test_init_rejects_unsupported_backbone(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "cards.models.posthoc_cbm.torch.load", lambda *a, **k: _FakePcbmModel("clip:RN50")
    )

    with pytest.raises(NotImplementedError):
        PosthocCBMBlackBox(
            checkpoint_path=str(tmp_path / "fake.ckpt"),
            target_class="a",
            post_hoc_cbm_path=str(tmp_path),
            device="cpu",
        )


# ---- PosthocCBMBlackBox.__call__ ----


def test_call_returns_target_class_logit_column():
    black_box = PosthocCBMBlackBox.__new__(PosthocCBMBlackBox)
    black_box.device = "cpu"
    black_box.target_index = 1
    black_box.backbone = lambda batch: batch  # already "flat" -> .flatten(1) is a no-op
    black_box.model = lambda emb: torch.tensor([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])

    result = black_box(torch.zeros(2, 4))

    assert torch.allclose(result, torch.tensor([20.0, 2.0]))
