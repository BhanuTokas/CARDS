"""Tests for cards.models.posthoc_cbm's dependency-free logic: resolving a
target class name to the checkpoint's output index, the CUB preprocessing
transform, and __call__'s target-class-column selection. Loading a real
checkpoint (needs post_hoc_cbm on sys.path + pytorchcv + the actual .ckpt
file) is integration-only -- verified via a real smoke run, not here."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from cards.models.posthoc_cbm import PosthocCBMBlackBox, cub_preprocess, resolve_target_index

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


# ---- PosthocCBMBlackBox.__call__ ----


def test_call_returns_target_class_logit_column():
    black_box = PosthocCBMBlackBox.__new__(PosthocCBMBlackBox)
    black_box.device = "cpu"
    black_box.target_index = 1
    black_box.backbone = lambda batch: batch  # already "flat" -> .flatten(1) is a no-op
    black_box.model = lambda emb: torch.tensor([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])

    result = black_box(torch.zeros(2, 4))

    assert torch.allclose(result, torch.tensor([20.0, 2.0]))
