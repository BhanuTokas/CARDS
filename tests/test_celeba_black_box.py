"""Tests for cards.models.celeba_black_box's dependency-free logic: task
name validation and __call__'s task-column selection. Loading the real
checkpoint (needs trained_models_new/celeba/resnet18_attractive_young.pt)
is integration-only -- verified via a real smoke run, not here (same
approach as test_posthoc_cbm.py)."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from cards.models.celeba_black_box import CelebaAttractiveYoungBlackBox

# ---- __init__ task_name validation (no checkpoint load needed -- the
# check runs before BACKBONES["celeba_attractive_young"] is touched) ----


def test_init_rejects_unknown_task_name():
    with pytest.raises(ValueError):
        CelebaAttractiveYoungBlackBox(task_name="Not_A_Real_Task", device="cpu")


# ---- __call__ ----


def test_call_returns_task_logit_column():
    black_box = CelebaAttractiveYoungBlackBox.__new__(CelebaAttractiveYoungBlackBox)
    black_box.device = "cpu"
    black_box.task_idx = 3  # Young's positive-class logit
    black_box.model = lambda batch: torch.tensor(
        [[1.0, 2.0, 3.0, 40.0], [5.0, 6.0, 7.0, 8.0]]
    )

    result = black_box(torch.zeros(2, 4))

    assert torch.allclose(result, torch.tensor([40.0, 8.0]))


# ---- preprocess ----


def test_preprocess_converts_to_rgb_before_delegating():
    black_box = CelebaAttractiveYoungBlackBox.__new__(CelebaAttractiveYoungBlackBox)
    seen = {}

    def fake_preprocess(image):
        seen["mode"] = image.mode
        return torch.zeros(3, 4, 4)

    black_box._preprocess = fake_preprocess

    black_box.preprocess(Image.new("L", (8, 8)))  # grayscale in, must convert to RGB

    assert seen["mode"] == "RGB"
