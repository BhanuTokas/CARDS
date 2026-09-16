"""Tests for cards.models.backbones' registry structure -- dependency-free
where possible (importing the module doesn't trigger a weight download;
BackboneSpec.transforms() just builds a preprocessing pipeline object).
Loading the real pretrained weights via load_native()/feature_extractor()
is integration-only, verified separately, not here."""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from cards.models.backbones import BACKBONES, BackboneSpec

_IMAGENET_NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def test_registry_has_resnet18_entry():
    assert "resnet18" in BACKBONES
    spec = BACKBONES["resnet18"]
    assert isinstance(spec, BackboneSpec)
    assert spec.name == "resnet18"
    assert spec.embed_dim == 512


def test_resnet18_hook_layer_is_last_nonlinear_block():
    # layer4 is the last learnable/nonlinear block before avgpool (which
    # has no learnable transformation) -- see the module docstring's
    # "second-to-last meaningful layer" convention.
    assert BACKBONES["resnet18"].hook_layer == "layer4"


def test_registry_has_resnet18_cub_entry():
    assert "resnet18_cub" in BACKBONES
    spec = BACKBONES["resnet18_cub"]
    assert spec.embed_dim == 512
    # confirmed directly against a loaded checkpoint's named_children():
    # top level is exactly {features, output}, features.stage4 exists.
    assert spec.hook_layer == "features.stage4"


def test_all_registry_entries_have_callables():
    for name, spec in BACKBONES.items():
        assert callable(spec.load_native), f"{name}.load_native must be callable"
        assert callable(spec.preprocess), f"{name}.preprocess must be callable"
        assert callable(spec.feature_extractor), f"{name}.feature_extractor must be callable"


def test_registry_has_celeba_attractive_young_lowres_entry():
    assert "celeba_attractive_young_lowres" in BACKBONES
    spec = BACKBONES["celeba_attractive_young_lowres"]
    assert spec.embed_dim == 512
    assert spec.hook_layer == "layer4"


def _random_rgb(size: tuple[int, int]) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))


def test_celeba_lowres_preprocess_destroys_fine_detail_on_a_high_res_source():
    # A CelebA-HQ-sized (1024x1024) source with per-pixel random noise --
    # any fine detail surviving straight to a direct 224x224 resize
    # would NOT survive an intermediate downsample to standard CelebA's
    # own 178x218 native resolution first. The two paths must diverge,
    # proving the degrade step is real, not a no-op.
    lowres_preprocess = BACKBONES["celeba_attractive_young_lowres"].preprocess
    direct_preprocess = T.Compose([T.Resize((224, 224)), T.ToTensor(), _IMAGENET_NORM])

    image = _random_rgb((1024, 1024))
    via_degrade = lowres_preprocess(image)
    via_direct = direct_preprocess(image)

    assert via_degrade.shape == (3, 224, 224)
    assert not torch.allclose(via_degrade, via_direct, atol=1e-3)


def test_celeba_lowres_preprocess_is_close_to_a_noop_on_an_already_lowres_source():
    # Standard CelebA's own native size (178x218) -- the intermediate
    # resize step should barely change anything (resizing 178x218 to
    # 178x218 is exact; PIL's default bilinear resampling for the
    # subsequent identical-size call is effectively lossless), so this
    # should land very close to a direct 224x224 resize.
    lowres_preprocess = BACKBONES["celeba_attractive_young_lowres"].preprocess
    direct_preprocess = T.Compose([T.Resize((224, 224)), T.ToTensor(), _IMAGENET_NORM])

    image = _random_rgb((178, 218))
    via_degrade = lowres_preprocess(image)
    via_direct = direct_preprocess(image)

    assert torch.allclose(via_degrade, via_direct, atol=1e-3)
