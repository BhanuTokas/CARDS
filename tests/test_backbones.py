"""Tests for cards.models.backbones' registry structure -- dependency-free
where possible (importing the module doesn't trigger a weight download;
BackboneSpec.transforms() just builds a preprocessing pipeline object).
Loading the real pretrained weights via load_native()/feature_extractor()
is integration-only, verified separately, not here."""

from __future__ import annotations

from cards.models.backbones import BACKBONES, BackboneSpec


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


def test_all_registry_entries_have_callables():
    for name, spec in BACKBONES.items():
        assert callable(spec.load_native), f"{name}.load_native must be callable"
        assert callable(spec.preprocess), f"{name}.preprocess must be callable"
        assert callable(spec.feature_extractor), f"{name}.feature_extractor must be callable"
