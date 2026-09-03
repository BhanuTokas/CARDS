"""Swappable pretrained-backbone registry for the CARDS vs. TCAV vs. PCBM
ImageNet comparison (notes/pcbm_correlation_investigation.md, v25+).

Distinct from `cards.models.posthoc_cbm`'s pytorchcv-based `resnet18`/
`resnet18_cub`/`resnet18_lowres` branches, which back the existing CUB/
CIFAR-100 checkpoints and stay untouched -- this module is the new,
torchvision-based backbone source for the ImageNet track specifically,
chosen for its single consistent `weights=...IMAGENET1K_V1` API across
CNN and Transformer families alike (needed once a ViT entry is added; see
the parent plan's Phase 6).

Per this investigation's surrogate-modeling decision: the "black box" all
three comparison methods (CARDS, TCAV, PCBM) explain is the *native*
off-the-shelf model's own real classification head, not a separately
fit one -- so `load_native()` returns the complete, unmodified pretrained
model. `feature_extractor()` derives PCBM's own backbone (native head
dropped) from that same loaded instance, so PCBM's concept-bottleneck
input and TCAV's hooked activations both trace back to identical,
frozen weights -- never two separately-loaded copies that could drift.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from cards.models.posthoc_cbm import cub_preprocess

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_CELEBA_CKPT = Path("trained_models_new/celeba/resnet18_attractive_young.pt")
_CELEBA_LOWRES_CKPT = Path("trained_models_new/celeba/resnet18_attractive_young_lowres.pt")
# Standard (non-HQ) CelebA's own native img_align_celeba resolution
# (width, height) -- the degrade target for the low-res variant below.
_STANDARD_CELEBA_SIZE = (178, 218)


@dataclass
class BackboneSpec:
    """One entry in `BACKBONES`. `hook_layer` is the dotted path (relative
    to the module returned by `load_native()`) TCAV should hook -- the
    second-to-last nonlinear/learnable layer before the native
    classification head, per the parent plan's generalized convention
    (the final pooling step has no learnable transformation, so the last
    *meaningful* layer is one before it)."""

    name: str
    embed_dim: int
    hook_layer: str
    load_native: Callable[[], nn.Module]
    preprocess: Callable[[Image.Image], torch.Tensor]
    feature_extractor: Callable[[nn.Module], nn.Module]


def _load_resnet18_native() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    return model.eval()


def _resnet18_feature_extractor(native_model: nn.Module) -> nn.Module:
    """Backbone-up-to-pooled-embedding, native `fc` head dropped -- for
    PCBM's own concept-bottleneck input. Reuses the exact same weights
    `load_native()` returned (not a second load), so PCBM's and TCAV's
    view of the backbone are guaranteed identical."""
    return nn.Sequential(*list(native_model.children())[:-1])


def _load_celeba_attractive_young_native() -> nn.Module:
    """Phase 1's own fine-tuned checkpoint
    (scripts/celeba/build/train_attractive_young_classifier.py), not stock
    ImageNet weights -- a 2-task 4-way-logit head ([0:2]=Attractive,
    [2:4]=Young), same torchvision resnet18 architecture as `resnet18`
    above, confirmed directly via named_children() to still expose
    `layer4`/`fc` under those same names once the checkpoint is loaded."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    state = torch.load(_CELEBA_CKPT, map_location="cpu")
    model.load_state_dict(state)
    return model.eval()


def _celeba_preprocess() -> transforms.Compose:
    """Matches train_attractive_young_classifier.py's own val_transform
    exactly (Resize to 224, no augmentation) -- ImageNet mean/std since
    the backbone started from ImageNet-pretrained init and was never
    renormalized during fine-tuning."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def _load_celeba_attractive_young_lowres_native() -> nn.Module:
    """train_attractive_young_classifier_lowres.py's own checkpoint --
    identical architecture/task-head shape to
    _load_celeba_attractive_young_native, trained instead on images first
    degraded to standard CelebA's own native resolution (notes/
    celeba_correlation_investigation.md's resolution-mismatch follow-up:
    the original checkpoint only ever saw CelebA-HQ's re-derived
    1024x1024 crops during training, a real train/eval mismatch when
    evaluated on genuinely low-resolution images)."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    state = torch.load(_CELEBA_LOWRES_CKPT, map_location="cpu")
    model.load_state_dict(state)
    return model.eval()


def _celeba_lowres_preprocess() -> transforms.Compose:
    """Degrades to standard CelebA's own native resolution BEFORE the
    final 224x224 resize -- for a CelebA-HQ 1024x1024 source image, this
    step permanently destroys the fine detail HQ's own super-resolution
    added back, simulating what a genuinely low-resolution capture would
    look like; for an already-low-res source image (standard CelebA's
    own img_align_celeba, or anything similarly sized), the resize is
    close to a no-op, falling through to the same final 224x224 pipeline
    as _celeba_preprocess -- one consistent function, correct either way,
    not two separate preprocessing paths to keep in sync."""
    width, height = _STANDARD_CELEBA_SIZE
    return transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def _load_resnet18_cub_native() -> nn.Module:
    from pytorchcv.model_provider import get_model as ptcv_get_model

    model = ptcv_get_model("resnet18_cub", pretrained=True)
    return model.eval()


def _resnet18_cub_feature_extractor(native_model: nn.Module) -> nn.Module:
    """`features` submodule only (pytorchcv's own `init_block`..`final_pool`
    stack, native `output` head dropped) -- confirmed via named_children()
    that resnet18_cub's top level is exactly `{features, output}`, and
    `hook_layer="features.stage4"` resolves on this same instance."""
    return native_model.features


BACKBONES: dict[str, BackboneSpec] = {
    "resnet18": BackboneSpec(
        name="resnet18",
        embed_dim=512,
        hook_layer="layer4",
        load_native=_load_resnet18_native,
        preprocess=ResNet18_Weights.IMAGENET1K_V1.transforms(),
        feature_extractor=_resnet18_feature_extractor,
    ),
    "resnet18_cub": BackboneSpec(
        name="resnet18_cub",
        embed_dim=512,
        hook_layer="features.stage4",
        load_native=_load_resnet18_cub_native,
        preprocess=cub_preprocess(),
        feature_extractor=_resnet18_cub_feature_extractor,
    ),
    "celeba_attractive_young": BackboneSpec(
        name="celeba_attractive_young",
        embed_dim=512,
        hook_layer="layer4",
        load_native=_load_celeba_attractive_young_native,
        preprocess=_celeba_preprocess(),
        feature_extractor=_resnet18_feature_extractor,
    ),
    "celeba_attractive_young_lowres": BackboneSpec(
        name="celeba_attractive_young_lowres",
        embed_dim=512,
        hook_layer="layer4",
        load_native=_load_celeba_attractive_young_lowres_native,
        preprocess=_celeba_lowres_preprocess(),
        feature_extractor=_resnet18_feature_extractor,
    ),
}
