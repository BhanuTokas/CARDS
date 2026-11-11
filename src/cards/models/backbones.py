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
from torchvision.models import (
    ResNet18_Weights,
    convnext_tiny,
    resnet18,
    vit_b_16,
)

from cards.models.posthoc_cbm import cub_preprocess

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_CELEBA_CKPT = Path("trained_models_new/celeba/resnet18_attractive_young.pt")
_CELEBA_LOWRES_CKPT = Path("trained_models_new/celeba/resnet18_attractive_young_lowres.pt")
_CELEBA_MALE_CKPT = Path("trained_models_new/celeba/resnet18_attractive_young_male.pt")
_CELEBA_OFFICIAL_TRAIN_CKPT = Path("trained_models_new/celeba/resnet18_official_train_attractive_male.pt")
_CELEBA_OFFICIAL_TRAIN_VIT_CKPT = Path("trained_models_new/celeba/vit_b_16_official_train_attractive_male.pt")
_CELEBA_OFFICIAL_TRAIN_CONVNEXT_CKPT = Path("trained_models_new/celeba/convnext_tiny_official_train_attractive_male.pt")
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


class _ViTEmbeddingExtractor(nn.Module):
    """Replicates torchvision's `VisionTransformer.forward()` up to (and
    including) CLS-token extraction, dropping only the final `heads`
    classification layer. Confirmed directly against torchvision's own
    source (`_process_input` + `forward`) -- ViT's forward does custom
    patchify/reshape/class-token-prepend logic between `conv_proj` and
    `encoder` that `nn.Sequential(*children[:-1])` can't replicate by
    blindly chaining submodules (unlike ResNet's plain conv-stack-then-
    pool structure, where that trick works). Reuses the loaded native
    model's own submodules directly, not a second load."""

    def __init__(self, native_model: nn.Module):
        super().__init__()
        self.conv_proj = native_model.conv_proj
        self.class_token = native_model.class_token
        self.encoder = native_model.encoder
        self.hidden_dim = native_model.hidden_dim
        self.patch_size = native_model.patch_size
        self.image_size = native_model.image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _c, h, w = x.shape
        p = self.patch_size
        n_h, n_w = h // p, w // p
        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        return x[:, 0]


def _vit_feature_extractor(native_model: nn.Module) -> nn.Module:
    return _ViTEmbeddingExtractor(native_model)


def _convnext_feature_extractor(native_model: nn.Module) -> nn.Module:
    """Backbone-up-to-pooled-embedding, native `classifier[2]` (the final
    Linear) dropped -- confirmed directly against the real module tree
    (`features`, `avgpool`, `classifier=Sequential(LayerNorm2d, Flatten,
    Linear)`). Keeps `classifier[0]`/`classifier[1]` (LayerNorm2d,
    Flatten) in the extractor -- they're pretrained normalization/
    reshaping the rest of the network expects before its final layer,
    not part of the "head" the way the Linear is, so dropping the whole
    `classifier` Sequential (the ResNet-style "drop last child" trick)
    would be wrong here."""
    return nn.Sequential(native_model.features, native_model.avgpool,
                          native_model.classifier[0], native_model.classifier[1])


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


def _load_celeba_attractive_young_male_native() -> nn.Module:
    """Extends the original Phase 1 checkpoint with a 3rd task, Male,
    prompted directly ("Can we extend to add Male to our analysis?" ->
    "Extend existing checkpoint") -- `train_attractive_young_male_
    classifier.py`'s own fresh training run, a 3-task 6-way-logit head
    ([0:2]=Attractive, [2:4]=Young, [4:6]=Male), same architecture and
    the SAME train/val split as the original 2-task checkpoint (only
    Male's own labels are new). Saved to a DIFFERENT checkpoint file
    (resnet18_attractive_young_male.pt) -- the original 2-task
    checkpoint is untouched, so every historical result in this track
    stays reproducible against it."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)
    state = torch.load(_CELEBA_MALE_CKPT, map_location="cpu")
    model.load_state_dict(state)
    return model.eval()


def _load_celeba_official_train_native() -> nn.Module:
    """`train_official_celeba_classifier.py`'s own checkpoint -- trained
    on standard CelebA's OFFICIAL train partition (162,770 images),
    NOT CelebAMask-HQ's 25,500-image curated split every other CelebA
    checkpoint in this track used, prompted directly ("Can we train a
    classifier on the CelebA official train set and see if it holds the
    same pattern?"). 2-task 4-way-logit head ([0:2]=Attractive,
    [2:4]=Male -- Young dropped per direct instruction). Images are
    genuinely native low-resolution captures, so uses the SAME plain
    `_celeba_preprocess` as the original checkpoint, not the HQ-
    degradation trick `_celeba_lowres_preprocess` needed."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    state = torch.load(_CELEBA_OFFICIAL_TRAIN_CKPT, map_location="cpu")
    model.load_state_dict(state)
    return model.eval()


def _load_celeba_official_train_vit_native() -> nn.Module:
    """`train_official_celeba_classifier_vit.py`'s own checkpoint --
    SAME official-train data/2-task-4-way-logit-head convention as
    `_load_celeba_official_train_native`, ViT-B/16 backbone instead of
    ResNet18, prompted directly ("I want to do the main CelebA
    experiment with ViT and one other model as the black box model").
    `model.heads` (torchvision's own Sequential wrapper) is replaced
    wholesale with a plain Linear, matching the training script's own
    head-swap exactly."""
    model = vit_b_16(weights=None)
    model.heads = nn.Linear(model.hidden_dim, 4)
    state = torch.load(_CELEBA_OFFICIAL_TRAIN_VIT_CKPT, map_location="cpu")
    model.load_state_dict(state)
    return model.eval()


def _load_celeba_official_train_convnext_native() -> nn.Module:
    """`train_official_celeba_classifier_convnext.py`'s own checkpoint --
    SAME official-train data/2-task-4-way-logit-head convention,
    ConvNeXt-Tiny backbone. Only `classifier[2]` (the final Linear) is
    replaced, matching the training script's own head-swap exactly --
    see `_convnext_feature_extractor`'s docstring for why the whole
    `classifier` Sequential isn't swapped wholesale."""
    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 4)
    state = torch.load(_CELEBA_OFFICIAL_TRAIN_CONVNEXT_CKPT, map_location="cpu")
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
    "celeba_attractive_young_male": BackboneSpec(
        name="celeba_attractive_young_male",
        embed_dim=512,
        hook_layer="layer4",
        load_native=_load_celeba_attractive_young_male_native,
        preprocess=_celeba_preprocess(),
        feature_extractor=_resnet18_feature_extractor,
    ),
    "celeba_official_train_attractive_male": BackboneSpec(
        name="celeba_official_train_attractive_male",
        embed_dim=512,
        hook_layer="layer4",
        load_native=_load_celeba_official_train_native,
        preprocess=_celeba_preprocess(),
        feature_extractor=_resnet18_feature_extractor,
    ),
    # hook_layer confirmed directly against a real vit_b_16(weights=None)
    # instance's own named_children() -- last of 12 transformer blocks
    # (encoder.layers.encoder_layer_0..11), not assumed by analogy to
    # ResNet's "layer4".
    "celeba_official_train_attractive_male_vit": BackboneSpec(
        name="celeba_official_train_attractive_male_vit",
        embed_dim=768,
        hook_layer="encoder.layers.encoder_layer_11",
        load_native=_load_celeba_official_train_vit_native,
        preprocess=_celeba_preprocess(),
        feature_extractor=_vit_feature_extractor,
    ),
    # hook_layer confirmed directly against a real convnext_tiny(weights=
    # None) instance's own named_children() -- features.7 is the last of
    # 8 top-level stages (stem + 4 stage/downsample pairs collapse to 8
    # Sequential entries in torchvision's own indexing).
    "celeba_official_train_attractive_male_convnext": BackboneSpec(
        name="celeba_official_train_attractive_male_convnext",
        embed_dim=768,
        hook_layer="features.7",
        load_native=_load_celeba_official_train_convnext_native,
        preprocess=_celeba_preprocess(),
        feature_extractor=_convnext_feature_extractor,
    ),
}
