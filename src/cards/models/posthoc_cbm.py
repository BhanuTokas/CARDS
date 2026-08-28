"""Adapter wrapping a trained Post-hoc CBM checkpoint
(github.com/mertyg/post-hoc-cbm) as a CARDS BlackBoxModel for Step 6/7
attribution scoring.

Two things this needs that aren't hard CARDS dependencies:
  - `pytorchcv`, to load the resnet18_cub backbone (`uv sync --extra pcbm`).
  - the post_hoc_cbm repo itself importable on sys.path -- its checkpoint
    .ckpt is a whole-object pickle referencing `models.pcbm_utils.
    PosthocLinearCBM` (cavs/intercepts/norms are plain attributes on that
    class, not registered buffers, so a state_dict alone can't reconstruct
    it -- confirmed while investigating the checkpoint, see
    scripts/cub/analysis/cub_concept_bank_accuracies.py). Unpickling itself only needs
    torch/numpy/torchvision (already CARDS deps) to be importable through
    post_hoc_cbm's `models` package -- pytorchcv is only needed for the
    backbone this module loads itself, kept separate from post_hoc_cbm's
    own `models.model_zoo.get_model` on purpose so this doesn't depend on
    that function's exact signature.

Two backbones are implemented: resnet18_cub (CUB) and plain resnet18 --
ImageNet-pretrained, used for the CIFAR-100 checkpoint (see
notes/ablation_scope_decision.md and docs/broden_label_corruption.md for
why: a CLIP backbone here would make the CARDS-vs-PCBM correlation
validation circular, since CARDS itself represents concepts via CLIP).
A CLIP RN50-backed checkpoint (e.g. the original unfiltered broden-concept
-bank ones) would need its own preprocess()/backbone branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# Exact preprocessing post_hoc_cbm used to train/eval this checkpoint
# (models/model_zoo.py's resnet18_cub branch) -- must match for the
# checkpoint's reported train/test accuracy to hold.
_CUB_MEAN = [0.5, 0.5, 0.5]
_CUB_STD = [2.0, 2.0, 2.0]

# Exact preprocessing post_hoc_cbm used for the plain (ImageNet-pretrained)
# resnet18 backbone (models/model_zoo.py's resnet18 branch) -- note this
# rescales via /255 on a pil_to_tensor uint8 tensor, not ToTensor's
# built-in /255, so it isn't just cub_preprocess with different constants.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def cub_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_CUB_MEAN, _CUB_STD),
        ]
    )


def resnet18_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            lambda x: x / 255.0,
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def resolve_target_index(idx_to_class: dict[int, str], target_class: str) -> int:
    """Concept-attribution target class name -> the checkpoint's output index."""
    class_to_idx = {name: idx for idx, name in idx_to_class.items()}
    if target_class not in class_to_idx:
        sample = ", ".join(sorted(class_to_idx)[:5])
        raise ValueError(f"target_class {target_class!r} not found; e.g. {sample}, ...")
    return class_to_idx[target_class]


def _load_pytorchcv_backbone(model_name: str, device: str) -> torch.nn.Module:
    from pytorchcv.model_provider import get_model as ptcv_get_model

    full_model = ptcv_get_model(model_name, pretrained=True)
    # Drop the final FC layer, keep everything up to the pooled feature
    # vector -- matches post_hoc_cbm's ResNetBottom (models/model_zoo.py).
    backbone = torch.nn.Sequential(*list(full_model.children())[:-1])
    return backbone.to(device).eval()


_BACKBONES = {
    "resnet18_cub": lambda device: _load_pytorchcv_backbone("resnet18_cub", device),
    "resnet18": lambda device: _load_pytorchcv_backbone("resnet18", device),
}
_PREPROCESSORS = {
    "resnet18_cub": cub_preprocess,
    "resnet18": resnet18_preprocess,
}


class PosthocCBMBlackBox:
    """Wraps a trained PosthocLinearCBM + its backbone as `b(x) -> scalar`,
    resolving the checkpoint's multi-class output to a single target
    class' logit -- the BlackBoxModel contract in cards.models.base.
    """

    def __init__(
        self,
        checkpoint_path: str,
        target_class: str,
        post_hoc_cbm_path: str,
        device: str = "cuda",
    ):
        repo_path = str(Path(post_hoc_cbm_path).resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        # weights_only=False is required -- PosthocLinearCBM's cavs/intercepts/
        # norms are plain attributes, not registered buffers, so a state_dict
        # alone can't reconstruct it (see the module docstring). This runs
        # arbitrary code embedded in the pickle: only point checkpoint_path at
        # a checkpoint you trust (e.g. one you or a known collaborator
        # produced), never at an untrusted or externally-supplied file.
        self.device = device
        self.model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.eval().to(device)

        backbone_name = self.model.backbone_name
        if backbone_name not in _BACKBONES:
            raise NotImplementedError(
                f"PosthocCBMBlackBox only supports {sorted(_BACKBONES)} backbones so far, "
                f"got {backbone_name!r}"
            )

        self.backbone = _BACKBONES[backbone_name](device)
        self._preprocess = _PREPROCESSORS[backbone_name]()
        self.target_index = resolve_target_index(self.model.idx_to_class, target_class)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image.convert("RGB"))

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device)
        embeddings = self.backbone(batch).flatten(1)
        logits = self.model(embeddings)
        return logits[:, self.target_index].detach().cpu()
