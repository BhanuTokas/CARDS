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
    scripts/cub_concept_bank_accuracies.py). Unpickling itself only needs
    torch/numpy/torchvision (already CARDS deps) to be importable through
    post_hoc_cbm's `models` package -- pytorchcv is only needed for the
    backbone this module loads itself, kept separate from post_hoc_cbm's
    own `models.model_zoo.get_model` on purpose so this doesn't depend on
    that function's exact signature.

Currently only the resnet18_cub backbone/preprocessing is implemented --
that's the only checkpoint verified in the CARDS investigation. A CLIP
RN50-backed checkpoint (e.g. the broden-concept-bank ones) would need its
own preprocess()/backbone branch.
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


def cub_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_CUB_MEAN, _CUB_STD),
        ]
    )


def resolve_target_index(idx_to_class: dict[int, str], target_class: str) -> int:
    """Concept-attribution target class name -> the checkpoint's output index."""
    class_to_idx = {name: idx for idx, name in idx_to_class.items()}
    if target_class not in class_to_idx:
        sample = ", ".join(sorted(class_to_idx)[:5])
        raise ValueError(f"target_class {target_class!r} not found; e.g. {sample}, ...")
    return class_to_idx[target_class]


def _load_resnet18_cub_backbone(device: str) -> torch.nn.Module:
    from pytorchcv.model_provider import get_model as ptcv_get_model

    full_model = ptcv_get_model("resnet18_cub", pretrained=True)
    # Drop the final FC layer, keep everything up to the pooled feature
    # vector -- matches post_hoc_cbm's ResNetBottom (models/model_zoo.py).
    backbone = torch.nn.Sequential(*list(full_model.children())[:-1])
    return backbone.to(device).eval()


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

        self.device = device
        self.model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.eval().to(device)

        if self.model.backbone_name != "resnet18_cub":
            raise NotImplementedError(
                f"PosthocCBMBlackBox only supports the resnet18_cub backbone so far, "
                f"got {self.model.backbone_name!r}"
            )

        self.backbone = _load_resnet18_cub_backbone(device)
        self._preprocess = cub_preprocess()
        self.target_index = resolve_target_index(self.model.idx_to_class, target_class)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image.convert("RGB"))

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device)
        embeddings = self.backbone(batch).flatten(1)
        logits = self.model(embeddings)
        return logits[:, self.target_index].detach().cpu()
