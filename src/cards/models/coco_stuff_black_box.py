"""BlackBoxModel adapter for the COCO-Stuff ResNet18 classifier.

Mirrors cards.models.celeba_black_box exactly: wraps the trained 20-class
checkpoint (scripts/coco_stuff/build/train_coco_stuff_classifier.py output)
as a CARDS BlackBoxModel that resolves its 20-way logit head to one class's
positive logit per the BlackBoxModel contract (cards.models.base).

`task_name` selects which of the 20 COCO-Stuff classes this run explains;
override per attribution run, e.g. model.task_name="dining table".
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from cards.data.coco_stuff import COCO_STUFF_20_CLASSES

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_coco_stuff_model(checkpoint: str | Path) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(COCO_STUFF_20_CLASSES))
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    return model.eval()


def _coco_stuff_preprocess() -> transforms.Compose:
    """Matches train_coco_stuff_classifier.py's val_transform exactly."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class CocoStuffBlackBox:
    """Wraps the coco_stuff checkpoint as `b(x) -> scalar`, resolving the
    20-class logit head to one class's own positive logit."""

    def __init__(self, task_name: str, checkpoint: str, device: str = "cuda"):
        if task_name not in COCO_STUFF_20_CLASSES:
            raise ValueError(
                f"task_name {task_name!r} not in COCO_STUFF_20_CLASSES "
                f"(valid: {COCO_STUFF_20_CLASSES})"
            )
        self.device = device
        self.model = _build_coco_stuff_model(checkpoint).to(device)
        self._preprocess = _coco_stuff_preprocess()
        self.task_idx = COCO_STUFF_20_CLASSES.index(task_name)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image.convert("RGB"))

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        logits = self.model(batch.to(self.device))  # (N, 20)
        return logits[:, self.task_idx].detach().cpu()  # (N,)
