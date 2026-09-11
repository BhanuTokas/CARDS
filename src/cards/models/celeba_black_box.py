"""Adapter wrapping the fine-tuned CelebA Attractive/Young classifier
(cards.models.backbones.BACKBONES["celeba_attractive_young"]) as a CARDS
BlackBoxModel for Step 6/7 attribution scoring -- the same checkpoint the
CelebA track's own TCAV/PCBM comparisons already use (BackboneSpec.
load_native()), resolved down to one task's positive-class logit per the
BlackBoxModel contract (cards.models.base).

Only a Hydra-wireable adapter class was missing before this; the
checkpoint, its 2-task 4-way-logit head, and every score computed against
it (notes/celeba_correlation_investigation.md v65+) are unchanged.

Task -> logit index mirrors the checkpoint's own head
(scripts/celeba/build/train_attractive_young_classifier.py: [0:2] =
not-Attractive/Attractive, [2:4] = not-Young/Young), ported from
scripts/celeba/run/run_cards_celeba_full.py's own TASK_POSITIVE_LOGIT_INDEX
(kept in sync here rather than importing a scripts/ module from src/cards/).
"""

from __future__ import annotations

import torch
from PIL import Image

from cards.data.celeba_attributes import TARGET_CLASSES
from cards.models.backbones import BACKBONES

TASK_POSITIVE_LOGIT_INDEX: dict[str, int] = {"Attractive": 1, "Young": 3}


class CelebaAttractiveYoungBlackBox:
    """Wraps a celeba_attractive_young-shaped checkpoint as `b(x) ->
    scalar`, resolving its 2-task 4-way-logit head to one task's own
    positive-class logit -- the BlackBoxModel contract.

    `backbone_name` selects which BACKBONES entry to load -- defaults to
    the original HQ-resolution-trained checkpoint; pass
    "celeba_attractive_young_lowres" for the low-resolution-trained
    variant (notes/celeba_correlation_investigation.md's resolution-
    mismatch follow-up). Both share this same 2-task 4-way head shape,
    so no other change is needed here.
    """

    def __init__(self, task_name: str, device: str = "cuda", backbone_name: str = "celeba_attractive_young"):
        if task_name not in TASK_POSITIVE_LOGIT_INDEX:
            raise ValueError(
                f"task_name {task_name!r} not in {sorted(TASK_POSITIVE_LOGIT_INDEX)} "
                f"(cards.data.celeba_attributes.TARGET_CLASSES={TARGET_CLASSES})"
            )
        spec = BACKBONES[backbone_name]
        self.device = device
        self.model = spec.load_native().to(device).eval()
        self._preprocess = spec.preprocess
        self.task_idx = TASK_POSITIVE_LOGIT_INDEX[task_name]

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image.convert("RGB"))

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch.to(self.device))[:, self.task_idx].detach().cpu()
