"""Protocol for pluggable black-box models used in Step 6/7 attribution
scoring (cards.pipeline.run). Any object satisfying this can be wired in
via a Hydra `_target_` config under configs/model/ -- see
configs/model/none.yaml for the "no model configured" default. The first
real adapter (a Post-hoc CBM, see ../post_hoc_cbm) is deferred until a
trained checkpoint exists to test it against.
"""

from __future__ import annotations

from typing import Protocol

import torch
from PIL import Image


class BlackBoxModel(Protocol):
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL image -> a single preprocessed input tensor (no batch dim).
        Independent of the CLIP preprocessing used for retrieval -- the
        black-box model may use an entirely different backbone."""
        ...

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """Batched preprocessed inputs -> a 1D tensor of per-image scalar
        scores (e.g. a specific target class' logit). Resolving a
        multi-class output down to a scalar is the adapter's job, not
        cards.pipeline's."""
        ...
