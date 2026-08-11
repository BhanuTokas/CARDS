"""Step 1 — Concept prompt construction.

Ensemble multiple template phrasings per concept (reusing the k-vs-1
prompt-ensembling construction from CounterConcept), average the resulting
text embeddings to get a single query vector `t_c` per concept.
"""

from __future__ import annotations

import torch

from cards.encoders.base import TextEncoder

DEFAULT_TEMPLATES: list[str] = [
    "a photo of {concept}",
    "a photo containing {concept}",
    "an image showing {concept}",
]


def build_concept_query(
    concept: str,
    encoder: TextEncoder,
    templates: list[str] = DEFAULT_TEMPLATES,
) -> torch.Tensor:
    """Embed `concept` under each template and average to a single query vector `t_c`."""
    raise NotImplementedError


def build_concept_bank(
    concepts: list[str],
    encoder: TextEncoder,
    templates: list[str] = DEFAULT_TEMPLATES,
) -> dict[str, torch.Tensor]:
    """Build `{concept: t_c}` for a full concept bank."""
    raise NotImplementedError
