"""Step 5 — Multi-concept orthogonalization.

Symmetric (Lowdin) orthogonalization over a set of concept directions,
chosen over Gram-Schmidt to avoid order-dependence.
"""

from __future__ import annotations

import torch

from cards.directions.estimate import ConceptDirection


def lowdin_orthogonalize(directions: list[ConceptDirection]) -> list[ConceptDirection]:
    """Symmetric orthogonalization: D_orth = D @ (D^T D)^(-1/2), applied to
    the stacked unit vectors. Order-independent, unlike Gram-Schmidt."""
    raise NotImplementedError
