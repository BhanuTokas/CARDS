"""Step 5 — Multi-concept orthogonalization.

Symmetric (Lowdin) orthogonalization over a set of concept directions,
chosen over Gram-Schmidt to avoid order-dependence.
"""

from __future__ import annotations

import torch

from cards.directions.estimate import ConceptDirection

# Below this, a direction is treated as (near-)linearly dependent on the rest
# of the set rather than a real, separate concept axis. Eigenvalues of a Gram
# matrix of unit vectors live in a predictable, well-conditioned range, so
# this mainly guards against near-duplicate concept directions.
_MIN_GRAM_EIGENVALUE = 1e-8


def lowdin_orthogonalize(directions: list[ConceptDirection]) -> list[ConceptDirection]:
    """Symmetric orthogonalization: V_orth = V @ S^(-1/2), where S = V^T V is
    the Gram matrix of the stacked unit vectors (columns of V) and S^(-1/2)
    is computed via eigendecomposition (S is symmetric PSD). Unlike
    Gram-Schmidt, the result doesn't depend on the order of `directions`.
    Each concept's `magnitude` (Delta_c from Step 4) is carried over
    unchanged -- only the unit vectors are touched here.

    A single direction (or none) is returned unchanged: orthogonalizing one
    vector against nothing is a no-op (a 1x1 Gram matrix is always [1]), so
    callers processing variable-size concept subsets don't need to special-case
    small sets themselves.
    """
    if len(directions) == 0:
        return []

    vectors = torch.stack([d.unit_vector for d in directions], dim=1)  # (dim, k)
    gram = vectors.T @ vectors  # (k, k)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    if eigenvalues.min().item() <= _MIN_GRAM_EIGENVALUE:
        raise ValueError(
            "directions are (near-)linearly dependent; Lowdin orthogonalization "
            "requires a positive-definite Gram matrix"
        )
    inv_sqrt_gram = eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.T
    orthogonalized = vectors @ inv_sqrt_gram  # (dim, k), columns now orthonormal

    return [
        ConceptDirection(concept=d.concept, unit_vector=orthogonalized[:, i], magnitude=d.magnitude)
        for i, d in enumerate(directions)
    ]
