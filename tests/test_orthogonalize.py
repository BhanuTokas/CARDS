"""Step 5 tests: symmetric (Lowdin) orthogonalization."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cards.directions.estimate import ConceptDirection
from cards.directions.orthogonalize import lowdin_orthogonalize


def _make_direction(concept: str, vector: torch.Tensor, magnitude: float = 1.0) -> ConceptDirection:
    return ConceptDirection(concept=concept, unit_vector=F.normalize(vector, dim=0), magnitude=magnitude)


def test_lowdin_orthogonalize_produces_orthonormal_set():
    generator = torch.Generator().manual_seed(0)
    directions = [
        _make_direction("a", torch.randn(5, generator=generator)),
        _make_direction("b", torch.randn(5, generator=generator)),
        _make_direction("c", torch.randn(5, generator=generator)),
    ]

    result = lowdin_orthogonalize(directions)

    vectors = torch.stack([d.unit_vector for d in result])
    gram = vectors @ vectors.T
    assert torch.allclose(gram, torch.eye(3), atol=1e-5)


def test_lowdin_orthogonalize_preserves_magnitude_and_concept_order():
    generator = torch.Generator().manual_seed(1)
    directions = [
        _make_direction("a", torch.randn(4, generator=generator), magnitude=1.5),
        _make_direction("b", torch.randn(4, generator=generator), magnitude=2.5),
    ]

    result = lowdin_orthogonalize(directions)

    assert [d.concept for d in result] == ["a", "b"]
    assert [d.magnitude for d in result] == [1.5, 2.5]


def test_lowdin_orthogonalize_is_order_independent():
    generator = torch.Generator().manual_seed(2)
    a = _make_direction("a", torch.randn(5, generator=generator))
    b = _make_direction("b", torch.randn(5, generator=generator))
    c = _make_direction("c", torch.randn(5, generator=generator))

    result_abc = {d.concept: d.unit_vector for d in lowdin_orthogonalize([a, b, c])}
    result_cab = {d.concept: d.unit_vector for d in lowdin_orthogonalize([c, a, b])}

    for concept in ("a", "b", "c"):
        assert torch.allclose(result_abc[concept], result_cab[concept], atol=1e-5)


def test_lowdin_orthogonalize_single_direction_is_noop():
    direction = _make_direction("a", torch.randn(4), magnitude=3.0)

    result = lowdin_orthogonalize([direction])

    assert len(result) == 1
    assert result[0].concept == "a"
    assert result[0].magnitude == 3.0
    assert torch.allclose(result[0].unit_vector, direction.unit_vector, atol=1e-5)


def test_lowdin_orthogonalize_empty_list_returns_empty():
    assert lowdin_orthogonalize([]) == []


def test_lowdin_orthogonalize_rejects_linearly_dependent_directions():
    vector = torch.randn(4)
    a = _make_direction("a", vector.clone())
    b = _make_direction("b", vector.clone())  # identical direction -> singular Gram matrix

    with pytest.raises(ValueError):
        lowdin_orthogonalize([a, b])
