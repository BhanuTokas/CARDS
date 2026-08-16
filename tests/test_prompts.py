"""Step 1 tests: prompt ensembling with a deterministic fake TextEncoder
(no real model needed — encode_text is the only surface build_concept_query
depends on)."""

from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

import pytest

from cards.concepts.prompts import (
    DEFAULT_TEMPLATES,
    build_concept_bank,
    build_concept_query,
    compute_text_center,
    demean_query,
)


class LookupTextEncoder:
    """Each distinct prompt string maps to a fixed, repeatable unit vector."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self._cache: dict[str, torch.Tensor] = {}

    def _embed_one(self, text: str) -> torch.Tensor:
        if text not in self._cache:
            seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**31)
            generator = torch.Generator().manual_seed(seed)
            vector = torch.randn(self.dim, generator=generator)
            self._cache[text] = F.normalize(vector, dim=0)
        return self._cache[text]

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        return torch.stack([self._embed_one(t) for t in texts])


def test_build_concept_query_matches_manual_average():
    encoder = LookupTextEncoder()
    templates = ["a photo of {concept}", "an image showing {concept}"]

    result = build_concept_query("dog", encoder, templates)

    prompts = [t.format(concept="dog") for t in templates]
    expected = F.normalize(encoder.encode_text(prompts).mean(dim=0), dim=0)
    assert torch.allclose(result, expected)


def test_build_concept_query_is_unit_norm():
    encoder = LookupTextEncoder()
    result = build_concept_query("cat", encoder)
    assert torch.isclose(result.norm(), torch.tensor(1.0), atol=1e-5)


def test_build_concept_bank_matches_individual_queries():
    encoder = LookupTextEncoder()
    concepts = ["dog", "cat", "car"]

    bank = build_concept_bank(concepts, encoder)

    assert set(bank.keys()) == set(concepts)
    for concept in concepts:
        assert torch.allclose(bank[concept], build_concept_query(concept, encoder))


def test_default_templates_use_concept_placeholder():
    for template in DEFAULT_TEMPLATES:
        assert "{concept}" in template


# ---- compute_text_center ----


def test_compute_text_center_matches_manual_mean():
    encoder = LookupTextEncoder()
    concepts = [f"concept_{i}" for i in range(12)]

    center = compute_text_center(concepts, encoder)

    expected = torch.stack([build_concept_query(c, encoder) for c in concepts]).mean(dim=0)
    assert torch.allclose(center, expected)


def test_compute_text_center_not_renormalized():
    # a mean of unit vectors pointing in varied directions has norm < 1 in
    # general -- confirms this is left as an offset, not forced back onto
    # the unit sphere like build_concept_query's result is.
    encoder = LookupTextEncoder()
    concepts = [f"concept_{i}" for i in range(12)]

    center = compute_text_center(concepts, encoder)

    assert center.norm().item() < 1.0 - 1e-4


def test_compute_text_center_rejects_too_small_reference_set():
    encoder = LookupTextEncoder()

    with pytest.raises(ValueError):
        compute_text_center(["dog", "cat"], encoder)


# ---- demean_query ----


def test_demean_query_subtracts_center():
    t_c = torch.tensor([1.0, 2.0, 3.0])
    text_center = torch.tensor([0.5, 0.5, 0.5])

    result = demean_query(t_c, text_center)

    assert torch.allclose(result, torch.tensor([0.5, 1.5, 2.5]))


def test_demean_query_not_renormalized():
    t_c = torch.tensor([1.0, 0.0, 0.0])
    text_center = torch.tensor([0.9, 0.0, 0.0])

    result = demean_query(t_c, text_center)

    assert torch.allclose(result, torch.tensor([0.1, 0.0, 0.0]))
    assert not torch.isclose(result.norm(), torch.tensor(1.0), atol=1e-3)
