"""Step 1 tests: prompt ensembling with a deterministic fake TextEncoder
(no real model needed — encode_text is the only surface build_concept_query
depends on)."""

from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

from cards.concepts.prompts import DEFAULT_TEMPLATES, build_concept_bank, build_concept_query


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
