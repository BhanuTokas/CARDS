"""Tests for cards.pipeline's helper functions. run() itself (the full
Hydra-wired orchestration) isn't unit tested here -- it needs a real
encoder/dataset/model; these tests cover the pure dispatch/computation
logic it's built from."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS
from cards.directions.estimate import ConceptDirection
from cards.pipeline import (
    ConceptResult,
    compute_delta_c,
    instantiate_encoder,
    instantiate_model,
    load_and_preprocess,
    load_dataset_pool,
    normalize_score,
    process_concept,
    resolve_demean_reference_concepts,
    retrieve_concept_sets,
    save_directions,
)
from cards.retrieval.pool import CandidatePool


class _LookupTextEncoder:
    """Each distinct prompt string maps to a fixed, repeatable unit vector."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self._cache: dict[str, torch.Tensor] = {}

    def encode_text(self, texts):
        vectors = []
        for text in texts:
            if text not in self._cache:
                seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**31)
                generator = torch.Generator().manual_seed(seed)
                self._cache[text] = F.normalize(torch.randn(self.dim, generator=generator), dim=0)
            vectors.append(self._cache[text])
        return torch.stack(vectors)


def _make_pool(n: int, dim: int = 8, seed: int = 0, labels=None) -> CandidatePool:
    generator = torch.Generator().manual_seed(seed)
    embeddings = F.normalize(torch.randn(n, dim, generator=generator), dim=-1)
    paths = [Path(f"img_{i}.jpg") for i in range(n)]
    return CandidatePool(paths=paths, embeddings=embeddings, labels=labels)


# ---- retrieve_concept_sets ----


def test_retrieve_concept_sets_naive():
    pool = _make_pool(10)
    cfg = OmegaConf.create({"retrieval": {"strategy": "naive"}, "k": 3})

    present, absent = retrieve_concept_sets(cfg, pool, pool.embeddings[0])

    assert len(present) == len(absent) == 3
    assert set(present).isdisjoint(absent)


def test_retrieve_concept_sets_matched():
    pool = _make_pool(10)
    cfg = OmegaConf.create({"retrieval": {"strategy": "matched"}, "k": 3})

    present, absent = retrieve_concept_sets(cfg, pool, pool.embeddings[0])

    assert len(present) == len(absent) == 3
    assert set(present).isdisjoint(absent)


def test_retrieve_concept_sets_stratified():
    pool = _make_pool(10, labels=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    cfg = OmegaConf.create({"retrieval": {"strategy": "stratified"}, "k": 2})

    present, absent = retrieve_concept_sets(cfg, pool, pool.embeddings[0])

    assert len(present) == len(absent) == 4  # 2 per stratum x 2 strata


def test_retrieve_concept_sets_aligned():
    pool = _make_pool(10)
    cfg = OmegaConf.create({"retrieval": {"strategy": "aligned"}, "k": 3})

    present, absent = retrieve_concept_sets(cfg, pool, pool.embeddings[0])

    assert len(present) == len(absent) == 3
    assert set(present).isdisjoint(absent)


def test_retrieve_concept_sets_rejects_unknown_strategy():
    pool = _make_pool(10)
    cfg = OmegaConf.create({"retrieval": {"strategy": "bogus"}, "k": 3})

    with pytest.raises(ValueError):
        retrieve_concept_sets(cfg, pool, pool.embeddings[0])


# ---- process_concept ----


def test_process_concept_returns_direction_and_indices():
    pool = _make_pool(10)
    cfg = OmegaConf.create({"retrieval": {"strategy": "naive"}, "k": 3})
    encoder = _LookupTextEncoder(dim=8)

    result = process_concept(cfg, encoder, pool, "dog")

    assert isinstance(result, ConceptResult)
    assert result.direction.concept == "dog"
    assert len(result.present_indices) == len(result.absent_indices) == 3


def test_process_concept_text_center_changes_retrieval():
    # A text_center offset large enough to plausibly flip which images
    # rank as top/bottom-k confirms process_concept actually threads it
    # into the query used for retrieval, not just accepting and ignoring it.
    pool = _make_pool(10)
    cfg = OmegaConf.create({"retrieval": {"strategy": "naive"}, "k": 3})
    encoder = _LookupTextEncoder(dim=8)
    t_c = encoder.encode_text(["a photo of dog"])[0]  # same query build_concept_query would produce internally

    without_center = process_concept(cfg, encoder, pool, "dog")
    with_center = process_concept(cfg, encoder, pool, "dog", text_center=t_c * 0.9)

    assert with_center.present_indices != without_center.present_indices


def test_process_concept_default_text_center_is_none():
    pool = _make_pool(10)
    cfg = OmegaConf.create({"retrieval": {"strategy": "naive"}, "k": 3})
    encoder = _LookupTextEncoder(dim=8)

    explicit_none = process_concept(cfg, encoder, pool, "dog", text_center=None)
    default = process_concept(cfg, encoder, pool, "dog")

    assert explicit_none.present_indices == default.present_indices


# ---- resolve_demean_reference_concepts ----


def test_resolve_demean_reference_concepts_prefers_explicit_override():
    cfg = OmegaConf.create({"demean_reference_concepts": ["a", "b", "c"]})

    result = resolve_demean_reference_concepts(cfg, ["dog"] * 20)

    assert result == ["a", "b", "c"]


def test_resolve_demean_reference_concepts_uses_own_concepts_when_enough():
    cfg = OmegaConf.create({"demean_reference_concepts": None})
    concepts = [f"concept_{i}" for i in range(10)]

    result = resolve_demean_reference_concepts(cfg, concepts)

    assert result == concepts


def test_resolve_demean_reference_concepts_falls_back_when_too_few(caplog):
    import logging

    caplog.set_level(logging.WARNING)
    cfg = OmegaConf.create({"demean_reference_concepts": None})

    result = resolve_demean_reference_concepts(cfg, ["dog"])

    assert result == GENERIC_REFERENCE_CONCEPTS
    assert any("built-in generic reference vocabulary" in record.message for record in caplog.records)


# ---- compute_delta_c ----


def test_compute_delta_c_euclidean_reuses_direction_magnitude():
    pool = _make_pool(6)
    direction = ConceptDirection(concept="dog", unit_vector=torch.randn(8), magnitude=3.5)
    result = ConceptResult(direction=direction, present_indices=[0, 1], absent_indices=[2, 3])
    cfg = OmegaConf.create({"normalization": {"distance_fn": "euclidean"}})

    assert compute_delta_c(cfg, pool, result) == pytest.approx(3.5)


def test_compute_delta_c_angular_computes_from_centroids():
    pool = _make_pool(6)
    direction = ConceptDirection(concept="dog", unit_vector=torch.randn(8), magnitude=3.5)
    result = ConceptResult(direction=direction, present_indices=[0, 1], absent_indices=[2, 3])
    cfg = OmegaConf.create({"normalization": {"distance_fn": "angular"}})

    present_centroid = pool.embeddings[[0, 1]].mean(dim=0)
    absent_centroid = pool.embeddings[[2, 3]].mean(dim=0)
    expected = 1.0 - F.cosine_similarity(
        present_centroid.unsqueeze(0), absent_centroid.unsqueeze(0)
    ).item()

    assert compute_delta_c(cfg, pool, result) == pytest.approx(expected)


def test_compute_delta_c_rejects_unknown_distance_fn():
    pool = _make_pool(6)
    direction = ConceptDirection(concept="dog", unit_vector=torch.randn(8), magnitude=3.5)
    result = ConceptResult(direction=direction, present_indices=[0], absent_indices=[1])
    cfg = OmegaConf.create({"normalization": {"distance_fn": "bogus"}})

    with pytest.raises(ValueError):
        compute_delta_c(cfg, pool, result)


# ---- normalize_score ----


def test_normalize_score_variance():
    cfg = OmegaConf.create({"normalization": {"method": "variance"}})
    present_outputs = torch.tensor([2.0, 4.0])
    absent_outputs = torch.tensor([0.0, 0.0])

    result = normalize_score(
        cfg, raw_score=3.0, present_outputs=present_outputs, absent_outputs=absent_outputs, delta_c=1.0
    )

    combined = torch.tensor([2.0, 4.0, 0.0, 0.0])
    expected = 3.0 / combined.std(unbiased=True).item()
    assert result == pytest.approx(expected)


def test_normalize_score_embedding_distance():
    cfg = OmegaConf.create({"normalization": {"method": "embedding_distance"}})

    result = normalize_score(
        cfg,
        raw_score=4.0,
        present_outputs=torch.tensor([1.0]),
        absent_outputs=torch.tensor([0.0]),
        delta_c=2.0,
    )

    assert result == pytest.approx(2.0)


def test_normalize_score_rejects_unknown_method():
    cfg = OmegaConf.create({"normalization": {"method": "bogus"}})

    with pytest.raises(ValueError):
        normalize_score(
            cfg,
            raw_score=1.0,
            present_outputs=torch.tensor([1.0]),
            absent_outputs=torch.tensor([0.0]),
            delta_c=1.0,
        )


# ---- load_and_preprocess ----


class _FakeBlackBox:
    def preprocess(self, image):
        return torch.tensor(image.size, dtype=torch.float32)  # (width, height)

    def __call__(self, batch):
        return batch.sum(dim=1)


def test_load_and_preprocess_stacks_in_order(tmp_path):
    sizes = [(2, 2), (4, 4), (6, 6)]
    paths = []
    for i, size in enumerate(sizes):
        path = tmp_path / f"img_{i}.png"
        Image.new("RGB", size).save(path)
        paths.append(path)

    result = load_and_preprocess(paths, _FakeBlackBox())

    assert result.shape == (3, 2)
    assert result.tolist() == [[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]]


# ---- load_dataset_pool ----


def test_load_dataset_pool_dispatches_cifar(monkeypatch, tmp_path):
    calls = {}

    def fake_load_cifar(root, variant, split):
        calls["args"] = (root, variant, split)
        return [(tmp_path / "a.jpg", 0)]

    monkeypatch.setattr("cards.pipeline.load_cifar", fake_load_cifar)
    cfg = OmegaConf.create(
        {"dataset": {"name": "cifar10", "root": str(tmp_path), "variant": "cifar10"}, "pool_source": "val"}
    )

    result = load_dataset_pool(cfg)

    assert calls["args"] == (Path(str(tmp_path)), "cifar10", "val")
    assert result == [(tmp_path / "a.jpg", 0)]


def test_load_dataset_pool_rejects_broden():
    cfg = OmegaConf.create({"dataset": {"name": "broden", "root": "x"}, "pool_source": "val"})

    with pytest.raises(ValueError):
        load_dataset_pool(cfg)


def test_load_dataset_pool_rejects_unknown_dataset():
    cfg = OmegaConf.create({"dataset": {"name": "bogus", "root": "x"}, "pool_source": "val"})

    with pytest.raises(ValueError):
        load_dataset_pool(cfg)


# ---- save_directions ----


def test_save_directions_round_trips(tmp_path):
    direction = ConceptDirection(concept="dog", unit_vector=torch.tensor([1.0, 0.0]), magnitude=2.0)
    result = ConceptResult(direction=direction, present_indices=[0], absent_indices=[1])
    path = tmp_path / "out" / "directions.pt"

    save_directions([result], path)

    loaded = torch.load(path, weights_only=False)
    assert loaded["dog"]["magnitude"] == pytest.approx(2.0)
    assert torch.allclose(loaded["dog"]["unit_vector"], torch.tensor([1.0, 0.0]))


# ---- instantiate_encoder ----


class _FakeEncoder:
    """A stand-in _target_ that -- like the real OpenClipEncoder/
    PerceptionEncoder -- doesn't accept a `name` kwarg, so this catches
    the same name-leaking bug instantiate_model already guards against."""

    def __init__(self, model_name: str):
        self.model_name = model_name


def test_instantiate_encoder_strips_name_before_instantiating():
    cfg = OmegaConf.create(
        {
            "encoder": {
                "name": "fake",
                "_target_": "test_pipeline._FakeEncoder",
                "model_name": "some-model",
            }
        }
    )

    encoder = instantiate_encoder(cfg)

    assert isinstance(encoder, _FakeEncoder)
    assert encoder.model_name == "some-model"


# ---- instantiate_model ----


class _FakeBlackBoxModel:
    """A stand-in _target_ that -- like the real PosthocCBMBlackBox --
    doesn't accept a `name` kwarg, so this catches the bug where cfg.model's
    `name` field (needed for the none-check) leaked into the instantiate()
    call and broke construction."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path


def test_instantiate_model_returns_none_when_unconfigured():
    cfg = OmegaConf.create({"model": {"name": "none"}})

    assert instantiate_model(cfg) is None


def test_instantiate_model_strips_name_before_instantiating():
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "fake",
                "_target_": "test_pipeline._FakeBlackBoxModel",
                "checkpoint_path": "some/path.ckpt",
            }
        }
    )

    black_box = instantiate_model(cfg)

    assert isinstance(black_box, _FakeBlackBoxModel)
    assert black_box.checkpoint_path == "some/path.ckpt"
