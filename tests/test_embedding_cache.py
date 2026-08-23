"""Tests for cards.retrieval.embedding_cache: the encode-once cache that
lets Hydra multirun sweeps reuse embeddings across jobs that share a
(dataset, split, encoder) identity but differ only in retrieval/
normalization -- see notes/ablation_scope_decision.md."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool


class _CountingImageEncoder:
    """Tracks how many times encode_images was called, so tests can assert
    a cache hit skips real encoding entirely."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.call_count = 0
        self._next_value = 0

    def encode_images(self, images):
        self.call_count += 1
        values = torch.arange(self._next_value, self._next_value + len(images), dtype=torch.float32)
        self._next_value += len(images)
        return values.unsqueeze(1).repeat(1, self.dim)


def _touch_image(path):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2)).save(path)


def _make_pairs(tmp_path, names):
    paths = [tmp_path / f"{name}.jpg" for name in names]
    for path in paths:
        _touch_image(path)
    return [(path, i) for i, path in enumerate(paths)]


# ---- cache_key_for ----


def test_cache_key_for_stable_across_equivalent_configs():
    cfg_a = OmegaConf.create(
        {
            "dataset": {"name": "cifar10", "root": "x", "variant": "cifar10"},
            "pool_source": "val",
            "encoder": {"model_name": "ViT-B-32", "pretrained": "openai"},
        }
    )
    cfg_b = OmegaConf.create(
        {
            "dataset": {"name": "cifar10", "root": "x", "variant": "cifar10"},
            "pool_source": "val",
            "encoder": {"model_name": "ViT-B-32", "pretrained": "openai"},
        }
    )

    assert cache_key_for(cfg_a) == cache_key_for(cfg_b)


def test_cache_key_for_differs_by_encoder():
    base = {
        "dataset": {"name": "cifar10", "root": "x", "variant": "cifar10"},
        "pool_source": "val",
    }
    cfg_clip = OmegaConf.create({**base, "encoder": {"model_name": "ViT-B-32", "pretrained": "openai"}})
    cfg_siglip = OmegaConf.create(
        {**base, "encoder": {"model_name": "ViT-B-16-SigLIP", "pretrained": "webli"}}
    )

    assert cache_key_for(cfg_clip) != cache_key_for(cfg_siglip)


def test_cache_key_for_handles_encoder_configs_with_different_schemas():
    # Regression test: cache_key_for used to hard-code cfg.encoder.pretrained,
    # which OpenClipEncoder-based configs have but PerceptionEncoder's
    # (model_name + perception_models_path, no `pretrained`) doesn't --
    # raised ConfigAttributeError the first time a non-OpenClip encoder
    # was actually run through the pipeline.
    base = {
        "dataset": {"name": "cifar10", "root": "x", "variant": "cifar10"},
        "pool_source": "val",
    }
    cfg_open_clip = OmegaConf.create({**base, "encoder": {"model_name": "ViT-B-32", "pretrained": "openai"}})
    cfg_perception = OmegaConf.create(
        {**base, "encoder": {"model_name": "PE-Core-B16-224", "perception_models_path": "../perception_models"}}
    )

    key = cache_key_for(cfg_perception)  # must not raise
    assert isinstance(key, str) and len(key) > 0
    assert key != cache_key_for(cfg_open_clip)


def test_cache_key_for_differs_by_dataset_and_pool_source():
    encoder = {"encoder": {"model_name": "ViT-B-32", "pretrained": "openai"}}
    cfg_val = OmegaConf.create(
        {"dataset": {"name": "cifar10", "root": "x", "variant": "cifar10"}, "pool_source": "val", **encoder}
    )
    cfg_train = OmegaConf.create(
        {"dataset": {"name": "cifar10", "root": "x", "variant": "cifar10"}, "pool_source": "train", **encoder}
    )
    cfg_other_dataset = OmegaConf.create(
        {"dataset": {"name": "cifar100", "root": "x", "variant": "cifar100"}, "pool_source": "val", **encoder}
    )

    assert cache_key_for(cfg_val) != cache_key_for(cfg_train)
    assert cache_key_for(cfg_val) != cache_key_for(cfg_other_dataset)


# ---- load_or_build_pool ----


def test_load_or_build_pool_builds_and_caches_on_first_call(tmp_path):
    cache_dir = tmp_path / "cache"
    pairs = _make_pairs(tmp_path, ["a", "b", "c"])
    encoder = _CountingImageEncoder()

    pool = load_or_build_pool(cache_dir, "key1", pairs, encoder)

    assert encoder.call_count == 1
    assert len(pool.paths) == 3
    assert pool.labels == [0, 1, 2]
    assert (cache_dir / "key1.pt").exists()


def test_load_or_build_pool_reuses_cache_without_reencoding(tmp_path):
    cache_dir = tmp_path / "cache"
    pairs = _make_pairs(tmp_path, ["a", "b", "c"])
    encoder = _CountingImageEncoder()

    first_pool = load_or_build_pool(cache_dir, "key1", pairs, encoder)
    second_pool = load_or_build_pool(cache_dir, "key1", pairs, encoder)

    assert encoder.call_count == 1  # second call was a cache hit, no re-encoding
    assert torch.equal(first_pool.embeddings, second_pool.embeddings)


def test_load_or_build_pool_recomputes_on_changed_pairs(tmp_path):
    cache_dir = tmp_path / "cache"
    pairs_a = _make_pairs(tmp_path, ["a", "b", "c"])
    pairs_b = _make_pairs(tmp_path, ["x", "y"])  # different paths, same key
    encoder = _CountingImageEncoder()

    load_or_build_pool(cache_dir, "key1", pairs_a, encoder)
    pool_b = load_or_build_pool(cache_dir, "key1", pairs_b, encoder)

    assert encoder.call_count == 2  # fingerprint mismatch forced a real recompute
    assert len(pool_b.paths) == 2


def test_load_or_build_pool_different_keys_dont_collide(tmp_path):
    cache_dir = tmp_path / "cache"
    pairs = _make_pairs(tmp_path, ["a", "b"])
    encoder = _CountingImageEncoder()

    load_or_build_pool(cache_dir, "key1", pairs, encoder)
    load_or_build_pool(cache_dir, "key2", pairs, encoder)

    assert encoder.call_count == 2
    assert (cache_dir / "key1.pt").exists()
    assert (cache_dir / "key2.pt").exists()
