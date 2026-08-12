"""Tests for CandidatePool.build (directory scan) and .from_pairs (explicit
list, e.g. from a cards.data.datasets loader)."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from cards.retrieval.pool import CandidatePool


class _CountingImageEncoder:
    """Assigns each image an embedding equal to its call order, so tests
    can verify path/embedding order survives batching."""

    def __init__(self, dim: int = 2):
        self.dim = dim
        self._next_value = 0

    def encode_images(self, images):
        values = torch.arange(self._next_value, self._next_value + len(images), dtype=torch.float32)
        self._next_value += len(images)
        return values.unsqueeze(1).repeat(1, self.dim)


def _touch_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2)).save(path)


def test_build_scans_directory_and_preserves_sorted_order(tmp_path):
    for name in ["c.jpg", "a.jpg", "b.jpg", "d.jpg", "e.jpg"]:
        _touch_image(tmp_path / name)

    pool = CandidatePool.build(tmp_path, _CountingImageEncoder(), batch_size=2)

    assert len(pool.paths) == 5
    assert pool.paths == sorted(pool.paths)
    assert pool.embeddings[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert pool.labels is None


def test_build_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError):
        CandidatePool.build(tmp_path, _CountingImageEncoder())


def test_build_rejects_mismatched_labels_length(tmp_path):
    _touch_image(tmp_path / "a.jpg")
    _touch_image(tmp_path / "b.jpg")

    with pytest.raises(ValueError):
        CandidatePool.build(tmp_path, _CountingImageEncoder(), labels=[0])


def test_from_pairs_preserves_given_order_and_labels(tmp_path):
    paths = [tmp_path / f"{name}.jpg" for name in ["z", "a", "m"]]
    for path in paths:
        _touch_image(path)
    pairs = [(paths[0], 5), (paths[1], 7), (paths[2], 9)]

    pool = CandidatePool.from_pairs(pairs, _CountingImageEncoder(), batch_size=2)

    assert pool.paths == paths  # not re-sorted, unlike build()
    assert pool.labels == [5, 7, 9]
    assert pool.embeddings[:, 0].tolist() == [0.0, 1.0, 2.0]


def test_from_pairs_rejects_empty_list():
    with pytest.raises(ValueError):
        CandidatePool.from_pairs([], _CountingImageEncoder())
