"""Tests for the CIFAR/CUB/MetaDataset/Broden dataset loaders. Uses
tmp_path fixtures with hand-built directory structures mirroring each
dataset's real on-disk layout -- no real data or network downloads."""

from __future__ import annotations

import pytest
from PIL import Image

from cards.data.datasets import (
    list_broden_concepts,
    load_broden,
    load_cifar,
    load_cub,
    load_metadataset,
)


def _touch_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2)).save(path)


# ---- load_cifar ----


def test_load_cifar_reads_already_materialized_directory(tmp_path):
    # pre-materialize so load_cifar skips the download/materialize branch entirely
    _touch_image(tmp_path / "val" / "airplane" / "0.png")
    _touch_image(tmp_path / "val" / "airplane" / "1.png")
    _touch_image(tmp_path / "val" / "bird" / "0.png")

    result = load_cifar(tmp_path, variant="cifar10", split="val")

    assert len(result) == 3
    assert {label for _, label in result} == {0, 1}  # airplane -> 0, bird -> 1 (alphabetical)
    assert dict(result)[tmp_path / "val" / "airplane" / "0.png"] == 0
    assert dict(result)[tmp_path / "val" / "bird" / "0.png"] == 1


def test_load_cifar_rejects_unknown_variant(tmp_path):
    with pytest.raises(ValueError):
        load_cifar(tmp_path, variant="cifar17")


def test_load_cifar_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError):
        load_cifar(tmp_path, split="test")


# ---- load_cub ----


def _make_cub_dataset(root):
    _touch_image(root / "images" / "001.Albatross" / "a.jpg")
    _touch_image(root / "images" / "001.Albatross" / "b.jpg")
    _touch_image(root / "images" / "002.Auklet" / "c.jpg")

    (root / "images.txt").write_text(
        "1 001.Albatross/a.jpg\n2 001.Albatross/b.jpg\n3 002.Auklet/c.jpg\n"
    )
    (root / "image_class_labels.txt").write_text("1 1\n2 1\n3 2\n")
    (root / "train_test_split.txt").write_text("1 1\n2 0\n3 0\n")


def test_load_cub_splits_and_zero_indexes_labels(tmp_path):
    _make_cub_dataset(tmp_path)

    train = load_cub(tmp_path, split="train")
    val = load_cub(tmp_path, split="val")

    assert train == [(tmp_path / "images" / "001.Albatross" / "a.jpg", 0)]
    assert set(val) == {
        (tmp_path / "images" / "001.Albatross" / "b.jpg", 0),
        (tmp_path / "images" / "002.Auklet" / "c.jpg", 1),
    }


def test_load_cub_rejects_unknown_split(tmp_path):
    _make_cub_dataset(tmp_path)
    with pytest.raises(ValueError):
        load_cub(tmp_path, split="test")


# ---- load_metadataset ----


def test_load_metadataset_reads_imagefolder_layout(tmp_path):
    _touch_image(tmp_path / "dog_snow" / "val" / "dog" / "0.jpg")
    _touch_image(tmp_path / "dog_snow" / "val" / "not_dog" / "0.jpg")

    result = load_metadataset(tmp_path, scenario="dog_snow", split="val")

    assert len(result) == 2
    assert {label for _, label in result} == {0, 1}


def test_load_metadataset_missing_scenario_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metadataset(tmp_path, scenario="does_not_exist", split="val")


# ---- broden ----


def _make_broden_dataset(root):
    _touch_image(root / "dog" / "positives" / "p0.png")
    _touch_image(root / "dog" / "positives" / "p1.png")
    _touch_image(root / "dog" / "negatives" / "n0.png")
    _touch_image(root / "cat" / "positives" / "p0.png")
    _touch_image(root / "cat" / "negatives" / "n0.png")


def test_list_broden_concepts(tmp_path):
    _make_broden_dataset(tmp_path)
    assert list_broden_concepts(tmp_path) == ["cat", "dog"]


def test_load_broden_returns_positive_and_negative_paths(tmp_path):
    _make_broden_dataset(tmp_path)

    positives, negatives = load_broden(tmp_path, concept="dog")

    assert positives == sorted((tmp_path / "dog" / "positives").iterdir())
    assert negatives == sorted((tmp_path / "dog" / "negatives").iterdir())


def test_load_broden_unknown_concept_raises(tmp_path):
    _make_broden_dataset(tmp_path)
    with pytest.raises(ValueError):
        load_broden(tmp_path, concept="airplane")
