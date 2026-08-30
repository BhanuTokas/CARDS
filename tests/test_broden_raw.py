"""Tests for cards.data.broden_raw's parsing/decoding logic, using small
synthetic fixtures (tmp_path) -- no dependency on the real NetDissect
checkout. Reading the actual index.csv/label.csv/masks against the local
Datasets is integration-only, verified separately, not here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cards.data.broden_raw import (
    BrodenRecord,
    _parse_category_field,
    _parse_label_number_field,
    _parse_mask_field,
    build_concept_index,
    concept_pixel_mask,
    decode_mask,
)

# ---- field parsing ----


def test_parse_category_field_single():
    assert _parse_category_field("object(4743)") == {"object": 4743}


def test_parse_category_field_multi():
    assert _parse_category_field("wall(15553);part(29)") == {"wall": 15553, "part": 29}


def test_parse_category_field_empty():
    assert _parse_category_field("") == {}


def test_parse_mask_field_multi(tmp_path):
    paths = _parse_mask_field("a/1_object.png;a/1_object_1.png", tmp_path)
    assert paths == [tmp_path / "images" / "a/1_object.png", tmp_path / "images" / "a/1_object_1.png"]


def test_parse_mask_field_empty(tmp_path):
    assert _parse_mask_field("", tmp_path) == []


def test_parse_label_number_field_multi():
    assert _parse_label_number_field("290;372") == [290, 372]


def test_parse_label_number_field_empty():
    assert _parse_label_number_field("") == []


# ---- decode_mask ----


def test_decode_mask_r_plus_256g(tmp_path):
    # pixel value = R + 256*G; a single pixel with R=38, G=0 -> label 38
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    arr[0, 0] = [38, 0, 0]
    arr[1, 1] = [0, 1, 0]  # R=0, G=1 -> 256
    path = tmp_path / "mask.png"
    Image.fromarray(arr).save(path)

    decoded = decode_mask(path)

    assert decoded[0, 0] == 38
    assert decoded[1, 1] == 256


# ---- concept_pixel_mask ----


def test_concept_pixel_mask_matches_label(tmp_path):
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[0:2, 0:2] = [38, 0, 0]  # label 38 in top-left quadrant
    mask_path = tmp_path / "img_object.png"
    Image.fromarray(arr).save(mask_path)

    record = BrodenRecord(
        image=tmp_path / "img.jpg", split="train", ih=4, iw=4, sh=4, sw=4,
        mask_paths={"object": [mask_path]},
    )

    mask = concept_pixel_mask(record, "object", 38)

    assert mask.shape == (4, 4)
    assert mask[0:2, 0:2].all()
    assert not mask[2:, 2:].any()


def test_concept_pixel_mask_none_when_no_mask_file():
    record = BrodenRecord(image=Path("x.jpg"), split="train", ih=4, iw=4, sh=4, sw=4, mask_paths={"object": []})

    assert concept_pixel_mask(record, "object", 38) is None


def test_concept_pixel_mask_none_for_image_category():
    record = BrodenRecord(image=Path("x.jpg"), split="train", ih=4, iw=4, sh=4, sw=4)

    assert concept_pixel_mask(record, "scene", 1) is None


def test_concept_pixel_mask_ors_multiple_part_planes(tmp_path):
    arr1 = np.zeros((4, 4, 3), dtype=np.uint8)
    arr1[0, 0] = [5, 0, 0]
    path1 = tmp_path / "p1.png"
    Image.fromarray(arr1).save(path1)

    arr2 = np.zeros((4, 4, 3), dtype=np.uint8)
    arr2[3, 3] = [5, 0, 0]
    path2 = tmp_path / "p2.png"
    Image.fromarray(arr2).save(path2)

    record = BrodenRecord(
        image=tmp_path / "img.jpg", split="train", ih=4, iw=4, sh=4, sw=4,
        mask_paths={"part": [path1, path2]},
    )

    mask = concept_pixel_mask(record, "part", 5)

    assert mask[0, 0] and mask[3, 3]
    assert mask.sum() == 2


def test_concept_pixel_mask_upsamples_from_lower_res(tmp_path):
    # mask at half resolution (sh/sw = ih/iw / 2), like real broden1_224 data
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    arr[0, 0] = [38, 0, 0]  # top-left quadrant at low res
    path = tmp_path / "mask.png"
    Image.fromarray(arr).save(path)

    record = BrodenRecord(
        image=tmp_path / "img.jpg", split="train", ih=4, iw=4, sh=2, sw=2,
        mask_paths={"object": [path]},
    )
    mask = concept_pixel_mask(record, "object", 38)

    assert mask.shape == (4, 4)
    assert mask[0:2, 0:2].all()


# ---- build_concept_index ----


def test_build_concept_index_buckets_by_label(tmp_path):
    arr1 = np.zeros((2, 2, 3), dtype=np.uint8)
    arr1[0, 0] = [38, 0, 0]  # car
    path1 = tmp_path / "a_object.png"
    Image.fromarray(arr1).save(path1)

    arr2 = np.zeros((2, 2, 3), dtype=np.uint8)
    arr2[0, 0] = [184, 0, 0]  # motorbike
    path2 = tmp_path / "b_object.png"
    Image.fromarray(arr2).save(path2)

    rec_car = BrodenRecord(image=tmp_path / "a.jpg", split="train", ih=2, iw=2, sh=2, sw=2, mask_paths={"object": [path1]})
    rec_moto = BrodenRecord(image=tmp_path / "b.jpg", split="train", ih=2, iw=2, sh=2, sw=2, mask_paths={"object": [path2]})
    rec_none = BrodenRecord(image=tmp_path / "c.jpg", split="train", ih=2, iw=2, sh=2, sw=2, mask_paths={"object": []})

    index = build_concept_index([rec_car, rec_moto, rec_none], "object")

    assert index[38] == [rec_car]
    assert index[184] == [rec_moto]
    assert 0 not in index


def test_build_concept_index_rejects_image_category():
    with pytest.raises(ValueError):
        build_concept_index([], "scene")
