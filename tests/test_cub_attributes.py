"""Tests for cards.data.cub_attributes -- synthetic fixtures, no real CUB
data needed."""

from __future__ import annotations

import pickle

from cards.data.cub_attributes import (
    ATTRIBUTE_PREFIX_TO_PARTS,
    CALIBRATED_PARTS,
    CUB_PART_NAMES,
    PART_AREA_RATIO,
    groundable_attributes,
    load_attribute_names,
    load_class_attributes,
)


def test_part_area_ratio_covers_all_15_parts_exactly_once():
    assert set(PART_AREA_RATIO) == set(CUB_PART_NAMES.values())


def test_calibrated_parts_is_the_8_cub70_validated_subset():
    assert CALIBRATED_PARTS == {
        "beak", "left_eye", "right_eye", "left_leg", "right_leg", "left_wing", "right_wing", "tail",
    }
    assert CALIBRATED_PARTS < set(PART_AREA_RATIO)


def test_load_attribute_names(tmp_path):
    (tmp_path / "attrs.txt").write_text("2 has_bill_shape::dagger\n11 has_wing_color::brown\n")

    names = load_attribute_names(tmp_path / "attrs.txt")

    assert names == ["has_bill_shape::dagger", "has_wing_color::brown"]


def test_groundable_attributes_maps_known_prefix_and_excludes_unmapped():
    names = ["has_bill_shape::dagger", "has_wing_color::brown", "has_upperparts_color::brown", "has_size::small"]

    result = groundable_attributes(names)

    assert result[0] == ("has_bill_shape", ["beak"])
    assert result[1] == ("has_wing_color", ["left_wing", "right_wing"])
    assert 2 not in result  # has_upperparts_color -- multi-region, excluded
    assert 3 not in result  # has_size -- whole-bird, excluded


def test_groundable_attributes_covers_every_declared_prefix():
    # every prefix ATTRIBUTE_PREFIX_TO_PARTS declares must actually be picked up
    names = [f"{prefix}::x" for prefix in ATTRIBUTE_PREFIX_TO_PARTS]

    result = groundable_attributes(names)

    assert len(result) == len(names)


def test_load_class_attributes_uses_first_seen_record_per_class(tmp_path):
    records_test = [
        {"class_label": 0, "attribute_label": [1, 0, 1]},
        {"class_label": 1, "attribute_label": [0, 1, 0]},
    ]
    with open(tmp_path / "test.pkl", "wb") as f:
        pickle.dump(records_test, f)
    with open(tmp_path / "train.pkl", "wb") as f:
        pickle.dump([{"class_label": 2, "attribute_label": [1, 1, 1]}], f)
    with open(tmp_path / "val.pkl", "wb") as f:
        pickle.dump([], f)

    result = load_class_attributes(tmp_path)

    assert result[1].tolist() == [True, False, True]  # class_label 0 -> CUB class_id 1
    assert result[2].tolist() == [False, True, False]
    assert result[3].tolist() == [True, True, True]
