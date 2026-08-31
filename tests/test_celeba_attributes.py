"""Tests for cards.data.celeba_attributes -- synthetic fixtures, no real
CelebA data needed (though the loader functions were also cross-checked
manually against the real, already-local list_attr_celeba.txt during
implementation)."""

from __future__ import annotations

from cards.data.celeba_attributes import (
    ATTRIBUTE_TO_REGIONS,
    CELEBA_MASK_CLASSES,
    EXCLUDED_ATTRIBUTES,
    GROUNDABLE_CONCEPTS,
    PILOT_CONCEPTS,
    TARGET_CLASSES,
    groundable_attributes,
    load_attribute_labels,
    load_attribute_names,
)


def _write_attr_file(tmp_path, names: list[str], rows: dict[str, list[int]]):
    lines = [str(len(rows)), " ".join(names)]
    for filename, values in rows.items():
        lines.append(filename + " " + " ".join(str(v) for v in values))
    (tmp_path / "list_attr_celeba.txt").write_text("\n".join(lines) + "\n")
    return tmp_path / "list_attr_celeba.txt"


def test_load_attribute_names(tmp_path):
    path = _write_attr_file(tmp_path, ["Smiling", "Young", "Attractive"], {})

    names = load_attribute_names(path)

    assert names == ["Smiling", "Young", "Attractive"]


def test_load_attribute_labels_converts_pm1_to_bool(tmp_path):
    path = _write_attr_file(tmp_path, ["Smiling", "Young"], {"000001.jpg": [1, -1], "000002.jpg": [-1, 1]})

    labels = load_attribute_labels(path)

    assert labels["000001.jpg"].tolist() == [True, False]
    assert labels["000002.jpg"].tolist() == [False, True]


def test_target_classes_are_never_groundable_concepts():
    # the whole point of the Attractive/Young split: no target class
    # should ever also appear as a key in the concept->region mapping
    assert not set(TARGET_CLASSES) & set(ATTRIBUTE_TO_REGIONS)


def test_target_and_excluded_and_groundable_partition_all_40_attributes():
    # every real CelebA attribute is accounted for exactly once across
    # the three buckets -- none silently dropped, none double-counted
    all_accounted = set(TARGET_CLASSES) | set(EXCLUDED_ATTRIBUTES) | set(ATTRIBUTE_TO_REGIONS)
    assert len(all_accounted) == len(TARGET_CLASSES) + len(EXCLUDED_ATTRIBUTES) + len(ATTRIBUTE_TO_REGIONS)
    assert len(all_accounted) == 40


def test_groundable_attributes_maps_known_attribute_and_excludes_others():
    names = ["Smiling", "Attractive", "Blurry", "Big_Nose"]

    result = groundable_attributes(names)

    assert result[0] == ("Smiling", ["mouth"])
    assert 1 not in result  # Attractive -- a target class, not a concept
    assert 2 not in result  # Blurry -- excluded, whole-image property
    assert result[3] == ("Big_Nose", ["nose"])


def test_all_mapped_regions_are_real_celebamask_hq_classes():
    for attribute, regions in ATTRIBUTE_TO_REGIONS.items():
        for region in regions:
            assert region in CELEBA_MASK_CLASSES, f"{attribute} maps to unknown region {region!r}"


def test_pilot_concepts_are_all_groundable():
    for concept in PILOT_CONCEPTS:
        assert concept in ATTRIBUTE_TO_REGIONS, f"pilot concept {concept!r} has no region mapping"


def test_pilot_concepts_span_multiple_distinct_regions():
    # the pilot is deliberately diverse (large/small masks, color/shape/
    # dynamic attributes) -- not accidentally all mapping to one region
    regions_used = {r for c in PILOT_CONCEPTS for r in ATTRIBUTE_TO_REGIONS[c]}
    assert len(regions_used) >= 5


def test_groundable_concepts_matches_attribute_to_regions_exactly():
    assert set(GROUNDABLE_CONCEPTS) == set(ATTRIBUTE_TO_REGIONS)
    assert len(GROUNDABLE_CONCEPTS) == len(set(GROUNDABLE_CONCEPTS))  # no duplicates
    assert GROUNDABLE_CONCEPTS == sorted(GROUNDABLE_CONCEPTS)


def test_pilot_concepts_are_a_subset_of_groundable_concepts():
    assert set(PILOT_CONCEPTS) <= set(GROUNDABLE_CONCEPTS)
