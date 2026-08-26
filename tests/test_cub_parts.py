"""Tests for cards.data.cub_parts -- synthetic fixtures, no real CUB/
CUB70 data needed."""

from __future__ import annotations

import numpy as np
from PIL import Image

from cards.data.cub_parts import (
    CUB70_TO_CUB_PART_ID,
    keypoint_patch_mask,
    load_cub70_mask,
    load_cub_segmentation,
    load_images_txt,
    load_keypoints,
)


def test_load_images_txt(tmp_path):
    (tmp_path / "images.txt").write_text("1 001.Foo/a.jpg\n2 001.Foo/b.jpg\n")

    result = load_images_txt(tmp_path)

    assert result["1"] == tmp_path / "images" / "001.Foo/a.jpg"
    assert result["2"] == tmp_path / "images" / "001.Foo/b.jpg"


def test_load_keypoints(tmp_path):
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts" / "part_locs.txt").write_text("1 2 10.5 20.5 1\n1 3 0.0 0.0 0\n")

    result = load_keypoints(tmp_path)

    assert result["1"][2] == (10.5, 20.5, True)
    assert result["1"][3] == (0.0, 0.0, False)


def test_load_cub70_mask_reads_binary_mask(tmp_path):
    class_dir = tmp_path / "AnnotationMasksPerclass" / "1"
    class_dir.mkdir(parents=True)
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[2:5, 2:5] = 255
    Image.fromarray(arr).save(class_dir / "bird_beak.png")

    mask = load_cub70_mask(tmp_path, "1", "bird", "beak")

    assert mask.shape == (10, 10)
    assert mask[2:5, 2:5].all()
    assert not mask[0, 0]


def test_load_cub70_mask_returns_none_when_missing(tmp_path):
    (tmp_path / "AnnotationMasksPerclass" / "1").mkdir(parents=True)

    assert load_cub70_mask(tmp_path, "1", "bird", "beak") is None


def test_load_cub_segmentation_reads_mirrored_png_path(tmp_path):
    (tmp_path / "images.txt").write_text("1 001.Foo/a.jpg\n")
    image_paths = load_images_txt(tmp_path)
    seg_dir = tmp_path / "segmentations" / "001.Foo"
    seg_dir.mkdir(parents=True)
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[3:7, 3:7] = 255
    Image.fromarray(arr).save(seg_dir / "a.png")

    mask = load_cub_segmentation(tmp_path, "1", image_paths)

    assert mask.shape == (10, 10)
    assert mask[3:7, 3:7].all()
    assert not mask[0, 0]


def test_keypoint_patch_mask_area_matches_target():
    mask = keypoint_patch_mask(x=50, y=50, target_area=314.0, image_shape=(100, 100))  # r=~10

    # discretization means this won't be exact, but should be close
    assert abs(mask.sum() - 314) < 40


def test_keypoint_patch_mask_centered_at_keypoint():
    mask = keypoint_patch_mask(x=50, y=50, target_area=100.0, image_shape=(100, 100))

    assert mask[50, 50]  # note: array indexing is [row=y, col=x]


def test_keypoint_patch_mask_clips_to_image_bounds():
    mask = keypoint_patch_mask(x=0, y=0, target_area=1000.0, image_shape=(20, 20))

    assert mask.shape == (20, 20)
    assert mask.sum() < 1000  # clipped, can't reach the full target area


def test_cub70_to_cub_part_id_has_eight_direct_matches():
    assert len(CUB70_TO_CUB_PART_ID) == 8
    assert set(CUB70_TO_CUB_PART_ID) == {
        "beak", "left_eye", "right_eye", "left_wing", "right_wing", "left_leg", "right_leg", "tail",
    }
