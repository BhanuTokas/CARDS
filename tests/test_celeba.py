"""Tests for cards.data.celeba -- synthetic fixtures matching the REAL
CelebAMask-HQ layout confirmed directly against the extracted archive
(bucket-of-2000 subfolders, <idx:05d>_<region>.png filenames, 512x512
binary masks), not the dataset's own README (which omits these
specifics). See cards.data.celeba's own module docstring for what was
confirmed and how."""

from __future__ import annotations

import numpy as np
from PIL import Image

from cards.data.celeba import load_celebamask_hq_image_paths, load_celebamask_hq_mask


def _write_mask(root, hq_index: int, region: str, mask: np.ndarray):
    bucket_dir = root / "CelebAMask-HQ-mask-anno" / str(hq_index // 2000)
    bucket_dir.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img.save(bucket_dir / f"{hq_index:05d}_{region}.png")


def test_load_celebamask_hq_image_paths(tmp_path):
    img_dir = tmp_path / "CelebA-HQ-img"
    img_dir.mkdir()
    (img_dir / "0.jpg").touch()
    (img_dir / "29999.jpg").touch()

    paths = load_celebamask_hq_image_paths(tmp_path)

    assert set(paths) == {0, 29999}
    assert paths[0].name == "0.jpg"


def test_load_celebamask_hq_mask_single_region(tmp_path):
    mask = np.zeros((512, 512), dtype=bool)
    mask[100:200, 100:200] = True
    _write_mask(tmp_path, 0, "hair", mask)

    result = load_celebamask_hq_mask(tmp_path, 0, ["hair"])

    assert result.shape == (512, 512)
    assert result.dtype == bool
    assert result[150, 150]
    assert not result[0, 0]


def test_load_celebamask_hq_mask_or_combines_multiple_regions(tmp_path):
    left = np.zeros((512, 512), dtype=bool)
    left[10:20, 10:20] = True
    right = np.zeros((512, 512), dtype=bool)
    right[400:410, 400:410] = True
    _write_mask(tmp_path, 5, "l_eye", left)
    _write_mask(tmp_path, 5, "r_eye", right)

    result = load_celebamask_hq_mask(tmp_path, 5, ["l_eye", "r_eye"])

    assert result[15, 15]  # from l_eye
    assert result[405, 405]  # from r_eye
    assert not result[256, 256]


def test_load_celebamask_hq_mask_absent_region_returns_all_false(tmp_path):
    (tmp_path / "CelebAMask-HQ-mask-anno" / "0").mkdir(parents=True)

    result = load_celebamask_hq_mask(tmp_path, 3, ["eye_g"])

    assert result.shape == (512, 512)
    assert not result.any()


def test_load_celebamask_hq_mask_uses_correct_bucket(tmp_path):
    # index 2500 -> bucket 1, not bucket 0
    mask = np.ones((512, 512), dtype=bool)
    _write_mask(tmp_path, 2500, "skin", mask)

    result = load_celebamask_hq_mask(tmp_path, 2500, ["skin"])

    assert result.all()
    assert (tmp_path / "CelebAMask-HQ-mask-anno" / "1" / "02500_skin.png").exists()


def test_load_celebamask_hq_mask_resizes_to_target_with_nearest_neighbor(tmp_path):
    mask = np.zeros((512, 512), dtype=bool)
    mask[:256, :] = True  # top half
    _write_mask(tmp_path, 0, "hair", mask)

    result = load_celebamask_hq_mask(tmp_path, 0, ["hair"], target_hw=(1024, 1024))

    assert result.shape == (1024, 1024)
    assert result.dtype == bool
    assert result[100, 100]  # still top half
    assert not result[900, 100]  # still bottom half
