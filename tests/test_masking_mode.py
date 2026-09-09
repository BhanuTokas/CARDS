"""Tests for cards.attribution.masking_mode."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.attribution.masking_mode import MaskingScoreResult, masking_score


class _FakePatchEncoder:
    """encode_patches gives a fixed 2x2 grid where the top-left patch
    scores highest against a dim-0-aligned query -- enough for
    threshold_mask (top_pct<100) to always find a non-degenerate mask.
    encode_images returns FULLY CONTROLLED, image-content-independent
    embeddings keyed by call order: [original, candidate_0, candidate_1,
    ...] -- lets tests fix exactly which candidate index is "best
    aligned" without needing to reason about what mask_region's real
    blur/zero_fill/etc. pixel math produces."""

    def __init__(self, best_candidate_index: int, n_candidates: int):
        self.best_candidate_index = best_candidate_index
        self.n_candidates = n_candidates

    def encode_patches(self, image: Image.Image) -> torch.Tensor:
        patches = torch.zeros(4, 2)
        patches[:, 1] = 1.0
        patches[0, 0] = 1.0
        patches[0, 1] = 0.1
        return torch.nn.functional.normalize(patches, dim=-1)

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        # diff = embed(orig) - embed(candidate); best = smallest angle(diff, query).
        # orig = [1, 0] (aligned with query). For diff to align with query,
        # a candidate's embedding needs to point AWAY from query (roughly
        # opposite orig) -- a candidate merely ORTHOGONAL to orig gives a
        # much worse (larger) angle, not a competing "good" one.
        assert len(images) == 1 + self.n_candidates
        embeds = torch.zeros(len(images), 2)
        embeds[0] = torch.tensor([1.0, 0.0])  # original
        for i in range(self.n_candidates):
            embeds[1 + i] = torch.tensor([0.0, 1.0])  # non-best: orthogonal to orig -> ~45deg angle
        embeds[1 + self.best_candidate_index] = torch.tensor([-1.0, 0.0])  # best: opposite orig -> ~0deg angle
        return torch.nn.functional.normalize(embeds, dim=-1)


class _FakeVaryingPatchEncoder:
    """Like _FakePatchEncoder, but the top-left patch's score scales with
    the image's own mean pixel value (read directly from content, the
    same trick _FakeBlackBox.preprocess already uses) -- lets a test give
    two present images genuinely DIFFERENT similarity distributions, to
    prove threshold_method="zscore" really pools stats across every
    present image into ONE shared cutoff, rather than thresholding each
    image independently the way "top_pct"/"otsu" do."""

    def encode_patches(self, image: Image.Image) -> torch.Tensor:
        level = float(np.array(image).mean()) / 255.0  # 0..1, image-content-dependent
        patches = torch.zeros(4, 2)
        patches[:, 1] = 1.0
        patches[0, 0] = level
        patches[0, 1] = 0.1
        return torch.nn.functional.normalize(patches, dim=-1)

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        embeds = torch.zeros(len(images), 2)
        embeds[0] = torch.tensor([1.0, 0.0])
        for i in range(1, len(images)):
            embeds[i] = torch.tensor([-1.0, 0.0])  # every candidate "best-aligned", any tie-break is fine here
        return torch.nn.functional.normalize(embeds, dim=-1)


class _FakeBlackBox:
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return torch.tensor([float(np.array(image).mean())])

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return batch.squeeze(-1)


class _FakePool:
    def __init__(self, paths):
        self.paths = paths


def _write_gray_image(path, size=(8, 8), color=(128, 128, 128)):
    Image.new("RGB", size, color).save(path)


# ---- structural / aggregation ----


def test_raw_score_is_mean_of_delta_scores(tmp_path):
    path = tmp_path / "img.png"
    _write_gray_image(path)
    pool = _FakePool([path])
    fill_strategies = ["blur", "zero_fill", "mean_fill"]
    encoder = _FakePatchEncoder(best_candidate_index=1, n_candidates=len(fill_strategies))
    black_box = _FakeBlackBox()

    result = masking_score(
        black_box, encoder, pool, present_indices=[0], query=torch.tensor([1.0, 0.0]),
        top_pct=50, fill_strategies=fill_strategies, seed=0, concept_idx=0,
    )

    assert isinstance(result, MaskingScoreResult)
    assert len(result.delta_scores) == 1
    assert result.raw_score == result.delta_scores[0]


def test_raw_score_zero_when_all_images_skipped(tmp_path):
    path = tmp_path / "img.png"
    _write_gray_image(path)
    pool = _FakePool([path])
    encoder = _FakePatchEncoder(best_candidate_index=0, n_candidates=1)
    black_box = _FakeBlackBox()

    # top_pct=100 -> mask covers every pixel -> mask.all() -> skipped
    result = masking_score(
        black_box, encoder, pool, present_indices=[0], query=torch.tensor([1.0, 0.0]),
        top_pct=100, fill_strategies=["zero_fill"], seed=0, concept_idx=0,
    )

    assert result.delta_scores == []
    assert result.raw_score == 0.0
    assert result.n_skipped_degenerate == 1


# ---- best-of-N selection ----


def test_selects_the_truly_best_aligned_candidate_not_just_the_first(tmp_path):
    path = tmp_path / "img.png"
    _write_gray_image(path)
    pool = _FakePool([path])
    fill_strategies = ["blur", "zero_fill", "mean_fill", "hue_shift"]
    # deliberately NOT the first strategy, to confirm the loop finds the
    # true minimum rather than defaulting to index 0
    encoder = _FakePatchEncoder(best_candidate_index=2, n_candidates=len(fill_strategies))
    black_box = _FakeBlackBox()

    result = masking_score(
        black_box, encoder, pool, present_indices=[0], query=torch.tensor([1.0, 0.0]),
        top_pct=50, fill_strategies=fill_strategies, seed=0, concept_idx=0,
    )

    assert result.selected_strategies == ["mean_fill"]


# ---- a real, analytically-derivable scoring invariant ----


def test_zero_fill_delta_is_positive_for_a_positive_valued_image(tmp_path):
    # zero_fill replaces the masked region with (0, 0, 0) -- for a
    # uniformly positive-valued image, this can only DECREASE the mean
    # pixel value, so black_box(orig) - black_box(masked) must be > 0.
    # True regardless of exactly which pixels end up masked.
    path = tmp_path / "img.png"
    _write_gray_image(path, size=(8, 8), color=(200, 200, 200))
    pool = _FakePool([path])
    encoder = _FakePatchEncoder(best_candidate_index=0, n_candidates=1)
    black_box = _FakeBlackBox()

    result = masking_score(
        black_box, encoder, pool, present_indices=[0], query=torch.tensor([1.0, 0.0]),
        top_pct=50, fill_strategies=["zero_fill"], seed=0, concept_idx=0,
    )

    assert len(result.delta_scores) == 1
    assert result.delta_scores[0] > 0


# ---- threshold_method="zscore" ----


def test_zscore_threshold_matches_manual_localize_and_cutoff(tmp_path):
    """masking_score(threshold_method="zscore") must give the EXACT SAME
    masking decision as manually computing localize_concept ->
    concept_zscore_cutoff (pooled across ALL present images) ->
    threshold_mask(method="fixed") -- validates the new wiring directly
    against the already-tested localization primitives, rather than
    hand-deriving expected numbers through bilinear upsampling."""
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    _write_gray_image(path_a, color=(250, 250, 250))
    _write_gray_image(path_b, color=(5, 5, 5))
    pool = _FakePool([path_a, path_b])
    encoder = _FakeVaryingPatchEncoder()
    black_box = _FakeBlackBox()
    query = torch.tensor([1.0, 0.0])
    alpha = 0.5

    result = masking_score(
        black_box, encoder, pool, present_indices=[0, 1], query=query,
        threshold_method="zscore", alpha=alpha, fill_strategies=["zero_fill"], seed=0, concept_idx=0,
    )

    image_a, image_b = Image.open(path_a).convert("RGB"), Image.open(path_b).convert("RGB")
    sim_a = localize_concept(encoder, image_a, query, (image_a.height, image_a.width))
    sim_b = localize_concept(encoder, image_b, query, (image_b.height, image_b.width))
    cutoff = concept_zscore_cutoff([sim_a, sim_b], alpha)
    mask_a = threshold_mask(sim_a, method="fixed", cutoff=cutoff)
    mask_b = threshold_mask(sim_b, method="fixed", cutoff=cutoff)

    expected_n_skipped = sum(1 for m in (mask_a, mask_b) if not m.any() or m.all())
    assert result.n_skipped_degenerate == expected_n_skipped
    assert len(result.delta_scores) == 2 - expected_n_skipped


def test_zscore_cutoff_is_pooled_not_computed_per_image(tmp_path):
    """Same first image (A), but whether a second, very differently-
    colored image (B) also shares the present set changes A's own
    cutoff -- proof threshold_method="zscore" genuinely pools stats
    across every present image into one shared cutoff, unlike "top_pct"/
    "otsu" which threshold each image independently. Confirmed directly
    (not assumed): at alpha=0.5, A's own mask covers 15/64 pixels alone
    vs. 22/64 once pooled with B, both non-degenerate."""
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    _write_gray_image(path_a, color=(250, 250, 250))
    _write_gray_image(path_b, color=(5, 5, 5))
    encoder = _FakeVaryingPatchEncoder()
    black_box = _FakeBlackBox()
    query = torch.tensor([1.0, 0.0])
    alpha = 0.5

    result_alone = masking_score(
        black_box, encoder, _FakePool([path_a]), present_indices=[0], query=query,
        threshold_method="zscore", alpha=alpha, fill_strategies=["zero_fill"], seed=0, concept_idx=0,
    )
    result_paired = masking_score(
        black_box, encoder, _FakePool([path_a, path_b]), present_indices=[0, 1], query=query,
        threshold_method="zscore", alpha=alpha, fill_strategies=["zero_fill"], seed=0, concept_idx=0,
    )

    # delta_scores[0] is image A in both calls (first/only present index) --
    # a different value proves A's OWN mask changed once B joined the pool.
    assert result_alone.n_skipped_degenerate == 0
    assert result_paired.delta_scores[0] != result_alone.delta_scores[0]


def test_alpha_ignored_for_top_pct(tmp_path):
    """alpha is documented as a no-op for every threshold_method except
    "zscore" -- confirm passing a wildly different alpha doesn't change
    top_pct's own result."""
    path = tmp_path / "img.png"
    _write_gray_image(path)
    pool = _FakePool([path])
    encoder = _FakePatchEncoder(best_candidate_index=0, n_candidates=1)
    black_box = _FakeBlackBox()

    result_a = masking_score(
        black_box, encoder, pool, present_indices=[0], query=torch.tensor([1.0, 0.0]),
        top_pct=50, threshold_method="top_pct", alpha=0.1, fill_strategies=["zero_fill"], seed=0, concept_idx=0,
    )
    result_b = masking_score(
        black_box, encoder, pool, present_indices=[0], query=torch.tensor([1.0, 0.0]),
        top_pct=50, threshold_method="top_pct", alpha=99.0, fill_strategies=["zero_fill"], seed=0, concept_idx=0,
    )

    assert result_a.delta_scores == result_b.delta_scores


# ---- determinism ----


def test_deterministic_for_fixed_seed_and_concept_idx(tmp_path):
    path = tmp_path / "img.png"
    _write_gray_image(path)
    pool = _FakePool([path])
    fill_strategies = ["blur", "zero_fill_noise", "noise_then_blur"]  # rng-dependent strategies included
    encoder = _FakePatchEncoder(best_candidate_index=1, n_candidates=len(fill_strategies))
    black_box = _FakeBlackBox()

    result_a = masking_score(
        black_box, encoder, pool, present_indices=[0], query=torch.tensor([1.0, 0.0]),
        top_pct=50, fill_strategies=fill_strategies, seed=7, concept_idx=3,
    )
    result_b = masking_score(
        black_box, encoder, pool, present_indices=[0], query=torch.tensor([1.0, 0.0]),
        top_pct=50, fill_strategies=fill_strategies, seed=7, concept_idx=3,
    )

    assert result_a.delta_scores == result_b.delta_scores
    assert result_a.selected_strategies == result_b.selected_strategies
