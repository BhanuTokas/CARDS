"""Tests for cards.validation.broden_purity."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from cards.retrieval.pool import CandidatePool
from cards.validation.broden_purity import (
    PurityResult,
    _concept_prompt_text,
    check_all_concepts_purity,
    check_concept_purity,
    flag_all_concepts_labels,
    flag_concept_labels,
    flag_suspect_labels,
    purity_metrics,
)


def _touch_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2)).save(path)


def _make_broden_concept(root: Path, concept: str, n_pos: int, n_neg: int) -> None:
    for i in range(n_pos):
        _touch_image(root / concept / "positives" / f"p{i}.png")
    for i in range(n_neg):
        _touch_image(root / concept / "negatives" / f"n{i}.png")


class _FixedEncoder:
    """Every text encodes to a fixed query vector; each image encodes per a
    path->vector lookup. Enough to test this module's orchestration --
    Step 1's own prompt-ensembling logic is covered in test_prompts.py."""

    def __init__(self, query_vector: torch.Tensor, image_vectors: dict[str, torch.Tensor]):
        self.query_vector = query_vector
        self.image_vectors = image_vectors

    def encode_text(self, texts):
        return torch.stack([self.query_vector for _ in texts])

    def encode_images(self, images):
        return torch.stack([self.image_vectors[image.filename] for image in images])


# ---- _concept_prompt_text ----


def test_concept_prompt_text_strips_scene_suffix_and_underscores():
    assert _concept_prompt_text("bathroom_s") == "bathroom"
    assert _concept_prompt_text("dining_room_s") == "dining room"


def test_concept_prompt_text_replaces_underscores_only():
    assert _concept_prompt_text("air_conditioner") == "air conditioner"


def test_concept_prompt_text_leaves_plain_words_unchanged():
    assert _concept_prompt_text("dog") == "dog"


# ---- purity_metrics ----


def _swapped_pair_pool() -> tuple[CandidatePool, torch.Tensor]:
    """3 positives / 3 negatives, cosine similarity to t_c strictly ranked:
    idx0(pos,.90) > idx3(neg,.85) > idx1(pos,.80) > idx4(neg,-.80) >
    idx5(neg,-.85) > idx2(pos,-.90) -- one positive looks unlike the
    concept, one negative looks like it."""
    paths = [Path(f"img_{i}.jpg") for i in range(6)]
    embeddings = torch.tensor(
        [
            [0.90, 0.43589],
            [0.80, 0.60000],
            [-0.90, 0.43589],
            [0.85, 0.52678],
            [-0.80, 0.60000],
            [-0.85, 0.52678],
        ]
    )
    labels = [1, 1, 1, 0, 0, 0]
    pool = CandidatePool(paths=paths, embeddings=embeddings, labels=labels)
    t_c = torch.tensor([1.0, 0.0])
    return pool, t_c


def test_purity_metrics_perfect_separation():
    paths = [Path(f"img_{i}.jpg") for i in range(6)]
    embeddings = torch.tensor([[1.0, 0.0]] * 3 + [[-1.0, 0.0]] * 3)
    labels = [1, 1, 1, 0, 0, 0]
    pool = CandidatePool(paths=paths, embeddings=embeddings, labels=labels)
    t_c = torch.tensor([1.0, 0.0])

    precision_at_k, negative_recall_at_k, average_precision = purity_metrics(pool, t_c)

    assert precision_at_k == pytest.approx(1.0)
    assert negative_recall_at_k == pytest.approx(1.0)
    assert average_precision == pytest.approx(1.0)


def test_purity_metrics_rejects_no_positives():
    paths = [Path(f"img_{i}.jpg") for i in range(3)]
    embeddings = torch.tensor([[-1.0, 0.0]] * 3)
    pool = CandidatePool(paths=paths, embeddings=embeddings, labels=[0, 0, 0])

    with pytest.raises(ValueError, match="positive"):
        purity_metrics(pool, torch.tensor([1.0, 0.0]))


def test_purity_metrics_rejects_no_negatives():
    paths = [Path(f"img_{i}.jpg") for i in range(3)]
    embeddings = torch.tensor([[1.0, 0.0]] * 3)
    pool = CandidatePool(paths=paths, embeddings=embeddings, labels=[1, 1, 1])

    with pytest.raises(ValueError, match="negative"):
        purity_metrics(pool, torch.tensor([1.0, 0.0]))


def test_purity_metrics_with_swapped_pair():
    pool, t_c = _swapped_pair_pool()

    precision_at_k, negative_recall_at_k, average_precision = purity_metrics(pool, t_c)

    # top-3: idx0(pos), idx3(neg), idx1(pos) -> 2/3 positive
    assert precision_at_k == pytest.approx(2 / 3)
    # bottom-3: idx4(neg), idx5(neg), idx2(pos) -> 2/3 negative
    assert negative_recall_at_k == pytest.approx(2 / 3)
    # ranked labels [1,0,1,0,0,1] -> AP = (1/1 + 2/3 + 3/6) / 3
    assert average_precision == pytest.approx((1.0 + 2 / 3 + 3 / 6) / 3)


# ---- flag_suspect_labels ----


def test_flag_suspect_labels_flags_worst_fraction_each_side():
    pool, t_c = _swapped_pair_pool()

    flags = flag_suspect_labels(pool, t_c, concept="dog", flag_fraction=0.15)

    assert len(flags) == 2
    by_path = {flag.path: flag for flag in flags}
    assert by_path[pool.paths[2]].flag == "labeled_positive_looks_unlike_concept"
    assert by_path[pool.paths[3]].flag == "labeled_negative_looks_like_concept"


def test_flag_suspect_labels_rejects_invalid_flag_fraction():
    pool = CandidatePool(paths=[Path("a.jpg")], embeddings=torch.tensor([[1.0, 0.0]]), labels=[1])

    with pytest.raises(ValueError):
        flag_suspect_labels(pool, torch.tensor([1.0, 0.0]), "dog", flag_fraction=0.0)
    with pytest.raises(ValueError):
        flag_suspect_labels(pool, torch.tensor([1.0, 0.0]), "dog", flag_fraction=1.5)


# ---- check_concept_purity / check_all_concepts_purity (integration) ----


def test_check_concept_purity_integration(tmp_path):
    _make_broden_concept(tmp_path, "dog", n_pos=2, n_neg=2)
    dog_dir = tmp_path / "dog"
    image_vectors = {}
    for path in sorted((dog_dir / "positives").iterdir()):
        image_vectors[str(path)] = torch.tensor([1.0, 0.0])
    for path in sorted((dog_dir / "negatives").iterdir()):
        image_vectors[str(path)] = torch.tensor([-1.0, 0.0])
    encoder = _FixedEncoder(query_vector=torch.tensor([1.0, 0.0]), image_vectors=image_vectors)

    result = check_concept_purity(tmp_path, "dog", encoder)

    assert isinstance(result, PurityResult)
    assert result.concept == "dog"
    assert result.n_positives == 2
    assert result.n_negatives == 2
    assert result.precision_at_k == pytest.approx(1.0)
    assert result.average_precision == pytest.approx(1.0)


def test_check_concept_purity_raises_without_both_sides(tmp_path):
    _touch_image(tmp_path / "dog" / "positives" / "p0.png")
    (tmp_path / "dog" / "negatives").mkdir(parents=True)  # empty
    encoder = _FixedEncoder(query_vector=torch.tensor([1.0, 0.0]), image_vectors={})

    with pytest.raises(ValueError):
        check_concept_purity(tmp_path, "dog", encoder)


def test_check_all_concepts_purity_skips_broken_concepts(tmp_path):
    _make_broden_concept(tmp_path, "dog", n_pos=2, n_neg=2)
    (tmp_path / "empty_concept" / "positives").mkdir(parents=True)
    (tmp_path / "empty_concept" / "negatives").mkdir(parents=True)

    dog_dir = tmp_path / "dog"
    image_vectors = {}
    for path in sorted((dog_dir / "positives").iterdir()):
        image_vectors[str(path)] = torch.tensor([1.0, 0.0])
    for path in sorted((dog_dir / "negatives").iterdir()):
        image_vectors[str(path)] = torch.tensor([-1.0, 0.0])
    encoder = _FixedEncoder(query_vector=torch.tensor([1.0, 0.0]), image_vectors=image_vectors)

    results = check_all_concepts_purity(tmp_path, encoder)

    assert len(results) == 1
    assert results[0].concept == "dog"


# ---- flag_concept_labels / flag_all_concepts_labels (integration) ----


def test_flag_concept_labels_integration(tmp_path):
    _make_broden_concept(tmp_path, "dog", n_pos=3, n_neg=3)
    dog_dir = tmp_path / "dog"
    pos_paths = sorted((dog_dir / "positives").iterdir())
    neg_paths = sorted((dog_dir / "negatives").iterdir())

    image_vectors = {
        str(pos_paths[0]): torch.tensor([0.90, 0.43589]),
        str(pos_paths[1]): torch.tensor([0.80, 0.60000]),
        str(pos_paths[2]): torch.tensor([-0.90, 0.43589]),  # mislabeled-looking positive
        str(neg_paths[0]): torch.tensor([0.85, 0.52678]),  # mislabeled-looking negative
        str(neg_paths[1]): torch.tensor([-0.80, 0.60000]),
        str(neg_paths[2]): torch.tensor([-0.85, 0.52678]),
    }
    encoder = _FixedEncoder(query_vector=torch.tensor([1.0, 0.0]), image_vectors=image_vectors)

    flags = flag_concept_labels(tmp_path, "dog", encoder, flag_fraction=0.15)

    assert len(flags) == 2
    by_path = {flag.path: flag for flag in flags}
    assert by_path[pos_paths[2]].flag == "labeled_positive_looks_unlike_concept"
    assert by_path[neg_paths[0]].flag == "labeled_negative_looks_like_concept"


def test_flag_all_concepts_labels_skips_broken_concepts(tmp_path):
    _make_broden_concept(tmp_path, "dog", n_pos=2, n_neg=2)
    (tmp_path / "empty_concept" / "positives").mkdir(parents=True)
    (tmp_path / "empty_concept" / "negatives").mkdir(parents=True)

    dog_dir = tmp_path / "dog"
    image_vectors = {}
    for path in sorted((dog_dir / "positives").iterdir()):
        image_vectors[str(path)] = torch.tensor([1.0, 0.0])
    for path in sorted((dog_dir / "negatives").iterdir()):
        image_vectors[str(path)] = torch.tensor([-1.0, 0.0])
    encoder = _FixedEncoder(query_vector=torch.tensor([1.0, 0.0]), image_vectors=image_vectors)

    flags = flag_all_concepts_labels(tmp_path, encoder, flag_fraction=0.5)

    assert len(flags) > 0
    assert all(flag.concept == "dog" for flag in flags)
