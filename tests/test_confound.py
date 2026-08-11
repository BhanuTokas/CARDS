"""Step 3 tests: confound-matched (projection + nearest-neighbor) and
stratified retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from cards.retrieval.confound import matched_retrieval, stratified_retrieval
from cards.retrieval.pool import CandidatePool


def test_matched_retrieval_matches_on_confound_not_concept():
    # dim 0 is the concept axis (t_c); dims 1-2 are confound axes. idx2/idx3
    # share their confound with idx0/idx1 respectively but sit on the
    # opposite side of the concept axis -- exactly the "concept flipped,
    # everything else held fixed" match we want.
    embeddings = torch.tensor(
        [
            [5.0, 1.0, 0.0],  # idx0: present, confound=(1,0)
            [5.0, 0.0, 2.0],  # idx1: present, confound=(0,2)
            [-5.0, 1.0, 0.0],  # idx2: candidate, matches idx0's confound
            [-5.0, 0.0, 2.0],  # idx3: candidate, matches idx1's confound
            [-5.0, -1.0, -2.0],  # idx4: candidate, matches neither
        ]
    )
    pool = CandidatePool(paths=[Path(f"img_{i}.jpg") for i in range(5)], embeddings=embeddings)
    t_c = torch.tensor([1.0, 0.0, 0.0])

    absent_indices = matched_retrieval(pool, present_indices=[0, 1], t_c=t_c)

    assert absent_indices == [2, 3]


def test_matched_retrieval_excludes_present_indices():
    generator = torch.Generator().manual_seed(0)
    embeddings = F.normalize(torch.randn(8, 4, generator=generator), dim=-1)
    pool = CandidatePool(paths=[Path(f"img_{i}.jpg") for i in range(8)], embeddings=embeddings)
    t_c = F.normalize(torch.randn(4, generator=torch.Generator().manual_seed(1)), dim=0)
    present_indices = [0, 1, 2]

    absent_indices = matched_retrieval(pool, present_indices=present_indices, t_c=t_c)

    assert set(absent_indices).isdisjoint(present_indices)
    assert len(absent_indices) == len(present_indices)


def test_matched_retrieval_raises_when_no_candidates_remain():
    embeddings = torch.randn(3, 4)
    pool = CandidatePool(paths=[Path(f"img_{i}.jpg") for i in range(3)], embeddings=embeddings)

    with pytest.raises(ValueError):
        matched_retrieval(pool, present_indices=[0, 1, 2], t_c=torch.randn(4))


def test_stratified_retrieval_respects_class_strata():
    n_per_class = 6
    dim = 4
    k = 2
    generator = torch.Generator().manual_seed(1)
    embeddings = F.normalize(torch.randn(2 * n_per_class, dim, generator=generator), dim=-1)
    labels = [0] * n_per_class + [1] * n_per_class
    pool = CandidatePool(
        paths=[Path(f"img_{i}.jpg") for i in range(2 * n_per_class)],
        embeddings=embeddings,
        labels=labels,
    )
    t_c = F.normalize(torch.randn(dim, generator=torch.Generator().manual_seed(2)), dim=0)

    present, absent = stratified_retrieval(pool, t_c, k=k)

    assert len(present) == len(absent) == k * 2
    labels_tensor = torch.tensor(labels)
    present_labels = labels_tensor[torch.tensor(present)]
    absent_labels = labels_tensor[torch.tensor(absent)]
    for class_label in (0, 1):
        assert (present_labels == class_label).sum().item() == k
        assert (absent_labels == class_label).sum().item() == k

    for class_label in (0, 1):
        class_indices = (labels_tensor == class_label).nonzero(as_tuple=True)[0]
        similarities = embeddings[class_indices] @ t_c
        order = torch.argsort(similarities, descending=True)
        expected_present = set(class_indices[order[:k]].tolist())
        expected_absent = set(class_indices[order[-k:]].tolist())
        actual_present = {i for i in present if labels[i] == class_label}
        actual_absent = {i for i in absent if labels[i] == class_label}
        assert actual_present == expected_present
        assert actual_absent == expected_absent


def test_stratified_retrieval_requires_labels():
    pool = CandidatePool(paths=[Path("a.jpg")], embeddings=torch.randn(1, 4))

    with pytest.raises(ValueError):
        stratified_retrieval(pool, torch.randn(4), k=1)
