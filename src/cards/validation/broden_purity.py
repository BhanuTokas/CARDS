"""Broden ground-truth retrieval-purity validation (design doc Section 2,
item 5): does CARDS' Step 1-2 CLIP retrieval agree with Broden's
already-labeled positives/negatives for a concept?

Two things live here:
  - purity metrics (precision@k / negative-recall@k / average precision)
    against ground truth, for setting a per-concept reliability gate
    (mirroring CCE's 0.7-accuracy filter) and comparing retrieval
    strategies.
  - label-quality flagging: surfaces the ground-truth positives CLIP ranks
    least like the concept, and the negatives it ranks most like the
    concept, as candidates for manual review. This caught real labeling
    errors in the local Broden copy during initial validation (several
    `air_conditioner`-labeled images were actually airport control towers).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from cards.concepts.prompts import build_concept_query
from cards.data.datasets import list_broden_concepts, load_broden
from cards.encoders.base import ImageTextEncoder
from cards.retrieval.pool import CandidatePool
from cards.retrieval.retrieve import retrieve_top_bottom_k


@dataclass
class PurityResult:
    concept: str
    n_positives: int
    n_negatives: int
    precision_at_k: float
    negative_recall_at_k: float
    average_precision: float


@dataclass
class LabelFlag:
    concept: str
    flag: str  # "labeled_positive_looks_unlike_concept" | "labeled_negative_looks_like_concept"
    path: Path
    rank: int
    pool_size: int
    similarity: float


def _concept_prompt_text(concept: str) -> str:
    """Broden concept folder names use underscores, and scene concepts carry
    a trailing "_s" tag (Broden/ADE20k convention, e.g. "bathroom_s") that
    isn't part of the word. Left as-is, `build_concept_query` would prompt
    CLIP with literal nonsense like "a photo of bathroom_s" instead of
    "a photo of bathroom" -- this showed up as anomalously low purity for
    every "_s" scene concept until caught during initial validation.
    """
    text = concept.removesuffix("_s")
    return text.replace("_", " ")


def _build_concept_pool(
    root: Path,
    concept: str,
    encoder: ImageTextEncoder,
) -> tuple[CandidatePool, torch.Tensor]:
    positives, negatives = load_broden(root, concept)
    if not positives or not negatives:
        raise ValueError(f"concept {concept!r} has no positives or no negatives under {root}")

    pairs = [(path, 1) for path in positives] + [(path, 0) for path in negatives]
    pool = CandidatePool.from_pairs(pairs, encoder)
    t_c = build_concept_query(_concept_prompt_text(concept), encoder)
    return pool, t_c


def purity_metrics(pool: CandidatePool, t_c: torch.Tensor) -> tuple[float, float, float]:
    """precision@k / negative-recall@k (k = min(n_pos, n_neg), via Step 2's
    naive top-k/bottom-k) and average precision over the full similarity
    ranking, against pool.labels as ground truth (1 = positive, 0 = negative).
    """
    true_labels = torch.tensor(pool.labels)
    n_positives_total = int((true_labels == 1).sum().item())
    n_negatives_total = int((true_labels == 0).sum().item())
    if n_positives_total == 0 or n_negatives_total == 0:
        raise ValueError(
            f"pool must have at least one positive and one negative label "
            f"(got {n_positives_total} positive, {n_negatives_total} negative)"
        )
    k = min(n_positives_total, n_negatives_total)

    present_indices, absent_indices = retrieve_top_bottom_k(pool, t_c, k)
    precision_at_k = (true_labels[present_indices] == 1).float().mean().item()
    negative_recall_at_k = (true_labels[absent_indices] == 0).float().mean().item()

    similarities = pool.embeddings @ t_c
    order = torch.argsort(similarities, descending=True)
    ranked_labels = true_labels[order]
    cumulative_true_positives = torch.cumsum(ranked_labels, dim=0).float()
    precision_at_each_rank = cumulative_true_positives / torch.arange(1, len(ranked_labels) + 1)
    average_precision = (precision_at_each_rank * ranked_labels).sum().item() / n_positives_total

    return precision_at_k, negative_recall_at_k, average_precision


def check_concept_purity(root: Path, concept: str, encoder: ImageTextEncoder) -> PurityResult:
    pool, t_c = _build_concept_pool(root, concept, encoder)
    precision_at_k, negative_recall_at_k, average_precision = purity_metrics(pool, t_c)
    n_positives = sum(1 for label in pool.labels if label == 1)
    n_negatives = len(pool.labels) - n_positives
    return PurityResult(
        concept=concept,
        n_positives=n_positives,
        n_negatives=n_negatives,
        precision_at_k=precision_at_k,
        negative_recall_at_k=negative_recall_at_k,
        average_precision=average_precision,
    )


def check_all_concepts_purity(
    root: Path,
    encoder: ImageTextEncoder,
    concepts: list[str] | None = None,
) -> list[PurityResult]:
    concepts = concepts if concepts is not None else list_broden_concepts(root)
    results = []
    for concept in concepts:
        try:
            results.append(check_concept_purity(root, concept, encoder))
        except ValueError:
            continue  # no positives or no negatives for this concept -- skip
    return results


def flag_suspect_labels(
    pool: CandidatePool,
    t_c: torch.Tensor,
    concept: str,
    flag_fraction: float = 0.15,
) -> list[LabelFlag]:
    """The most-disagreed-with `flag_fraction` of each side: ground-truth
    positives CLIP ranks least like the concept, and ground-truth negatives
    it ranks most like the concept."""
    if not 0 < flag_fraction <= 1:
        raise ValueError(f"flag_fraction must be in (0, 1], got {flag_fraction}")

    similarities = pool.embeddings @ t_c
    order = torch.argsort(similarities, descending=True)
    rank_of = torch.empty(len(pool.paths), dtype=torch.long)
    rank_of[order] = torch.arange(len(pool.paths))

    positive_indices = [i for i in range(len(pool.paths)) if pool.labels[i] == 1]
    negative_indices = [i for i in range(len(pool.paths)) if pool.labels[i] == 0]

    n_flag_positive = max(1, round(flag_fraction * len(positive_indices))) if positive_indices else 0
    n_flag_negative = max(1, round(flag_fraction * len(negative_indices))) if negative_indices else 0

    suspect_positives = sorted(positive_indices, key=lambda i: rank_of[i].item(), reverse=True)[:n_flag_positive]
    suspect_negatives = sorted(negative_indices, key=lambda i: rank_of[i].item())[:n_flag_negative]

    flags = []
    for i in suspect_positives:
        flags.append(
            LabelFlag(
                concept=concept,
                flag="labeled_positive_looks_unlike_concept",
                path=pool.paths[i],
                rank=rank_of[i].item(),
                pool_size=len(pool.paths),
                similarity=similarities[i].item(),
            )
        )
    for i in suspect_negatives:
        flags.append(
            LabelFlag(
                concept=concept,
                flag="labeled_negative_looks_like_concept",
                path=pool.paths[i],
                rank=rank_of[i].item(),
                pool_size=len(pool.paths),
                similarity=similarities[i].item(),
            )
        )
    return flags


def flag_concept_labels(
    root: Path,
    concept: str,
    encoder: ImageTextEncoder,
    flag_fraction: float = 0.15,
) -> list[LabelFlag]:
    pool, t_c = _build_concept_pool(root, concept, encoder)
    return flag_suspect_labels(pool, t_c, concept, flag_fraction)


def flag_all_concepts_labels(
    root: Path,
    encoder: ImageTextEncoder,
    concepts: list[str] | None = None,
    flag_fraction: float = 0.15,
) -> list[LabelFlag]:
    concepts = concepts if concepts is not None else list_broden_concepts(root)
    flags = []
    for concept in concepts:
        try:
            flags.extend(flag_concept_labels(root, concept, encoder, flag_fraction))
        except ValueError:
            continue  # no positives or no negatives for this concept -- skip
    return flags
