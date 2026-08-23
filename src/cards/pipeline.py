"""End-to-end orchestration: wires configs/config.yaml's dataset/encoder/
retrieval/normalization/model selections to the Steps 1-7 pipeline.

Kept separate from scripts/run_attribution.py so each piece (dataset
dispatch, retrieval-strategy dispatch, normalization dispatch, ...) is
unit-testable via normal imports rather than only through the Hydra CLI
entrypoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image

from cards.attribution.global_mode import global_score
from cards.attribution.normalization import (
    angular_distance,
    embedding_distance_normalize,
    variance_normalize,
)
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.datasets import load_cifar, load_cub, load_metadataset
from cards.directions.estimate import ConceptDirection, estimate_direction
from cards.directions.orthogonalize import lowdin_orthogonalize
from cards.encoders.base import ImageTextEncoder
from cards.retrieval.aligned import aligned_retrieval
from cards.retrieval.confound import matched_retrieval, stratified_retrieval
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.pool import CandidatePool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.retrieval.symmetric_aligned import symmetric_aligned_retrieval
from cards.utils.seed import set_seed

log = logging.getLogger(__name__)


@dataclass
class ConceptResult:
    direction: ConceptDirection
    present_indices: list[int]
    absent_indices: list[int]


def load_dataset_pool(cfg: DictConfig) -> list[tuple[Path, int]]:
    """Resolves cfg.dataset + cfg.pool_source to a (path, label) pool.

    Broden isn't included: the local copy is ground-truth positives/
    negatives per concept for the retrieval-purity validation check (design
    doc Section 2, item 5), not a generic pool -- call
    cards.data.datasets.load_broden directly for that experiment instead.
    """
    name = cfg.dataset.name
    if name in ("cifar10", "cifar100"):
        return load_cifar(Path(cfg.dataset.root), variant=cfg.dataset.variant, split=cfg.pool_source)
    if name == "cub":
        return load_cub(Path(cfg.dataset.root), split=cfg.pool_source)
    if name == "metadataset":
        return load_metadataset(
            Path(cfg.dataset.root), scenario=cfg.dataset.scenario, split=cfg.pool_source
        )
    raise ValueError(
        f"dataset {name!r} isn't usable as a generic retrieval pool through this pipeline "
        f"(broden is ground-truth pairs, not a pool -- see load_broden)"
    )


def retrieve_concept_sets(
    cfg: DictConfig,
    pool: CandidatePool,
    t_c: torch.Tensor,
) -> tuple[list[int], list[int]]:
    """Steps 2/3: dispatches to naive / matched / stratified / aligned /
    aligned_symmetric / aligned_symmetric_constrained retrieval per
    cfg.retrieval.strategy."""
    strategy = cfg.retrieval.strategy
    if strategy == "naive":
        return retrieve_top_bottom_k(pool, t_c, cfg.k)
    if strategy == "matched":
        present_indices, _ = retrieve_top_bottom_k(pool, t_c, cfg.k)
        absent_indices = matched_retrieval(pool, present_indices, t_c)
        return present_indices, absent_indices
    if strategy == "stratified":
        return stratified_retrieval(pool, t_c, cfg.k)
    if strategy == "aligned":
        present_indices, _ = retrieve_top_bottom_k(pool, t_c, cfg.k)
        absent_indices = aligned_retrieval(pool, present_indices, t_c, cfg.k)
        return present_indices, absent_indices
    if strategy == "aligned_symmetric":
        return symmetric_aligned_retrieval(pool, t_c, cfg.k)
    if strategy == "aligned_symmetric_constrained":
        pool_multiplier = cfg.retrieval.get("present_pool_multiplier", 3)
        return symmetric_aligned_retrieval(
            pool, t_c, cfg.k, present_candidate_pool_size=pool_multiplier * cfg.k
        )
    raise ValueError(f"unknown retrieval strategy {strategy!r}")


def resolve_demean_reference_concepts(cfg: DictConfig, concepts: list[str]) -> list[str]:
    """Which concepts to compute the demean_query text center from:
    cfg.demean_reference_concepts if given, else `concepts` itself if
    that's a big enough sample (>=10, compute_text_center's own floor),
    else CARDS' built-in generic vocabulary -- logged loudly every time,
    since it's a generic stand-in for the run's own concept bank, not a
    substitute for one.
    """
    if cfg.demean_reference_concepts:
        return list(cfg.demean_reference_concepts)
    if len(concepts) >= 10:
        return concepts
    log.warning(
        "demean_query is enabled but only %d concept(s) were given (%s) and "
        "no demean_reference_concepts was set -- falling back to CARDS' "
        "built-in generic reference vocabulary (%d concepts) to compute the "
        "de-meaning text center. This is a generic stand-in, not specific to "
        "this run's concept bank; pass demean_reference_concepts explicitly "
        "(e.g. the full concept bank this run's concepts are drawn from) for "
        "a better estimate.",
        len(concepts),
        concepts,
        len(GENERIC_REFERENCE_CONCEPTS),
    )
    return GENERIC_REFERENCE_CONCEPTS


def process_concept(
    cfg: DictConfig,
    encoder: ImageTextEncoder,
    pool: CandidatePool,
    concept: str,
    text_center: torch.Tensor | None = None,
) -> ConceptResult:
    """Steps 1, 2/3, 4 for a single concept.

    `text_center`, when given, de-means the Step 1 query before retrieval
    (see cards.concepts.prompts.demean_query) -- an ablation toggle
    (cfg.demean_query), not always applied, since it changes which
    images get retrieved and therefore isn't free to turn on
    unconditionally when comparing against prior runs/results.
    """
    t_c = build_concept_query(concept, encoder)
    if text_center is not None:
        t_c = demean_query(t_c, text_center)
    present_indices, absent_indices = retrieve_concept_sets(cfg, pool, t_c)
    direction = estimate_direction(
        concept, pool.embeddings[present_indices], pool.embeddings[absent_indices]
    )
    return ConceptResult(direction=direction, present_indices=present_indices, absent_indices=absent_indices)


def compute_delta_c(cfg: DictConfig, pool: CandidatePool, result: ConceptResult) -> float:
    """Delta_c for Step 7's embedding-distance normalization: Euclidean
    reuses Step 4's already-computed magnitude; angular is the sub-ablation
    alternative, recomputed from the P_c/N_c centroids."""
    distance_fn = cfg.normalization.get("distance_fn", "euclidean")
    if distance_fn == "euclidean":
        return result.direction.magnitude
    if distance_fn == "angular":
        present_centroid = pool.embeddings[result.present_indices].mean(dim=0)
        absent_centroid = pool.embeddings[result.absent_indices].mean(dim=0)
        return angular_distance(present_centroid, absent_centroid)
    raise ValueError(f"unknown distance_fn {distance_fn!r}")


def normalize_score(
    cfg: DictConfig,
    raw_score: float,
    present_outputs: torch.Tensor,
    absent_outputs: torch.Tensor,
    delta_c: float,
) -> float:
    """Step 7."""
    method = cfg.normalization.method
    if method == "variance":
        return variance_normalize(present_outputs, absent_outputs)
    if method == "embedding_distance":
        return embedding_distance_normalize(raw_score, delta_c)
    raise ValueError(f"unknown normalization method {method!r}")


def instantiate_encoder(cfg: DictConfig) -> ImageTextEncoder:
    """Step 1/2 encoder, pluggable via cfg.encoder's `_target_` -- same
    strip-`name`-then-hydra.utils.instantiate pattern as
    instantiate_model, so adding a new encoder (e.g. Perception Encoder)
    is a new configs/encoder/*.yaml, not a code change here.
    """
    encoder_cfg = {k: v for k, v in cfg.encoder.items() if k != "name"}
    return hydra.utils.instantiate(encoder_cfg)


def instantiate_model(cfg: DictConfig):
    """Steps 6-7 black-box model, or None if cfg.model.name == 'none'.

    cfg.model carries a plain `name` field (used only for this check)
    alongside the _target_-based instantiation spec; strip it before handing
    the rest to hydra.utils.instantiate, so it isn't passed through as an
    unexpected constructor kwarg.
    """
    if cfg.model.name == "none":
        return None
    # Only `name` is stripped -- this couples the config schema to today's
    # single non-_target_ metadata field. A future model config that adds
    # another plain (non-constructor) field alongside `name` would need the
    # same treatment here.
    model_cfg = {k: v for k, v in cfg.model.items() if k != "name"}
    return hydra.utils.instantiate(model_cfg)


def load_and_preprocess(paths: list[Path], black_box) -> torch.Tensor:
    """Loads raw images and runs the black-box model's own preprocessing --
    a different transform than the CLIP preprocessing used for retrieval,
    since the black-box model may use a different backbone entirely."""
    return torch.stack([black_box.preprocess(Image.open(path).convert("RGB")) for path in paths])


def save_directions(results: list[ConceptResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            result.direction.concept: {
                "unit_vector": result.direction.unit_vector,
                "magnitude": result.direction.magnitude,
            }
            for result in results
        },
        path,
    )


def run(cfg: DictConfig) -> list[ConceptResult]:
    set_seed(cfg.seed)
    log.info("Config:\n%s", cfg)

    concepts = list(cfg.concepts)
    if not concepts:
        raise ValueError("cfg.concepts must be a non-empty list")

    encoder = instantiate_encoder(cfg)

    pairs = load_dataset_pool(cfg)
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    log.info(
        "Candidate pool: %d images from dataset=%s split=%s",
        len(pool.paths),
        cfg.dataset.name,
        cfg.pool_source,
    )

    text_center = None
    if cfg.demean_query:
        reference_concepts = resolve_demean_reference_concepts(cfg, concepts)
        text_center = compute_text_center(reference_concepts, encoder)
        log.info(
            "demean_query enabled: text center computed from %d reference concepts",
            len(reference_concepts),
        )

    results = [
        process_concept(cfg, encoder, pool, concept, text_center=text_center) for concept in concepts
    ]

    if cfg.orthogonalize and len(results) > 1:
        orthogonalized = lowdin_orthogonalize([r.direction for r in results])
        for result, direction in zip(results, orthogonalized):
            result.direction = direction

    output_dir = Path(cfg.output_dir)
    save_directions(results, output_dir / "directions.pt")
    for result in results:
        log.info("concept=%s magnitude=%.4f", result.direction.concept, result.direction.magnitude)

    black_box = instantiate_model(cfg)
    if black_box is None:
        log.info("model=none -- skipping Steps 6-7 (attribution scoring); directions saved to %s", output_dir)
        return results

    for concept, result in zip(concepts, results):
        present_images = load_and_preprocess([pool.paths[i] for i in result.present_indices], black_box)
        absent_images = load_and_preprocess([pool.paths[i] for i in result.absent_indices], black_box)
        raw_score, present_outputs, absent_outputs = global_score(black_box, present_images, absent_images)
        delta_c = compute_delta_c(cfg, pool, result)
        normalized_score = normalize_score(cfg, raw_score, present_outputs, absent_outputs, delta_c)
        log.info("concept=%s raw_score=%.4f normalized_score=%.4f", concept, raw_score, normalized_score)

    return results
