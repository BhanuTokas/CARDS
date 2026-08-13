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
from cards.concepts.prompts import build_concept_query
from cards.data.datasets import load_cifar, load_cub, load_metadataset
from cards.directions.estimate import ConceptDirection, estimate_direction
from cards.directions.orthogonalize import lowdin_orthogonalize
from cards.encoders.base import ImageTextEncoder
from cards.encoders.open_clip_encoder import OpenClipEncoder
from cards.retrieval.confound import matched_retrieval, stratified_retrieval
from cards.retrieval.pool import CandidatePool
from cards.retrieval.retrieve import retrieve_top_bottom_k
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
    """Steps 2/3: dispatches to naive / matched / stratified retrieval per
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
    raise ValueError(f"unknown retrieval strategy {strategy!r}")


def process_concept(
    cfg: DictConfig,
    encoder: ImageTextEncoder,
    pool: CandidatePool,
    concept: str,
) -> ConceptResult:
    """Steps 1, 2/3, 4 for a single concept."""
    t_c = build_concept_query(concept, encoder)
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

    encoder = OpenClipEncoder(cfg.encoder.model_name, cfg.encoder.pretrained, device=cfg.device)

    pairs = load_dataset_pool(cfg)
    pool = CandidatePool.from_pairs(pairs, encoder)
    log.info(
        "Candidate pool: %d images from dataset=%s split=%s",
        len(pool.paths),
        cfg.dataset.name,
        cfg.pool_source,
    )

    results = [process_concept(cfg, encoder, pool, concept) for concept in concepts]

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
