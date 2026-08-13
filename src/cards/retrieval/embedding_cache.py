"""On-disk cache for CandidatePool embeddings, keyed on (dataset, split,
encoder) identity.

Hydra multirun sweeps (see notes/ablation_scope_decision.md) launch each
job as a separate process with no shared memory. Retrieval strategy and
normalization are downstream of the embeddings and don't change them, so
without this cache a grid sweeping those axes pays for a full re-encoding
pass on every job even when the (dataset, encoder) pair repeats.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from cards.encoders.base import ImageEncoder
from cards.retrieval.pool import CandidatePool

log = logging.getLogger(__name__)


def cache_key_for(cfg: DictConfig) -> str:
    """Identifies the (dataset, split, encoder) combination that produces a
    given set of embeddings -- independent of retrieval/normalization/model,
    which don't affect the pool."""
    payload = {
        "dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
        "pool_source": cfg.pool_source,
        "encoder": {"model_name": cfg.encoder.model_name, "pretrained": cfg.encoder.pretrained},
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _paths_fingerprint(paths: list[Path]) -> str:
    joined = "\n".join(str(p) for p in paths)
    return hashlib.sha1(joined.encode()).hexdigest()


def load_or_build_pool(
    cache_dir: Path,
    key: str,
    pairs: list[tuple[Path, int]],
    encoder: ImageEncoder,
    batch_size: int = 256,
) -> CandidatePool:
    """CandidatePool.from_pairs, but reuses a cached embeddings tensor from
    a prior call with the same key if the underlying pairs are unchanged.

    Staleness check is a fingerprint of the path list (order included, not
    file contents) rather than a full re-scan or content hash -- cheap, and
    sufficient to catch "the dataset loader's output changed" without
    reading every image again just to validate the cache.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{key}.pt"
    paths = [path for path, _ in pairs]
    labels = [label for _, label in pairs]
    fingerprint = _paths_fingerprint(paths)

    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=True)
        if cached["fingerprint"] == fingerprint:
            log.info("embedding cache hit: %s (%d images)", cache_path, len(paths))
            return CandidatePool(paths=paths, embeddings=cached["embeddings"], labels=labels)
        log.warning("embedding cache at %s is stale (pool contents changed) -- recomputing", cache_path)

    pool = CandidatePool.from_pairs(pairs, encoder, batch_size=batch_size)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"fingerprint": fingerprint, "embeddings": pool.embeddings}, cache_path)
    log.info("embedding cache written: %s (%d images)", cache_path, len(paths))
    return pool
