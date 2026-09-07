"""One-time SigLIP embedding precompute for the FTW tile pool, split out
from run_conceptmask_ftw_bigearthnet.py's main() so the short, real-batched-
compute embedding step can run on a GPU allocation while the long
per-concept masking + `ftw_tools.cli inference run` loop runs on a
separate, cheaper CPU-only allocation.

Rationale, direct feedback: "wouldn't majority of time be spent on compute
that doesn't use GPU? That seems wasteful" -- correct. The CLI inference
phase is CPU-only regardless (measured no GPU speedup there, notes v7) and
scales with K x concepts x 2, so it dominates wall time; embedding is a
one-time pass over the whole tile pool and is comparatively short. Holding
a GPU node for the entire multi-hour job to cover only this short step
wastes it -- see notes v9.

Run this FIRST (with FTW_EMBEDDING_CACHE set), then run
run_conceptmask_ftw_bigearthnet.py (same env var) on a separate CPU-only
allocation -- it loads the cache instead of recomputing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from omegaconf import OmegaConf

import run_conceptmask_ftw_bigearthnet as ftw_run
from cards.concepts.prompts import compute_text_center, GENERIC_REFERENCE_CONCEPTS
from cards.pipeline import instantiate_encoder


def main():
    if ftw_run.EMBEDDING_CACHE is None:
        raise SystemExit(
            "FTW_EMBEDDING_CACHE is not set -- this script only makes sense when caching "
            "to a path the main run script will later load. Set it to the same value for both."
        )

    tile_paths = ftw_run.list_local_tiles()
    print(f"{len(tile_paths)} FTW tiles total.", flush=True)

    cfg = OmegaConf.create({
        "seed": ftw_run.SEED, "device": ftw_run.DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": ftw_run.DEVICE},
    })
    encoder = instantiate_encoder(cfg)
    # Not strictly needed for embedding alone (only affects query building
    # downstream), computed here anyway so any encoder-load issues surface
    # in this short job rather than the long one.
    compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    pool = ftw_run.build_pool(tile_paths, encoder)
    print(f"Done. pool embeddings: {pool.embeddings.shape} -> {ftw_run.EMBEDDING_CACHE}", flush=True)


if __name__ == "__main__":
    main()
