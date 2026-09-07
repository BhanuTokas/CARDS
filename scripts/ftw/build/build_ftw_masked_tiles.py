"""GPU stage of the two-job HPC split: retrieval + localization + best-of-
family masking for every BigEarthNet concept, writing persistent orig/
masked GeoTIFF pairs plus a manifest that scripts/ftw/run/
score_ftw_masked_tiles.py (CPU-only) consumes for the ftw_tools CLI
inference stage.

Split out because concept masking's own encoder calls -- `localize_concept`
(patch-similarity) and `encode_images` (best-of-family fill selection,
in run_conceptmask_ftw_bigearthnet.build_masked_tiles_for_concept) -- are
real batched neural-net forward passes that benefit from GPU, UNLIKE the
`ftw_tools.cli inference run` subprocess calls (measured zero GPU benefit,
see run_conceptmask_ftw_bigearthnet's module docstring). Direct feedback:
"wouldn't concept masking also require GPU as we verify which type of
masking works best?" -- correct; this is the corrected split (notes v10),
replacing an earlier, wrong assumption that only the initial tile-pool
embedding needed GPU.

Run this FIRST (GPU job), then scripts/ftw/run/score_ftw_masked_tiles.py
(CPU-only job) on the manifest this writes.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from omegaconf import OmegaConf

import run_conceptmask_ftw_bigearthnet as ftw_run
from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.bigearthnet import BIGEARTHNET_19_CLASSES
from cards.pipeline import instantiate_encoder, orthogonalize_queries

MASKED_TILES_DIR = Path(os.environ.get("FTW_MASKED_TILES_DIR", "ftw_masked_tiles"))


def main():
    MASKED_TILES_DIR.mkdir(parents=True, exist_ok=True)
    tile_paths = ftw_run.list_local_tiles()
    print(f"{len(tile_paths)} FTW tiles total, K={ftw_run.K}.", flush=True)

    cfg = OmegaConf.create({
        "seed": ftw_run.SEED, "device": ftw_run.DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": ftw_run.DEVICE},
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    pool = ftw_run.build_pool(tile_paths, encoder)
    print(f"pool embeddings: {pool.embeddings.shape}", flush=True)

    # Best-known config (CelebA track: z-score alpha=1.0/orth=True/SigLIP) --
    # queries are jointly orthogonalized BEFORE retrieval, all 19 at once.
    raw_queries = {
        c: demean_query(build_concept_query(f"a satellite image of {c.lower()}", encoder), text_center)
        for c in BIGEARTHNET_19_CLASSES
    }
    queries = orthogonalize_queries(raw_queries)

    manifest_rows = []
    summary_rows = []
    for concept_idx, concept_name in enumerate(BIGEARTHNET_19_CLASSES):
        t_c = queries[concept_name]
        jobs, n_skipped_degenerate, n_present = ftw_run.build_masked_tiles_for_concept(
            pool, encoder, concept_idx, concept_name, t_c, MASKED_TILES_DIR)
        for idx, orig_tif, masked_tif in jobs:
            manifest_rows.append({
                "concept_idx": concept_idx, "concept_name": concept_name, "tile_idx": idx,
                "orig_tif": str(orig_tif), "masked_tif": str(masked_tif),
            })
        summary_rows.append({
            "concept_idx": concept_idx, "concept_name": concept_name,
            "n_present": n_present, "n_masked": len(jobs), "n_skipped_degenerate": n_skipped_degenerate,
        })
        print(f"[{concept_idx + 1}/{len(BIGEARTHNET_19_CLASSES)}] {concept_name:<70s} "
              f"n_present={n_present} n_masked={len(jobs)} n_skipped_degenerate={n_skipped_degenerate}", flush=True)

    manifest_path = MASKED_TILES_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["concept_idx", "concept_name", "tile_idx", "orig_tif", "masked_tif"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_path = MASKED_TILES_DIR / "concept_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["concept_idx", "concept_name", "n_present", "n_masked", "n_skipped_degenerate"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nWrote {len(manifest_rows)} masked-tile jobs across {len(BIGEARTHNET_19_CLASSES)} concepts "
          f"to {manifest_path} (+ {summary_path}).", flush=True)


if __name__ == "__main__":
    main()
