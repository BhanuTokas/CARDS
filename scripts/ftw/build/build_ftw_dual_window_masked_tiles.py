"""GPU stage of the dual-window PRUE two-job HPC split -- mirrors
scripts/ftw/build/build_ftw_masked_tiles.py (the single-window version)
exactly, swapped to the dual-window pipeline's own functions (combined-
embedding pool, per-window independent localization, 4-condition
masking). Retrieval + localization + best-of-family masking are all real
SigLIP forward passes that benefit from GPU, same reasoning as the
single-window split (notes v9/v10) -- only the `ftw_tools.cli inference
run` calls (score stage) are CPU-only-fine.

Writes 4 GeoTIFFs per non-degenerate present location (orig, mask_a,
mask_b, mask_both) plus a manifest + concept_summary to
FTW_DUAL_MASKED_TILES_DIR. Run scripts/ftw/run/score_ftw_dual_window_
masked_tiles.py (CPU-only) on that manifest AFTER this job completes.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from omegaconf import OmegaConf

import run_conceptmask_ftw_dual_window as ftw_dual
from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.bigearthnet import BIGEARTHNET_19_CLASSES
from cards.pipeline import instantiate_encoder, orthogonalize_queries

MASKED_TILES_DIR = Path(os.environ.get("FTW_DUAL_MASKED_TILES_DIR", "ftw_dual_window_masked_tiles"))


def main():
    MASKED_TILES_DIR.mkdir(parents=True, exist_ok=True)
    tile_paths = ftw_dual.list_local_window_a_tiles()
    print(f"{len(tile_paths)} dual-window-capable locations, K={ftw_dual.K}.", flush=True)

    cfg = OmegaConf.create({
        "seed": ftw_dual.SEED, "device": ftw_dual.DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": ftw_dual.DEVICE},
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    pool = ftw_dual.build_pool(tile_paths, encoder)
    print(f"pool embeddings: {pool.embeddings.shape}", flush=True)

    raw_queries = {
        c: demean_query(build_concept_query(f"a satellite image of {c.lower()}", encoder), text_center)
        for c in BIGEARTHNET_19_CLASSES
    }
    queries = orthogonalize_queries(raw_queries)

    manifest_rows = []
    summary_rows = []
    for concept_idx, concept_name in enumerate(BIGEARTHNET_19_CLASSES):
        t_c = queries[concept_name]
        jobs, n_skipped_degenerate, n_present = ftw_dual.build_masked_tiles_for_concept(
            pool, encoder, concept_idx, concept_name, t_c, MASKED_TILES_DIR)
        for idx, orig_tif, mask_a_tif, mask_b_tif, mask_both_tif in jobs:
            manifest_rows.append({
                "concept_idx": concept_idx, "concept_name": concept_name, "tile_idx": idx,
                "orig_tif": str(orig_tif), "mask_a_tif": str(mask_a_tif),
                "mask_b_tif": str(mask_b_tif), "mask_both_tif": str(mask_both_tif),
            })
        summary_rows.append({
            "concept_idx": concept_idx, "concept_name": concept_name,
            "n_present": n_present, "n_masked": len(jobs), "n_skipped_degenerate": n_skipped_degenerate,
        })
        print(f"[{concept_idx + 1}/{len(BIGEARTHNET_19_CLASSES)}] {concept_name:<70s} "
              f"n_present={n_present} n_masked={len(jobs)} n_skipped_degenerate={n_skipped_degenerate}", flush=True)

    manifest_path = MASKED_TILES_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "concept_idx", "concept_name", "tile_idx", "orig_tif", "mask_a_tif", "mask_b_tif", "mask_both_tif",
        ])
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
