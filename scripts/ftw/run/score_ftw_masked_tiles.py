"""CPU-only stage of the two-job HPC split: `ftw_tools.cli inference run`
on every orig/masked GeoTIFF pair written by scripts/ftw/build/
build_ftw_masked_tiles.py (the GPU stage), computing per-concept field-
channel deltas and the final raw_score CSV.

Zero encoder/GPU dependency by construction: imports only run_
conceptmask_ftw_bigearthnet's CLI-invocation helper (run_ftw_inference,
via score_jobs) and never instantiates an encoder or touches torch's CUDA
path. Safe to run on a CPU-only allocation -- see notes v10 and the
module docstrings of run_conceptmask_ftw_bigearthnet.py /
build_ftw_masked_tiles.py for the full split rationale.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import run_conceptmask_ftw_bigearthnet as ftw_run

MASKED_TILES_DIR = Path(os.environ.get("FTW_MASKED_TILES_DIR", "ftw_masked_tiles"))


def main():
    manifest_path = MASKED_TILES_DIR / "manifest.csv"
    summary_path = MASKED_TILES_DIR / "concept_summary.csv"
    if not manifest_path.exists() or not summary_path.exists():
        raise SystemExit(
            f"{manifest_path} / {summary_path} not found -- run "
            f"scripts/ftw/build/build_ftw_masked_tiles.py (GPU job) first, with the same "
            f"FTW_MASKED_TILES_DIR ({MASKED_TILES_DIR})."
        )

    with open(manifest_path, newline="") as f:
        manifest_rows = list(csv.DictReader(f))
    with open(summary_path, newline="") as f:
        summary_by_name = {row["concept_name"]: row for row in csv.DictReader(f)}

    by_concept = defaultdict(list)  # preserves concept order: manifest was written in BIGEARTHNET_19_CLASSES order
    for row in manifest_rows:
        by_concept[row["concept_name"]].append((int(row["tile_idx"]), Path(row["orig_tif"]), Path(row["masked_tif"])))

    print(f"{len(manifest_rows)} masked-tile jobs across {len(by_concept)} concepts, "
          f"N_PARALLEL_INFERENCE={ftw_run.N_PARALLEL_INFERENCE}.", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=ftw_run.N_PARALLEL_INFERENCE) as executor:
        for concept_idx, (concept_name, jobs) in enumerate(by_concept.items()):
            deltas, n_failed_inference = ftw_run.score_jobs(jobs, MASKED_TILES_DIR, executor)
            raw_score = float(np.mean(deltas)) if deltas else float("nan")
            summary = summary_by_name.get(concept_name, {})
            rows.append({
                "concept_name": concept_name, "raw_score": raw_score,
                "n_present": summary.get("n_present", ""), "n_scored": len(deltas),
                "n_skipped_degenerate": summary.get("n_skipped_degenerate", ""),
                "n_failed_inference": n_failed_inference,
            })
            print(f"[{concept_idx + 1}/{len(by_concept)}] {concept_name:<70s} "
                  f"raw_score={raw_score:+.4f} (n_scored={len(deltas)}, n_failed_inference={n_failed_inference})", flush=True)

    out_name = os.environ.get("FTW_OUTPUT_NAME", "conceptmask_ftw_bigearthnet_full.csv")
    ftw_run.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    out_path = ftw_run.RESULTS_DIR / out_name
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["concept_name", "raw_score", "n_present", "n_scored",
                                                "n_skipped_degenerate", "n_failed_inference"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} concept scores to {out_path}", flush=True)


if __name__ == "__main__":
    main()
