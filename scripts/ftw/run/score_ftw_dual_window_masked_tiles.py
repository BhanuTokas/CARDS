"""CPU-only stage of the dual-window PRUE two-job HPC split -- mirrors
scripts/ftw/run/score_ftw_masked_tiles.py (the single-window version)
exactly: `ftw_tools.cli inference run` on every GeoTIFF written by
scripts/ftw/build/build_ftw_dual_window_masked_tiles.py (the GPU stage),
computing three per-concept deltas (mask_a, mask_b, mask_both, each
relative to the shared orig-both baseline). Zero encoder/GPU dependency
by construction -- imports only run_conceptmask_ftw_dual_window's CLI-
invocation + scoring helpers (score_jobs), never instantiates an encoder.

Resumable from the start (the single-window version only grew this after
being interrupted twice mid-run -- notes v11): each concept's row is
written and flushed immediately after that concept finishes, and a rerun
against the same output file skips concepts already present instead of
redoing them.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_conceptmask_ftw_dual_window as ftw_dual

MASKED_TILES_DIR = Path(os.environ.get("FTW_DUAL_MASKED_TILES_DIR", "ftw_dual_window_masked_tiles"))
FIELDNAMES = [
    "concept_name", "raw_score_mask_a", "raw_score_mask_b", "raw_score_mask_both",
    "n_present", "n_masked", "n_skipped_degenerate", "n_failed_inference",
]


def main():
    manifest_path = MASKED_TILES_DIR / "manifest.csv"
    summary_path = MASKED_TILES_DIR / "concept_summary.csv"
    if not manifest_path.exists() or not summary_path.exists():
        raise SystemExit(
            f"{manifest_path} / {summary_path} not found -- run "
            f"scripts/ftw/build/build_ftw_dual_window_masked_tiles.py (GPU job) first, with the same "
            f"FTW_DUAL_MASKED_TILES_DIR ({MASKED_TILES_DIR})."
        )

    with open(manifest_path, newline="") as f:
        manifest_rows = list(csv.DictReader(f))
    with open(summary_path, newline="") as f:
        summary_by_name = {row["concept_name"]: row for row in csv.DictReader(f)}

    by_concept = defaultdict(list)  # preserves concept order: manifest was written in BIGEARTHNET_19_CLASSES order
    for row in manifest_rows:
        by_concept[row["concept_name"]].append((
            int(row["tile_idx"]), Path(row["orig_tif"]), Path(row["mask_a_tif"]),
            Path(row["mask_b_tif"]), Path(row["mask_both_tif"]),
        ))

    out_name = os.environ.get("FTW_DUAL_OUTPUT_NAME", "conceptmask_ftw_dual_window_full.csv")
    ftw_dual.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    out_path = ftw_dual.RESULTS_DIR / out_name

    already_done = {}
    if out_path.exists():
        with open(out_path, newline="") as f:
            already_done = {row["concept_name"]: row for row in csv.DictReader(f)}
        if already_done:
            print(f"Resuming: {len(already_done)}/{len(by_concept)} concepts already scored in {out_path}.", flush=True)

    print(f"{len(manifest_rows)} masked-tile jobs across {len(by_concept)} concepts, "
          f"N_PARALLEL_INFERENCE={ftw_dual.N_PARALLEL_INFERENCE}.", flush=True)

    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=ftw_dual.N_PARALLEL_INFERENCE) as executor:
            for concept_idx, (concept_name, jobs) in enumerate(by_concept.items()):
                if concept_name in already_done:
                    print(f"[{concept_idx + 1}/{len(by_concept)}] {concept_name:<70s} SKIPPED (already scored)", flush=True)
                    continue

                deltas_a, deltas_b, deltas_both, n_failed = ftw_dual.score_jobs(jobs, MASKED_TILES_DIR, executor)
                score_a = sum(deltas_a) / len(deltas_a) if deltas_a else float("nan")
                score_b = sum(deltas_b) / len(deltas_b) if deltas_b else float("nan")
                score_both = sum(deltas_both) / len(deltas_both) if deltas_both else float("nan")
                summary = summary_by_name.get(concept_name, {})
                row = {
                    "concept_name": concept_name,
                    "raw_score_mask_a": score_a, "raw_score_mask_b": score_b, "raw_score_mask_both": score_both,
                    "n_present": summary.get("n_present", ""), "n_masked": summary.get("n_masked", ""),
                    "n_skipped_degenerate": summary.get("n_skipped_degenerate", ""), "n_failed_inference": n_failed,
                }
                writer.writerow(row)
                out_f.flush()  # survive an interruption immediately after this concept, not just at process exit
                print(f"[{concept_idx + 1}/{len(by_concept)}] {concept_name:<70s} "
                      f"mask_a={score_a:+.4f} mask_b={score_b:+.4f} mask_both={score_both:+.4f} "
                      f"(n_failed={n_failed})", flush=True)

    print(f"\nDone -- {out_path} has all {len(by_concept)} concept scores.", flush=True)


if __name__ == "__main__":
    main()
