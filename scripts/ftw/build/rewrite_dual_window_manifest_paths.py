"""One-off helper: rewrites a downloaded dual-window manifest.csv's Sol
absolute paths to point at wherever it was downloaded to locally --
same fix applied to the single-window manifest earlier in this
investigation (notes v9), generalized here for the dual-window
manifest's 4 path columns (orig_tif, mask_a_tif, mask_b_tif,
mask_both_tif) instead of the single-window one's 2 (orig_tif,
masked_tif). Matches on filename only (Path(...).name), so it's
agnostic to whatever the Sol directory prefix actually was.

Usage: edit LOCAL_DIR below to wherever you downloaded the folder, then
run this once before scripts/ftw/run/score_ftw_dual_window_masked_tiles.py.
"""

from __future__ import annotations

import csv
from pathlib import Path

LOCAL_DIR = Path(r"C:\Users\btokas\Projects\Datasets\ftw_dual_window_masked_tiles")  # confirmed
PATH_COLUMNS = ["orig_tif", "mask_a_tif", "mask_b_tif", "mask_both_tif"]


def main():
    manifest_path = LOCAL_DIR / "manifest.csv"
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        for col in PATH_COLUMNS:
            row[col] = str(LOCAL_DIR / Path(row[col]).name)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["concept_idx", "concept_name", "tile_idx"] + PATH_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Rewrote {len(rows)} rows in {manifest_path} to point at {LOCAL_DIR}")
    print("Example row:", rows[0] if rows else "(empty)")


if __name__ == "__main__":
    main()
