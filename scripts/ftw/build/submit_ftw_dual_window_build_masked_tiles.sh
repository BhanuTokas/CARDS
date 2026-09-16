#!/bin/bash
#SBATCH --job-name=conceptmask_ftw_dual_build
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 0-06:00:00   # unknown until measured -- covers combined-embedding pool build + PER-WINDOW
                         # localization/masking (2x the encoder work of the single-window build job)
                         # for ALL K x 19 concepts; bumped above the single-window job's 4h as a guess
#SBATCH -o logs/conceptmask_ftw_dual_build_%j.out
#SBATCH -e logs/conceptmask_ftw_dual_build_%j.err
#
# GPU stage of the DUAL-WINDOW two-job split -- same reasoning as
# submit_ftw_build_masked_tiles.sh (single-window): retrieval pool
# embedding + localization + best-of-family fill selection are real
# SigLIP forward passes that benefit from GPU; only the CLI inference
# calls (score stage) are CPU-only-fine. This job does MORE encoder work
# than the single-window one -- combined window_a+window_b pool
# embedding, then INDEPENDENT per-window localization/masking (own
# sim_map, own cutoff, own fill selection per window, not shared --
# notes v10/v13) -- so treat the walltime guess above as a rough
# starting point, not a measured value.
#
# Writes 4 GeoTIFFs per non-degenerate present location (orig, mask_a,
# mask_b, mask_both) + a manifest to FTW_DUAL_MASKED_TILES_DIR. Run
# scripts/ftw/run/submit_ftw_dual_window_hpc.sh (CPU-only) on that
# manifest AFTER this job completes.

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh
# See submit_ftw_build_masked_tiles.sh for why -- a stale mamba-module PROJ
# database otherwise leaks onto rasterio's search path even after purge.
unset PROJ_LIB PROJ_DATA GDAL_DATA

set -euo pipefail

# ---- must match submit_ftw_dual_window_hpc.sh's FTW_DUAL_MASKED_TILES_DIR exactly ----
export FTW_ROOT=/data/hkerner/FTW/ftw                                                                              # confirmed -- same as the single-window job
export FTW_DUAL_MASKED_TILES_DIR=/data/hkerner/btokas/Datasets/ftw_dual_window_masked_tiles                        # TODO: confirm location, needs real disk space -- 4 GeoTIFFs per present, non-degenerate location (2x the single-window job's own 2-per-location)
export FTW_DUAL_K=50   # TODO: pick a real value -- local pilot was K=10 (scripts/ftw/run/run_conceptmask_ftw_dual_window.py's own default)

mkdir -p logs "$FTW_DUAL_MASKED_TILES_DIR"

# SLURM copies the submitted script into its own spool location and runs it
# from there, so `dirname "$0"` does NOT reliably point back at the repo --
# see submit_ftw_build_masked_tiles.sh for the confirmed failure. Submit
# this job with sbatch from the CARDS repo root.
cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/ftw/build/build_ftw_dual_window_masked_tiles.py ]; then
    echo "ERROR: scripts/ftw/build/build_ftw_dual_window_masked_tiles.py not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi

# --extra ftw: rasterio lives in CARDS' own optional "ftw" dependency group
# -- see submit_ftw_build_masked_tiles.sh for the same fix/reason.
uv run --extra ftw python scripts/ftw/build/build_ftw_dual_window_masked_tiles.py
