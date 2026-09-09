#!/bin/bash
#SBATCH --job-name=conceptmask_ftw_dual_score
#SBATCH -c 32
#SBATCH --mem 32G   # matches the single-window score job's own (measured-safe-enough) value -- see
                     # submit_ftw_hpc.sh for why 32G, not 24G: same per-process CLI subprocess cost applies here
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 2-00:00:00   # unknown until measured -- this job runs 4x the CLI calls per location vs the
                         # single-window score job's 2x (orig/mask_a/mask_b/mask_both), budget accordingly
#SBATCH -o logs/conceptmask_ftw_dual_score_%j.out
#SBATCH -e logs/conceptmask_ftw_dual_score_%j.err
#
# CPU-only, deliberately -- same reasoning as submit_ftw_hpc.sh (single-
# window): the `ftw_tools.cli inference run` calls get zero benefit from
# GPU. This is the SECOND job of the dual-window two-job split -- run
# scripts/ftw/build/submit_ftw_dual_window_build_masked_tiles.sh (GPU)
# FIRST; it writes the 4 GeoTIFFs/location + manifest this job reads from
# FTW_DUAL_MASKED_TILES_DIR.

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh
# See submit_ftw_build_masked_tiles.sh / submit_ftw_hpc.sh for why both of
# these are unset -- a stale mamba-module PROJ database, and (unconfirmed
# but cheap to rule out) dynamic-linker search-path pollution, both
# observed surviving `module purge` on Sol.
unset PROJ_LIB PROJ_DATA GDAL_DATA
unset LD_LIBRARY_PATH LD_PRELOAD

set -euo pipefail

# ---- fill these in for the real Sol filesystem layout ----
export FTW_DUAL_MASKED_TILES_DIR=/data/hkerner/btokas/Datasets/ftw_dual_window_masked_tiles                        # TODO: confirm -- MUST match the build job's value exactly
#
# CHECKPOINT: unlike the single-window job, this defaults to the REGISTRY
# NAME (FTW_PRUE_EFNET_B7_CCBY), which ftw_tools.cli auto-downloads via
# torch.hub on first use and caches under ~/.cache/torch/hub/checkpoints/.
# That download needs internet access -- SLURM COMPUTE NODES OFTEN DON'T
# HAVE IT (login nodes usually do). Two options, pick one:
#   (a) Pre-warm the cache once from the LOGIN node (has internet), before
#       submitting this job -- e.g. run a single manual `ftw_tools.cli
#       inference run` call with --model FTW_PRUE_EFNET_B7_CCBY there, or
#       just let the FIRST submission of this job fail/retry once the
#       cache is warm. Leave FTW_DUAL_CHECKPOINT unset for this path.
#   (b) Download the .ckpt manually (same pattern as the single-window
#       job's FTW_CHECKPOINT) and point FTW_DUAL_CHECKPOINT at that local
#       file instead -- avoids the auto-download/internet dependency
#       entirely. Uncomment and fill in:
# export FTW_DUAL_CHECKPOINT=/data/hkerner/btokas/utils/ftw-baselines/models/prue_efnetb7_ccby_checkpoint.ckpt
#
export FTW_BASELINES_VENV_PYTHON=/data/hkerner/btokas/utils/ftw-baselines/.venv/bin/python                         # confirmed -- same as the single-window job
export FTW_BASELINES_ROOT=/data/hkerner/btokas/utils/ftw-baselines                                                 # confirmed -- same as the single-window job
export FTW_RESULTS_DIR=results
export FTW_N_PARALLEL_INFERENCE=16   # matches the single-window score job's own dialed-back value -- raise
                                      # toward 32 once a run completes cleanly at this level (same caveat)
export FTW_DUAL_OUTPUT_NAME=conceptmask_ftw_dual_window_full.csv

mkdir -p logs

# See submit_ftw_build_masked_tiles.sh for why this is $SLURM_SUBMIT_DIR
# and not a dirname "$0" trick -- confirmed the latter fails under sbatch.
cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/ftw/run/score_ftw_dual_window_masked_tiles.py ]; then
    echo "ERROR: scripts/ftw/run/score_ftw_dual_window_masked_tiles.py not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi

# --extra ftw: rasterio lives in CARDS' own optional "ftw" dependency
# group -- see submit_ftw_build_masked_tiles.sh for the same fix/reason.
uv run --extra ftw python scripts/ftw/run/score_ftw_dual_window_masked_tiles.py
