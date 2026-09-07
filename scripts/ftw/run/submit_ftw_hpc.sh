#!/bin/bash
#SBATCH --job-name=conceptmask_ftw_score
#SBATCH -c 32
#SBATCH --mem 32G   # bumped from 24G after a crash ("Aborted!", near-zero diagnostic) at 32-way
                     # parallel CLI subprocess invocations -- each is a FRESH independent Python
                     # process (own interpreter + torch + GDAL/rasterio + model weights), not a
                     # shared worker. Rough estimate ~1-1.5GB/process x 16 (N_PARALLEL_INFERENCE,
                     # dialed back below) ~= 16-24GB -- 32G is a real but not wildly excessive
                     # margin, unmeasured/unconfirmed since stderr gave no real diagnostic
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 2-00:00:00   # unknown until a full-dataset timing is measured, adjust
#SBATCH -o logs/conceptmask_ftw_score_%j.out
#SBATCH -e logs/conceptmask_ftw_score_%j.err
#
# CPU-only, deliberately: no -G/--gres=gpu request. This is the SECOND job
# of the two-job split -- run scripts/ftw/build/submit_ftw_build_masked_tiles.sh
# (GPU) FIRST; it writes the orig/masked GeoTIFF pairs + manifest this job
# reads from FTW_MASKED_TILES_DIR.
#
# Only the `ftw_tools.cli inference run` subprocess calls happen here --
# measured to get ZERO speedup from GPU (a single 256x256-tile invocation
# is ~10-13s either way -- per-call latency is import/model-load/CUDA-init
# overhead, not compute; notes v7). The encoder-dependent work (pool
# embedding, localization, best-of-family fill selection) all happened in
# the build job already -- see that script's docstring and notes v10 for
# why it's split this way (direct feedback: "wouldn't concept masking
# also require GPU..." -- correct, so it's NOT in this CPU-only job).

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh
# See submit_ftw_build_masked_tiles.sh for why -- a stale mamba-module PROJ
# database otherwise leaks onto rasterio's search path even after purge.
unset PROJ_LIB PROJ_DATA GDAL_DATA

set -euo pipefail

# ---- fill these in for the real Sol filesystem layout ----
export FTW_MASKED_TILES_DIR=/data/hkerner/btokas/Datasets/ftw_masked_tiles                                         # confirmed -- MUST match the build job's value exactly
export FTW_CHECKPOINT=/data/hkerner/btokas/utils/ftw-baselines/models/3_Class_FULL_FTW_Pretrained_singleWindow_v2.ckpt   # confirmed
export FTW_BASELINES_VENV_PYTHON=/data/hkerner/btokas/utils/ftw-baselines/.venv/bin/python                         # confirmed
export FTW_BASELINES_ROOT=/data/hkerner/btokas/utils/ftw-baselines                                                 # confirmed
export FTW_RESULTS_DIR=results
export FTW_N_PARALLEL_INFERENCE=16   # dialed back from SLURM_CPUS_PER_TASK (32) as a belt-and-
                                      # suspenders retry alongside the --mem bump above -- raise
                                      # back toward 32 once a run completes cleanly at this level
export FTW_OUTPUT_NAME=conceptmask_ftw_bigearthnet_full.csv

mkdir -p logs

# See submit_ftw_build_masked_tiles.sh for why this is $SLURM_SUBMIT_DIR
# and not a dirname "$0" trick -- confirmed the latter fails under sbatch.
cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/ftw/run/score_ftw_masked_tiles.py ]; then
    echo "ERROR: scripts/ftw/run/score_ftw_masked_tiles.py not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi

# --extra ftw: rasterio lives in CARDS' own optional "ftw" dependency
# group -- see submit_ftw_build_masked_tiles.sh for the same fix/reason.
uv run --extra ftw python scripts/ftw/run/score_ftw_masked_tiles.py
