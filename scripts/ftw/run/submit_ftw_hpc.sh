#!/bin/bash
#SBATCH --job-name=conceptmask_ftw_score
#SBATCH -c 32
#SBATCH --mem 24G
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

set -euo pipefail

# ---- fill these in for the real Sol filesystem layout ----
export FTW_MASKED_TILES_DIR=/data/hkerner/btokas/Datasets/ftw_masked_tiles                                         # confirmed -- MUST match the build job's value exactly
export FTW_CHECKPOINT=/data/hkerner/btokas/utils/ftw-baselines/models/3_Class_FULL_FTW_Pretrained_singleWindow_v2.ckpt   # confirmed
export FTW_BASELINES_VENV_PYTHON=/data/hkerner/btokas/utils/ftw-baselines/.venv/bin/python                         # confirmed
export FTW_BASELINES_ROOT=/data/hkerner/btokas/utils/ftw-baselines                                                 # confirmed
export FTW_RESULTS_DIR=results
export FTW_N_PARALLEL_INFERENCE=${SLURM_CPUS_PER_TASK}
export FTW_OUTPUT_NAME=conceptmask_ftw_bigearthnet_full.csv

mkdir -p logs

cd "$(dirname "$0")/../../.."   # repo root (scripts/ftw/run/ -> CARDS/)
uv run python scripts/ftw/run/score_ftw_masked_tiles.py
