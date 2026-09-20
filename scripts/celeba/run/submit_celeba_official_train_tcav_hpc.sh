#!/bin/bash
# CPU-only counterpart of submit_celeba_official_train_pipeline_gpu_hpc.sh,
# specifically for run_tcav_celeba_official_train.py (DEVICE="cpu",
# captum's Concept.data_iter never moves batches to CUDA -- matching
# this track's standing TCAV convention throughout, e.g. submit_tcav_
# shortcut_experiment_official_train_hpc.sh).
#
# Submit with, e.g.:
#   sbatch --export=CARDS_BACKBONE_NAME=celeba_official_train_attractive_male_vit \
#       scripts/celeba/run/submit_celeba_official_train_tcav_hpc.sh
#SBATCH -c 8
#SBATCH --mem 16G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 0-12:00:00   # unknown until measured
#SBATCH -o logs/celeba_official_train_tcav_%x_%j.out
#SBATCH -e logs/celeba_official_train_tcav_%x_%j.err

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh

set -euo pipefail

: "${CARDS_BACKBONE_NAME:?Set CARDS_BACKBONE_NAME via --export=CARDS_BACKBONE_NAME=... (e.g. celeba_official_train_attractive_male_vit)}"

export CELEBA_ROOT=/data/hkerner/Datasets/CelebA/celeba   # confirmed 2026-09-17
export CARDS_CKPT_DIR=trained_models_new/celeba
export CARDS_RESULTS_DIR=results

mkdir -p logs "$CARDS_RESULTS_DIR"

cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/celeba/run/run_tcav_celeba_official_train.py ]; then
    echo "ERROR: scripts/celeba/run/run_tcav_celeba_official_train.py not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi
if [ ! -d ../post_hoc_cbm ]; then
    echo "ERROR: ../post_hoc_cbm not found next to $(pwd)." >&2
    exit 1
fi

echo "CARDS_BACKBONE_NAME=$CARDS_BACKBONE_NAME"
uv run python scripts/celeba/run/run_tcav_celeba_official_train.py
