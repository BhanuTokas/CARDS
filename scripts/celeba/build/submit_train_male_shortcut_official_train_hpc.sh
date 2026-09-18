#!/bin/bash
#SBATCH --job-name=celeba_shortcut_male_official_train
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 1-00:00:00   # unknown until measured -- see submit_train_attractive_shortcut_official_train_hpc.sh
#SBATCH -o logs/celeba_shortcut_male_official_train_%j.out
#SBATCH -e logs/celeba_shortcut_male_official_train_%j.err
#
# Male companion to submit_train_attractive_shortcut_official_train_hpc.sh
# -- same job shape (11 checkpoints, rate=0,10,20,...,100%, GPU, single
# node), trains train_male_shortcut_classifiers_official_train.py instead.
# Independent of the Attractive job: submit both, they don't share
# checkpoints or need to run in any particular order.
#
# Resumable: skips any rate whose checkpoint (.pt) + run-info (.pkl) pair
# already exists in CARDS_CKPT_DIR.

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh

set -euo pipefail

# ---- must match the Attractive job's CELEBA_ROOT exactly ----
export CELEBA_ROOT=/data/hkerner/Datasets/CelebA/celeba   # confirmed 2026-09-17 -- must contain img_align_celeba/, list_attr_celeba.txt, list_eval_partition.txt
export CARDS_CKPT_DIR=trained_models_new/celeba                 # relative to repo root, same as local
export CARDS_RESULTS_DIR=results                                # relative to repo root, same as local
export CARDS_NUM_WORKERS=8                                       # matches -c 8 above

mkdir -p logs "$CARDS_CKPT_DIR" "$CARDS_RESULTS_DIR"

cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/celeba/build/train_male_shortcut_classifiers_official_train.py ]; then
    echo "ERROR: scripts/celeba/build/train_male_shortcut_classifiers_official_train.py not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi

uv run python scripts/celeba/build/train_male_shortcut_classifiers_official_train.py
