#!/bin/bash
#SBATCH --job-name=celeba_shortcut_attractive_official_train
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 1-00:00:00   # unknown until measured -- 11 rates x up to 15 epochs over 162,770 train
                         # images (ResNet18, single 2-way head); early stopping (patience=3) means
                         # most rates finish well under MAX_EPOCHS. Re-budget after the first rate logs.
#SBATCH -o logs/celeba_shortcut_attractive_official_train_%j.out
#SBATCH -e logs/celeba_shortcut_attractive_official_train_%j.err
#
# Trains 11 checkpoints (rate=0,10,20,...,100%) of the Attractive shortcut-
# learning classifier on standard CelebA's official train partition
# (162,770 images), for train_attractive_shortcut_classifiers_official_
# train.py. GPU, single node -- plain ResNet18 fine-tune, no distributed
# training needed.
#
# Resumable: the script skips any rate whose checkpoint (.pt) + run-info
# (.pkl) pair already exists in CARDS_CKPT_DIR, so a preempted/timed-out
# job can just be resubmitted as-is.

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh

set -euo pipefail

# ---- fill these in for the real Sol filesystem layout ----
export CELEBA_ROOT=/data/hkerner/btokas/Datasets/CelebA/celeba   # TODO: confirm -- must contain img_align_celeba/, list_attr_celeba.txt, list_eval_partition.txt
export CARDS_CKPT_DIR=trained_models_new/celeba                 # relative to repo root, same as local
export CARDS_RESULTS_DIR=results                                # relative to repo root, same as local
export CARDS_NUM_WORKERS=8                                       # matches -c 8 above

mkdir -p logs "$CARDS_CKPT_DIR" "$CARDS_RESULTS_DIR"

# SLURM copies the submitted script into its own spool location and runs it
# from there, so `dirname "$0"` does NOT reliably point back at the repo --
# $SLURM_SUBMIT_DIR is the directory `sbatch` was invoked FROM. Submit this
# job with sbatch from the CARDS repo root.
cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/celeba/build/train_attractive_shortcut_classifiers_official_train.py ]; then
    echo "ERROR: scripts/celeba/build/train_attractive_shortcut_classifiers_official_train.py not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi

uv run python scripts/celeba/build/train_attractive_shortcut_classifiers_official_train.py
