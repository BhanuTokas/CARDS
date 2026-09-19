#!/bin/bash
#SBATCH --job-name=celeba_official_train_vit
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 1-00:00:00   # unknown until measured -- ViT-B/16 is a heavier forward/backward pass than
                         # ResNet18 per image; up to 15 epochs over 162,770 official-train images,
                         # early stopping (patience=3) likely cuts this well short of MAX_EPOCHS
#SBATCH -o logs/celeba_official_train_vit_%j.out
#SBATCH -e logs/celeba_official_train_vit_%j.err
#
# Trains the ViT-B/16 counterpart of the official-train Attractive+Male
# classifier (train_official_celeba_classifier_vit.py) -- prompted
# directly ("I want to do the main CelebA experiment with ViT and one
# other model as the black box model" -> "Official-train arc"). See
# submit_train_official_celeba_classifier_convnext_hpc.sh for the other
# new architecture. Checkpoint: vit_b_16_official_train_attractive_
# male.pt, registered as BACKBONES["celeba_official_train_attractive_
# male_vit"] once this lands.

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh

set -euo pipefail

# ---- fill these in for the real Sol filesystem layout ----
export CELEBA_ROOT=/data/hkerner/Datasets/CelebA/celeba   # confirmed 2026-09-17 -- must contain img_align_celeba/, list_attr_celeba.txt, list_eval_partition.txt
export CARDS_CKPT_DIR=trained_models_new/celeba            # relative to repo root, same as local
export CARDS_RESULTS_DIR=results                           # relative to repo root, same as local
export CARDS_NUM_WORKERS=8                                 # matches -c 8 above

mkdir -p logs "$CARDS_CKPT_DIR" "$CARDS_RESULTS_DIR"

# SLURM copies the submitted script into its own spool location and runs it
# from there, so `dirname "$0"` does NOT reliably point back at the repo --
# $SLURM_SUBMIT_DIR is the directory `sbatch` was invoked FROM. Submit this
# job with sbatch from the CARDS repo root.
cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/celeba/build/train_official_celeba_classifier_vit.py ]; then
    echo "ERROR: scripts/celeba/build/train_official_celeba_classifier_vit.py not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi

uv run python scripts/celeba/build/train_official_celeba_classifier_vit.py
