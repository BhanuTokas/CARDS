#!/bin/bash
#SBATCH --job-name=celeba_official_train_convnext
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 1-00:00:00   # unknown until measured -- see submit_train_official_celeba_classifier_vit_hpc.sh
#SBATCH -o logs/celeba_official_train_convnext_%j.out
#SBATCH -e logs/celeba_official_train_convnext_%j.err
#
# Trains the ConvNeXt-Tiny counterpart of the official-train Attractive+
# Male classifier (train_official_celeba_classifier_convnext.py) -- the
# second new architecture, alongside ViT-B/16. Checkpoint: convnext_
# tiny_official_train_attractive_male.pt, registered as BACKBONES[
# "celeba_official_train_attractive_male_convnext"] once this lands.

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh

set -euo pipefail

# ---- fill these in for the real Sol filesystem layout ----
export CELEBA_ROOT=/data/hkerner/Datasets/CelebA/celeba   # confirmed 2026-09-17 -- must contain img_align_celeba/, list_attr_celeba.txt, list_eval_partition.txt
export CARDS_CKPT_DIR=trained_models_new/celeba            # relative to repo root, same as local
export CARDS_RESULTS_DIR=results                           # relative to repo root, same as local
export CARDS_NUM_WORKERS=8                                 # matches -c 8 above

mkdir -p logs "$CARDS_CKPT_DIR" "$CARDS_RESULTS_DIR"

cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/celeba/build/train_official_celeba_classifier_convnext.py ]; then
    echo "ERROR: scripts/celeba/build/train_official_celeba_classifier_convnext.py not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi

uv run python scripts/celeba/build/train_official_celeba_classifier_convnext.py
