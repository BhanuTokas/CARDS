#!/bin/bash
#SBATCH --job-name=abl6_normalization
#SBATCH --partition=public
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/rnair21/logs/abl6_normalization_%j.out
#SBATCH --error=/scratch/rnair21/logs/abl6_normalization_%j.err

# No GPU needed. Requires: alpha_ablation_per_image_scores.csv (from abl5_alpha job)

export CONDA_PKGS_DIRS=/scratch/rnair21/.cache/conda/pkgs
export PIP_CACHE_DIR=/scratch/rnair21/.cache/pip
export HF_HOME=/scratch/rnair21/.cache/huggingface
export TORCH_HOME=/scratch/rnair21/.cache/torch
export TRANSFORMERS_CACHE=/scratch/rnair21/.cache/huggingface/transformers
export XDG_CACHE_HOME=/scratch/rnair21/.cache
export PYTHONNOUSERSITE=1

module purge
module load gcc/12.5.0

cd /scratch/rnair21/CARDS
/scratch/rnair21/envs/cards_env/bin/python \
    scripts/coco_stuff/ablations/ablate_normalization.py
