#!/bin/bash
#SBATCH --job-name=local_attr_coco
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/btokas/logs/local_attr_coco_%j.out
#SBATCH --error=/scratch/btokas/logs/local_attr_coco_%j.err

# Per-image local attribution experiment:
#   100 train2017 test images × 103 concepts × 20 classes = 206,000 triples.
#   Methods: CARDS hybrid, baseline (absent-set logit delta), TCAV (gradient × CAV).
# Output:
#   results/coco_stuff/local_attribution_coco_stuff_pairs.csv
#   Prints pooled + per-class Spearman rho for all 3 methods × 2 strategies.

export CONDA_PKGS_DIRS=/scratch/rnair21/.cache/conda/pkgs
export PIP_CACHE_DIR=/scratch/rnair21/.cache/pip
export HF_HOME=/scratch/rnair21/.cache/huggingface
export TORCH_HOME=/scratch/rnair21/.cache/torch
export TRANSFORMERS_CACHE=/scratch/rnair21/.cache/huggingface/transformers
export XDG_CACHE_HOME=/scratch/rnair21/.cache
export PYTHONNOUSERSITE=1

module load gcc/12.5.0
module load cuda-12.4.1-gcc-12.1.0

cd /scratch/rnair21/CARDS
/scratch/rnair21/envs/cards_env/bin/python \
    scripts/coco_stuff/analysis/local_attribution_coco_stuff.py