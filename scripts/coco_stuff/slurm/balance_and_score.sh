#!/bin/bash
#SBATCH --job-name=coco_balance
#SBATCH --partition=public
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/rnair21/logs/coco_balance_%j.out
#SBATCH --error=/scratch/rnair21/logs/coco_balance_%j.err

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
    scripts/coco_stuff/build/balance_faithfulness_and_score.py --min_count 100
