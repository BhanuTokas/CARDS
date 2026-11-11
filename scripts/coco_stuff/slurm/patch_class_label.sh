#!/bin/bash
#SBATCH --job-name=patch_faith
#SBATCH --partition=public
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/rnair21/logs/patch_faith_%j.out
#SBATCH --error=/scratch/rnair21/logs/patch_faith_%j.err

export CONDA_PKGS_DIRS=/scratch/rnair21/.cache/conda/pkgs
export PIP_CACHE_DIR=/scratch/rnair21/.cache/pip
export HF_HOME=/scratch/rnair21/.cache/huggingface
export TORCH_HOME=/scratch/rnair21/.cache/torch
export TRANSFORMERS_CACHE=/scratch/rnair21/.cache/huggingface/transformers
export XDG_CACHE_HOME=/scratch/rnair21/.cache
export PYTHONNOUSERSITE=1

module load gcc/12.5.0

cd /scratch/rnair21/CARDS
/scratch/rnair21/envs/cards_env/bin/python \
    scripts/coco_stuff/build/patch_faithfulness_add_class_label.py
