#!/bin/bash
#SBATCH --job-name=coco_eval
#SBATCH --partition=public
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/rnair21/logs/coco_eval_%j.out
#SBATCH --error=/scratch/rnair21/logs/coco_eval_%j.err

# --- Cache dirs: keep everything off home quota ---
export CONDA_PKGS_DIRS=/scratch/rnair21/.cache/conda/pkgs
export PIP_CACHE_DIR=/scratch/rnair21/.cache/pip
export HF_HOME=/scratch/rnair21/.cache/huggingface
export TORCH_HOME=/scratch/rnair21/.cache/torch
export TRANSFORMERS_CACHE=/scratch/rnair21/.cache/huggingface/transformers
export XDG_CACHE_HOME=/scratch/rnair21/.cache
export PYTHONNOUSERSITE=1

cd /scratch/rnair21/CARDS
/scratch/rnair21/envs/cards_env/bin/python scripts/coco_stuff/analysis/score_cards_vs_faithfulness_coco_stuff.py
