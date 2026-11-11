#!/bin/bash
#SBATCH --job-name=coco_train_vit
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/rnair21/logs/coco_train_vit_%j.out
#SBATCH --error=/scratch/rnair21/logs/coco_train_vit_%j.err

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
    scripts/coco_stuff/build/train_coco_stuff_classifier.py \
    --arch vit_b_16
