#!/bin/bash
#SBATCH --job-name=cards_model_variants
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/rnair21/logs/cards_model_variants_%j.out
#SBATCH --error=/scratch/rnair21/logs/cards_model_variants_%j.err

# Runs CARDS for all 3 backbone variants sequentially on the same GPU.
# Prerequisites:
#   train_vit.sh and train_convnext.sh must have completed successfully.
#   The ResNet18 checkpoint already exists from 01_train_classifier.sh.
#
# Outputs:
#   results/coco_stuff/model_variants/coco_stuff_cards_scores.csv
#   results/coco_stuff/model_variants/coco_stuff_vit_b_16_cards_scores.csv
#   results/coco_stuff/model_variants/coco_stuff_convnext_tiny_cards_scores.csv
#   (+ matching *_summary.csv for each backbone)

export CONDA_PKGS_DIRS=/scratch/rnair21/.cache/conda/pkgs
export PIP_CACHE_DIR=/scratch/rnair21/.cache/pip
export HF_HOME=/scratch/rnair21/.cache/huggingface
export TORCH_HOME=/scratch/rnair21/.cache/torch
export TRANSFORMERS_CACHE=/scratch/rnair21/.cache/huggingface/transformers
export XDG_CACHE_HOME=/scratch/rnair21/.cache
export PYTHONNOUSERSITE=1

module load gcc/12.5.0
module load cuda-12.4.1-gcc-12.1.0

PYTHON=/scratch/rnair21/envs/cards_env/bin/python
SCRIPT=scripts/coco_stuff/run/run_cards_coco_stuff_model_variants.py

cd /scratch/rnair21/CARDS

for BACKBONE in coco_stuff coco_stuff_vit_b_16 coco_stuff_convnext_tiny; do
    echo "=== BACKBONE_NAME=${BACKBONE} ==="
    BACKBONE_NAME=${BACKBONE} ${PYTHON} ${SCRIPT}
    echo "=== done ${BACKBONE} ==="
done
