#!/bin/bash
#SBATCH --job-name=mv_pcbm
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/rnair21/logs/mv_pcbm_%j.out
#SBATCH --error=/scratch/rnair21/logs/mv_pcbm_%j.err

# PCBM attribution for all 3 backbone variants (ResNet18, ViT-B/16, ConvNeXt-Tiny).
# Heavy job: encodes all ~118K COCO train2017 images per backbone.
# Prerequisite: train_vit.sh and train_convnext.sh must have completed.
#
# Outputs (results/coco_stuff/model_variants/):
#   {backbone}_pcbm_concept_weights.csv
#   {backbone}_pcbm_summary.csv   (rho vs. faithfulness, both strategies)

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
SCRIPT=scripts/coco_stuff/run/train_pcbm_coco_stuff_model_variants.py

cd /scratch/rnair21/CARDS

for BACKBONE in coco_stuff coco_stuff_vit_b_16 coco_stuff_convnext_tiny; do
    echo "=== BACKBONE_NAME=${BACKBONE} (PCBM) ==="
    BACKBONE_NAME=${BACKBONE} ${PYTHON} ${SCRIPT}
    echo "=== done ${BACKBONE} ==="
done
