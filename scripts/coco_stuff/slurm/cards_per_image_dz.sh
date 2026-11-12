#!/bin/bash
#SBATCH --job-name=cards_dz
#SBATCH --account=grp_hkerner
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/data/hkerner/rnair21/CARDS/results/coco_stuff/logs/cards_dz_%x_%j.out
#SBATCH --error=/data/hkerner/rnair21/CARDS/results/coco_stuff/logs/cards_dz_%x_%j.err
# CARDS (alpha 1.5) rerun that also saves per-image deltas (for Cohen's d). Outputs get suffix _dz, existing files untouched.
# sbatch --export=ALL,BACKBONE_NAME=coco_stuff|coco_stuff_vit_b_16|coco_stuff_convnext_tiny ... (repo root)
export PYTHONNOUSERSITE=1 OUT_SUFFIX=_dz
export HF_HOME=/scratch/rnair21/.cache/huggingface TORCH_HOME=/scratch/rnair21/.cache/torch XDG_CACHE_HOME=/scratch/rnair21/.cache
module load gcc/12.5.0
module load cuda-12.4.1-gcc-12.1.0
cd /data/hkerner/rnair21/CARDS
/scratch/rnair21/envs/cards_env/bin/python scripts/coco_stuff/run/run_cards_coco_stuff_model_variants.py
