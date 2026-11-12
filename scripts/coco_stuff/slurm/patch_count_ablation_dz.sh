#!/bin/bash
#SBATCH --job-name=patch_dz
#SBATCH --account=grp_hkerner
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/hkerner/rnair21/CARDS/results/coco_stuff/logs/patch_dz_%j.out
#SBATCH --error=/data/hkerner/rnair21/CARDS/results/coco_stuff/logs/patch_dz_%j.err
# Patch-count (L) ablation, ResNet18 black box. Stops after L=196 if it does not reproduce the headline cell.
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/rnair21/.cache/huggingface TORCH_HOME=/scratch/rnair21/.cache/torch XDG_CACHE_HOME=/scratch/rnair21/.cache
module load gcc/12.5.0
module load cuda-12.4.1-gcc-12.1.0
cd /data/hkerner/rnair21/CARDS
OUT_SUFFIX=_dz /scratch/rnair21/envs/cards_env/bin/python scripts/coco_stuff/ablations/ablate_patch_count.py
