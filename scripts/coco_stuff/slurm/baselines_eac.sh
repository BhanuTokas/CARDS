#!/bin/bash
#SBATCH --job-name=ceac
#SBATCH --account=grp_hkerner
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/hkerner/rnair21/CARDS/results/coco_stuff/logs/ceac_%x_%j.out
#SBATCH --error=/data/hkerner/rnair21/CARDS/results/coco_stuff/logs/ceac_%x_%j.err
# EAC for one black box: sbatch --export=ALL,CARDS_ARCH=resnet18|vit|convnext[,EAC_LIMIT=3,TAG=_smoke] ... (repo root)
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/rnair21/.cache/huggingface TORCH_HOME=/scratch/rnair21/.cache/torch XDG_CACHE_HOME=/scratch/rnair21/.cache
export GT_MASKS_DIR=results/coco_stuff/gt_masks CARDS_SAM_CACHE=results/coco_stuff/baselines/sam_cache
module load gcc/12.5.0
module load cuda-12.4.1-gcc-12.1.0
cd /data/hkerner/rnair21/CARDS
D=results/coco_stuff/baselines/${CARDS_ARCH}; mkdir -p $D $CARDS_SAM_CACHE
/scratch/rnair21/envs/cards_env/bin/python scripts/coco_stuff/baselines/eac_coco.py weights/sam_vit_h_4b8939.pth $D/eac${TAG}.csv
