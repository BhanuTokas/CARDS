#!/bin/bash
#SBATCH --job-name=cbase
#SBATCH --account=grp_hkerner
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/hkerner/rnair21/CARDS/results/coco_stuff/logs/cbase_%x_%j.out
#SBATCH --error=/data/hkerner/rnair21/CARDS/results/coco_stuff/logs/cbase_%x_%j.err
# ConceptSAE + ConceptSHAP for one black box: sbatch --export=ALL,CARDS_ARCH=resnet18|vit|convnext ... (repo root)
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/rnair21/.cache/huggingface TORCH_HOME=/scratch/rnair21/.cache/torch XDG_CACHE_HOME=/scratch/rnair21/.cache
module load gcc/12.5.0
module load cuda-12.4.1-gcc-12.1.0
cd /data/hkerner/rnair21/CARDS
PY=/scratch/rnair21/envs/cards_env/bin/python
D=results/coco_stuff/baselines/${CARDS_ARCH}; mkdir -p $D
$PY scripts/coco_stuff/baselines/extract_feats_coco.py && \
$PY scripts/coco_stuff/baselines/concept_sae_coco.py $D/sae.csv && \
$PY scripts/coco_stuff/baselines/concept_shap_coco.py $D/cshap.csv
