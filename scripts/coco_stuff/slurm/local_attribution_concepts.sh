#!/bin/bash
#SBATCH --job-name=local_attr_coco_by_concept
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-102
#SBATCH --output=/scratch/btokas/logs/local_attr_coco_by_concept_%A_%a.out
#SBATCH --error=/scratch/btokas/logs/local_attr_coco_by_concept_%A_%a.err
# Per-concept local attribution visualization, sharded as a SLURM array job:
#   one task per Broden concept (0-102, same 103 balanced concepts, same
#   deterministic order as the other COCO-Stuff local-attribution scripts),
#   each task sweeps ALL 20 COCO-Stuff classes and scores up to 200
#   class-present train2017 images per class with Hide and Seek + TCAV.
# Output (per task):
#   results/coco_stuff/local_attribution_viz/by_concept/{concept_name}/{class_name}.png
#   (20 grids per task, 103 tasks -> 2,060 grids total)
#
# --time=04:00:00 is an estimate, not measured: each task scores at most
# 20 classes x 200 images = 4,000 images (vs. the old all-classes-x-all-
# concepts run's much larger per-class image counts), so it should need
# well under the old job's 24h -- but re-check actual per-task wall time
# after the first array task finishes and adjust before resubmitting the
# rest if it's off.
export CONDA_PKGS_DIRS=/scratch/rnair21/.cache/conda/pkgs
export PIP_CACHE_DIR=/scratch/rnair21/.cache/pip
export HF_HOME=/scratch/rnair21/.cache/huggingface
export TORCH_HOME=/scratch/rnair21/.cache/torch
export TRANSFORMERS_CACHE=/scratch/rnair21/.cache/huggingface/transformers
export XDG_CACHE_HOME=/scratch/rnair21/.cache
export PYTHONNOUSERSITE=1
export COCO_CONCEPT_INDEX=$SLURM_ARRAY_TASK_ID
module load gcc/12.5.0
module load cuda-12.4.1-gcc-12.1.0
cd /scratch/rnair21/CARDS
/scratch/rnair21/envs/cards_env/bin/python \
    scripts/coco_stuff/analysis/local_attribution_concepts.py
