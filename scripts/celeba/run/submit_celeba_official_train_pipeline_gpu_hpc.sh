#!/bin/bash
# Generic GPU submission template for the official-train pipeline
# (ground truth, masking hybrid, PCBM CAV-fit, PCBM surrogate, PCBM
# CLIP-concepts), now that every stage is parameterized by
# CARDS_BACKBONE_NAME (celeba_official_train_attractive_male /
# _vit / _convnext) -- prompted directly ("We can do it on HPC if that
# will help?" for the ViT-B/16 + ConvNeXt-Tiny "full pipeline with PCBM
# variants"). One template instead of a separate .sh per (script,
# backbone) combination -- matches submit_caption_generation.sh's own
# --export convention.
#
# Submit with, e.g.:
#   sbatch --export=CARDS_SCRIPT=scripts/celeba/ground_truth/run_celeba_official_train_faithfulness.py,CARDS_BACKBONE_NAME=celeba_official_train_attractive_male_vit \
#       scripts/celeba/run/submit_celeba_official_train_pipeline_gpu_hpc.sh
#   sbatch --export=CARDS_SCRIPT=scripts/celeba/run/train_pcbm_clip_concepts_celeba_official_train.py,CARDS_BACKBONE_NAME=celeba_official_train_attractive_male_vit,CARDS_EXTRA_ARG=siglip \
#       scripts/celeba/run/submit_celeba_official_train_pipeline_gpu_hpc.sh
#
# Full sequence per backbone (celeba_official_train_attractive_male_vit,
# then _convnext), IN ORDER -- everything after step 1 depends on its
# ground truth CSVs existing:
#   1. scripts/celeba/ground_truth/run_celeba_official_train_faithfulness.py   (GPU, this template)
#   2. scripts/celeba/run/run_cards_celeba_masking_hybrid_official_train.py    (GPU, this template)
#   3. scripts/celeba/run/run_tcav_celeba_official_train.py                    (CPU -- submit_celeba_official_train_tcav_hpc.sh instead)
#   4. scripts/celeba/build/fit_celeba_official_train_cavs.py                  (GPU, this template)
#   5. scripts/celeba/run/train_pcbm_surrogate_celeba_official_train.py        (GPU, this template -- needs step 4's CAVs first)
#   6. scripts/celeba/run/train_pcbm_clip_concepts_celeba_official_train.py siglip     (GPU, this template, CARDS_EXTRA_ARG=siglip)
#   7. scripts/celeba/run/train_pcbm_clip_concepts_celeba_official_train.py clip_rn50  (GPU, this template, CARDS_EXTRA_ARG=clip_rn50)
#   8. scripts/celeba/analysis/score_all_methods_against_official_train_faithfulness.py (GPU node not actually needed, but harmless -- needs steps 2/3/5/6/7 first)
#
# `cd "$SLURM_SUBMIT_DIR"` (not `dirname "$0"`) because SLURM copies the
# submitted script into its own spool location before running it.
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 1-00:00:00   # unknown until measured -- varies a lot by stage (ground truth/PCBM-CAV-fit
                         # are the heaviest; PCBM-CLIP-concepts' one-time image embedding is the next
                         # heaviest; masking hybrid and the final comparison are lighter)
#SBATCH -o logs/celeba_official_train_pipeline_%x_%j.out
#SBATCH -e logs/celeba_official_train_pipeline_%x_%j.err

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh

set -euo pipefail

: "${CARDS_SCRIPT:?Set CARDS_SCRIPT (a python script path) via --export=CARDS_SCRIPT=...}"
: "${CARDS_BACKBONE_NAME:?Set CARDS_BACKBONE_NAME via --export=CARDS_BACKBONE_NAME=... (e.g. celeba_official_train_attractive_male_vit)}"
CARDS_EXTRA_ARG="${CARDS_EXTRA_ARG:-}"

# ---- fill these in for the real Sol filesystem layout ----
# No CARDS_CONCEPT_ROOT needed -- fit_celeba_official_train_cavs.py switched
# to whole-image CAVs (official-train-sourced), not the celeba_full_concepts
# region-crop bank, confirmed directly ("Switch PCBM conventional's concept
# source to whole images for the new architectures?" -> "Yes, whole images
# for all 3 (redo ResNet18 too)").
export CELEBA_HQ_ROOT=/data/hkerner/Datasets/CelebAMask-HQ            # TODO: confirm -- only ground truth generation + the masking hybrid's leakage check need this
export CELEBA_ROOT=/data/hkerner/Datasets/CelebA/celeba               # confirmed 2026-09-17
export CARDS_CKPT_DIR=trained_models_new/celeba
export CARDS_RESULTS_DIR=results
export CARDS_CACHE_DIR=embedding_cache
export CARDS_TRAINED_CONCEPTS_DIR=trained_concepts_new/celeba_full
export CARDS_TRAINED_MODELS_DIR=trained_models_new/celeba_full

mkdir -p logs "$CARDS_RESULTS_DIR" "$CARDS_CACHE_DIR" "$CARDS_TRAINED_CONCEPTS_DIR" "$CARDS_TRAINED_MODELS_DIR"

cd "$SLURM_SUBMIT_DIR"
if [ ! -f "$CARDS_SCRIPT" ]; then
    echo "ERROR: $CARDS_SCRIPT not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi
if [[ "$CARDS_SCRIPT" == *post_hoc_cbm* || "$CARDS_SCRIPT" == *pcbm* || "$CARDS_SCRIPT" == *tcav* ]] && [ ! -d ../post_hoc_cbm ]; then
    echo "ERROR: ../post_hoc_cbm not found next to $(pwd)." >&2
    exit 1
fi

echo "CARDS_SCRIPT=$CARDS_SCRIPT  CARDS_BACKBONE_NAME=$CARDS_BACKBONE_NAME  CARDS_EXTRA_ARG=$CARDS_EXTRA_ARG"

if [ -n "$CARDS_EXTRA_ARG" ]; then
    uv run --extra pcbm-training python "$CARDS_SCRIPT" "$CARDS_EXTRA_ARG"
else
    uv run --extra pcbm-training python "$CARDS_SCRIPT"
fi
