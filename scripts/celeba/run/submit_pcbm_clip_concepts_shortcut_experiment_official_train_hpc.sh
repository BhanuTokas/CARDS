#!/bin/bash
#SBATCH --job-name=celeba_shortcut_pcbm_clip_concepts_official_train
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem 24G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 0-08:00:00   # unknown until measured -- image embedding is done ONCE per backbone (shared
                         # across both tasks, all 11 rates); the per-(task,rate) work is just a cheap
                         # linear-probe refit over several lambda candidates, much lighter than the
                         # conventional (CAV-refit) PCBM job
#SBATCH -o logs/celeba_shortcut_pcbm_clip_concepts_official_train_%A_%a.out
#SBATCH -e logs/celeba_shortcut_pcbm_clip_concepts_official_train_%A_%a.err
#
# Runs run_pcbm_clip_concepts_shortcut_experiment_official_train.py: PCBM's
# "CLIP concepts" variant (frozen SigLIP or CLIP-RN50 backbone, no gradient/
# CAV-fit through the classifier) against all 22 official-train shortcut
# checkpoints. Requires post_hoc_cbm as a SIBLING directory to this repo.
#
# Takes the backbone as its own arg -- submit TWICE, once per encoder:
#   sbatch scripts/celeba/run/submit_pcbm_clip_concepts_shortcut_experiment_official_train_hpc.sh siglip
#   sbatch scripts/celeba/run/submit_pcbm_clip_concepts_shortcut_experiment_official_train_hpc.sh clip_rn50
#
# `--extra pcbm-training` is REQUIRED, confirmed directly by a real
# ModuleNotFoundError on Sol ("No module named 'pandas'") on the sibling
# conventional-PCBM job -- this script's own `from train_pcbm import
# run_linear_probe` hits the identical transitive import chain (train_pcbm
# -> post_hoc_cbm's data/__init__.py -> concept_loaders.py -> pandas).
# pandas lives behind CARDS' own `pcbm-training` extra (pyproject.toml).

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh

set -euo pipefail

BACKBONE="${1:?Usage: sbatch $0 <siglip|clip_rn50>}"
case "$BACKBONE" in
    siglip|clip_rn50) ;;
    *) echo "ERROR: unknown backbone '$BACKBONE' -- must be siglip or clip_rn50." >&2; exit 1 ;;
esac

# ---- fill these in for the real Sol filesystem layout ----
# No CELEBA_HQ_ROOT needed -- this script uses only official CelebA (CelebA-HQ isn't on this cluster).
export CELEBA_ROOT=/data/hkerner/Datasets/CelebA/celeba      # confirmed 2026-09-17 -- must match the training jobs' CELEBA_ROOT
export CARDS_CKPT_DIR=trained_models_new/celeba                     # confirmed 2026-09-17 -- left as the default, unedited, matching the training jobs' own default
export CARDS_RESULTS_DIR=results
export CARDS_PCBM_CLIP_OUT_DIR=trained_models_new/celeba_shortcut_official_train_clip_concepts

mkdir -p logs "$CARDS_RESULTS_DIR" "$CARDS_PCBM_CLIP_OUT_DIR/$BACKBONE"

cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/celeba/run/run_pcbm_clip_concepts_shortcut_experiment_official_train.py ]; then
    echo "ERROR: scripts/celeba/run/run_pcbm_clip_concepts_shortcut_experiment_official_train.py not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi
if [ ! -d ../post_hoc_cbm ]; then
    echo "ERROR: ../post_hoc_cbm not found next to $(pwd) -- this script imports run_linear_probe from it." >&2
    exit 1
fi

uv run --extra pcbm-training python scripts/celeba/run/run_pcbm_clip_concepts_shortcut_experiment_official_train.py "$BACKBONE"
