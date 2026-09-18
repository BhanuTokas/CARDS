#!/bin/bash
#SBATCH --job-name=celeba_shortcut_conceptmask_official_train
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 1-00:00:00   # unknown until measured -- 26 concepts x 22 (task,rate) checkpoints x up to
                         # 50 present images x 7 fill strategies; localization/retrieval itself is
                         # cached ONCE (shared across both tasks and all 11 rates), so most of this
                         # walltime is the per-(task,rate) masking/scoring loop, not re-embedding.
#SBATCH -o logs/celeba_shortcut_conceptmask_official_train_%j.out
#SBATCH -e logs/celeba_shortcut_conceptmask_official_train_%j.err
#
# Runs run_attribution_shortcut_experiment_official_train.py: the masking-
# hybrid/ConceptMask shortcut-reliance check against all 22 official-train
# shortcut checkpoints (Attractive + Male, 11 rates each). Requires the
# checkpoints from submit_train_{attractive,male}_shortcut_official_train_
# hpc.sh to already exist in CARDS_CKPT_DIR.

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh

set -euo pipefail

# ---- fill these in for the real Sol filesystem layout ----
# No CELEBA_HQ_ROOT needed -- this script uses only official CelebA (CelebA-HQ isn't on this cluster).
export CELEBA_ROOT=/data/hkerner/Datasets/CelebA/celeba      # confirmed 2026-09-17 -- must match the training jobs' CELEBA_ROOT
export CARDS_CKPT_DIR=trained_models_new/celeba                     # confirmed 2026-09-17 -- left as the default, unedited, matching the training jobs' own default
export CARDS_RESULTS_DIR=results

mkdir -p logs "$CARDS_RESULTS_DIR" embedding_cache

cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/celeba/run/run_attribution_shortcut_experiment_official_train.py ]; then
    echo "ERROR: scripts/celeba/run/run_attribution_shortcut_experiment_official_train.py not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi

uv run python scripts/celeba/run/run_attribution_shortcut_experiment_official_train.py
