#!/bin/bash
#SBATCH --job-name=celeba_shortcut_tcav_official_train
#SBATCH -c 8
#SBATCH --mem 16G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 0-12:00:00   # unknown until measured -- CPU-only (captum's TCAV never moves batches to
                         # CUDA, matching this track's standing TCAV convention), 26 concepts x 22
                         # (task,rate) checkpoints, but exemplar/random/control pools are built ONCE
#SBATCH -o logs/celeba_shortcut_tcav_official_train_%j.out
#SBATCH -e logs/celeba_shortcut_tcav_official_train_%j.err
#
# CPU-only, deliberately -- no -G/--gres=gpu request, matching run_tcav_
# shortcut_experiment.py's own DEVICE="cpu" (captum's Concept.data_iter
# never moves batches to CUDA).
#
# Runs run_tcav_shortcut_experiment_official_train.py: "conventional" TCAV
# (native ResNet18 layer4 activation space) against all 22 official-train
# shortcut checkpoints. Requires post_hoc_cbm as a SIBLING directory to
# this repo (../post_hoc_cbm relative to CARDS root) -- confirm it's
# checked out at that path on Sol too, same as this track's other
# TCAV/PCBM scripts.

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh

set -euo pipefail

# ---- fill these in for the real Sol filesystem layout ----
# No CELEBA_HQ_ROOT needed -- this script uses only official CelebA (CelebA-HQ isn't on this cluster).
export CELEBA_ROOT=/data/hkerner/Datasets/CelebA/celeba      # confirmed 2026-09-17 -- must match the training jobs' CELEBA_ROOT
export CARDS_CKPT_DIR=trained_models_new/celeba                     # confirmed 2026-09-17 -- left as the default, unedited, matching the training jobs' own default
export CARDS_RESULTS_DIR=results

mkdir -p logs "$CARDS_RESULTS_DIR"

cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/celeba/run/run_tcav_shortcut_experiment_official_train.py ]; then
    echo "ERROR: scripts/celeba/run/run_tcav_shortcut_experiment_official_train.py not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi
if [ ! -d ../post_hoc_cbm ]; then
    echo "ERROR: ../post_hoc_cbm not found next to $(pwd) -- this script imports captum/concept helpers from it." >&2
    exit 1
fi

uv run python scripts/celeba/run/run_tcav_shortcut_experiment_official_train.py
