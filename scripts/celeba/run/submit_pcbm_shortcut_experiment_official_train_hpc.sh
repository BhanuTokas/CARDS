#!/bin/bash
#SBATCH --job-name=celeba_shortcut_pcbm_conventional_official_train
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 1-12:00:00   # unknown until measured -- CAVs are refit from scratch per (task,rate) since
                         # they're tied to that specific classifier's own activation space (22 refits
                         # total), each followed by a full ~25,500-train-image + val-image embed pass
#SBATCH -o logs/celeba_shortcut_pcbm_conventional_official_train_%j.out
#SBATCH -e logs/celeba_shortcut_pcbm_conventional_official_train_%j.err
#
# Runs run_pcbm_shortcut_experiment_official_train.py: "conventional" PCBM
# (CAVs fit in each classifier's OWN ResNet18 activation space) against all
# 22 official-train shortcut checkpoints. Requires post_hoc_cbm as a
# SIBLING directory to this repo (../post_hoc_cbm relative to CARDS root).
#
# `--extra pcbm-training` is REQUIRED, confirmed directly by a real
# ModuleNotFoundError on Sol ("No module named 'pandas'"): this script's
# `from train_pcbm import run_linear_probe` runs post_hoc_cbm's own
# `data/__init__.py` (imports concept_loaders.py, which imports pandas)
# even though run_linear_probe itself never touches pandas -- a
# transitive import-time cost of importing anything from train_pcbm.py.
# pandas lives behind CARDS' own `pcbm-training` extra (pyproject.toml),
# not the default dependency set.

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh

set -euo pipefail

# ---- fill these in for the real Sol filesystem layout ----
# No CELEBA_HQ_ROOT / CARDS_CONCEPT_ROOT needed -- this script uses WHOLE
# official-train images filtered by attribute label, not CelebA-HQ region
# crops (CelebA-HQ isn't on this cluster).
export CELEBA_ROOT=/data/hkerner/Datasets/CelebA/celeba               # confirmed 2026-09-17 -- must match the training jobs' CELEBA_ROOT
export CARDS_CKPT_DIR=trained_models_new/celeba                              # confirmed 2026-09-17 -- left as the default, unedited, matching the training jobs' own default
export CARDS_RESULTS_DIR=results
export CARDS_PCBM_OUT_DIR=trained_models_new/celeba_shortcut_official_train

mkdir -p logs "$CARDS_RESULTS_DIR" "$CARDS_PCBM_OUT_DIR"

cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/celeba/run/run_pcbm_shortcut_experiment_official_train.py ]; then
    echo "ERROR: scripts/celeba/run/run_pcbm_shortcut_experiment_official_train.py not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi
if [ ! -d ../post_hoc_cbm ]; then
    echo "ERROR: ../post_hoc_cbm not found next to $(pwd) -- this script imports PosthocLinearCBM/run_linear_probe from it." >&2
    exit 1
fi

uv run --extra pcbm-training python scripts/celeba/run/run_pcbm_shortcut_experiment_official_train.py
