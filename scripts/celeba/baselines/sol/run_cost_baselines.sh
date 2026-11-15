#!/bin/bash
#SBATCH -J cost_baselines
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Full-scale wall-clock cost of ConceptSAE, ConceptSHAP and EAC on CelebA (benchmark_cost_baselines.py), on the same
# hardware as the paper's A100 cost table. Starts cold (fresh feature and SAM caches); EAC runs twice to split SAM
# (reusable) from the black-box-dependent part.
# Submit from the repo root:
#   mkdir -p logs && sbatch scripts/celeba/baselines/sol/run_cost_baselines.sh
# Optional: --export=ALL,CARDS_ARCH=vit (or convnext), RUN=<fresh folder>.
# Needs segment-anything (git clone, commit dca509f) at $SAM_REPO and the ViT-H checkpoint at $SAM_CKPT, and
# trained_models_new/celeba/ (the black-box checkpoints).

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1
export UV_PYTHON_PREFERENCE=only-managed
cd "$SLURM_SUBMIT_DIR"
[ -f scripts/celeba/baselines/benchmark_cost_baselines.py ] || { echo "submit from the repo root"; exit 1; }

export CARDS_ARCH="${CARDS_ARCH:-resnet18}"
export CELEBA_ROOT="${CELEBA_ROOT:-/data/hkerner/Datasets/CelebA/celeba}"
export CELEBA_HQ_ROOT="${CELEBA_HQ_ROOT:-/data/hkerner/Datasets/CelebAMask-HQ}"
SAM_REPO="${SAM_REPO:-/data/hkerner/$USER/segment-anything}"
SAM_CKPT="${SAM_CKPT:-/data/hkerner/$USER/sam_weights/sam_vit_h_4b8939.pth}"
RUN="${RUN:-results/baselines/cost_${CARDS_ARCH}_${SLURM_JOB_ID}}"
[ -f "$SAM_CKPT" ] && [ -d "$SAM_REPO/segment_anything" ] || { echo "missing SAM checkpoint or repo"; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader

PYTHONPATH="$SAM_REPO" uv run --locked python scripts/celeba/baselines/benchmark_cost_baselines.py "$SAM_CKPT" "$RUN"
