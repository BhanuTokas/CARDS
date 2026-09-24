#!/bin/bash
#SBATCH --job-name=cost_bench_pcbm_conventional
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 0-06:00:00   # local RTX 4090: SigLIP full train+val embed (same 182,637 images) took ~20min;
                         # ResNet-18 is far smaller, generous ceiling for a cold Sol node + CAV fit phase
#SBATCH -o logs/cost_bench_pcbm_conventional_%j.out
#SBATCH -e logs/cost_bench_pcbm_conventional_%j.err
#
# Runs scripts/celeba/analysis/benchmark_computational_cost_pcbm_conventional.py:
# the fourth row of the computational cost comparison ("PCBM (conventional,
# ResNet-18)" -- CAVs fit directly in the black-box model's own activation
# space, no separate concept encoder), at the same full scale as the other
# three methods' existing benchmark (results/computational_cost_benchmark_
# full_scale.csv). Requires post_hoc_cbm as a SIBLING directory to this repo
# (../post_hoc_cbm relative to CARDS root) and the pcbm-training extra
# (pandas, imported transitively by train_pcbm.py's own data/__init__.py).

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh

set -euo pipefail

export CELEBA_ROOT=/data/hkerner/Datasets/CelebA/celeba   # confirmed 2026-09-17, matches other official-train jobs
export CARDS_RESULTS_DIR=results

mkdir -p logs "$CARDS_RESULTS_DIR"

cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/celeba/analysis/benchmark_computational_cost_pcbm_conventional.py ]; then
    echo "ERROR: benchmark script not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi
if [ ! -d ../post_hoc_cbm ]; then
    echo "ERROR: ../post_hoc_cbm not found next to $(pwd) -- this script imports PosthocLinearCBM/run_linear_probe/ConceptBank from it." >&2
    exit 1
fi

uv run --extra pcbm-training python scripts/celeba/analysis/benchmark_computational_cost_pcbm_conventional.py
