#!/bin/bash
#SBATCH --job-name=cost_bench_tcav
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 0-04:00:00   # local (CPU-only, per captum's own device constraint): ~406s; generous ceiling for a slower Sol CPU node
#SBATCH -o logs/cost_bench_tcav_%j.out
#SBATCH -e logs/cost_bench_tcav_%j.err
#
# Runs scripts/celeba/analysis/benchmark_computational_cost_tcav.py: TCAV's
# row of the computational cost comparison, at the same full scale as the
# other methods (results/computational_cost_benchmark_full_scale.csv).
# NO GPU requested: captum's Concept.data_iter never moves batches to CUDA
# (same constraint as every other TCAV script in this codebase), so a GPU
# allocation here would sit idle. Requires post_hoc_cbm as a SIBLING
# directory to this repo (../post_hoc_cbm relative to CARDS root) for
# ListDataset.

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh

set -euo pipefail

export CELEBA_ROOT=/data/hkerner/Datasets/CelebA/celeba   # confirmed 2026-09-17, matches other official-train jobs
export CARDS_RESULTS_DIR=results

mkdir -p logs "$CARDS_RESULTS_DIR"

cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/celeba/analysis/benchmark_computational_cost_tcav.py ]; then
    echo "ERROR: benchmark script not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi
if [ ! -d ../post_hoc_cbm ]; then
    echo "ERROR: ../post_hoc_cbm not found next to $(pwd) -- this script imports ListDataset from it." >&2
    exit 1
fi

uv run python scripts/celeba/analysis/benchmark_computational_cost_tcav.py
