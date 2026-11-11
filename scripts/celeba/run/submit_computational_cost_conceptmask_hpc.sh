#!/bin/bash
#SBATCH --job-name=cost_bench_conceptmask
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 0-02:00:00   # local RTX 4090: full run (encoder_load+pool_embed+query_build+scoring, 26 concepts x 2 tasks, raw 19,867-image official-val pool) totaled ~349s
#SBATCH -o logs/cost_bench_conceptmask_%j.out
#SBATCH -e logs/cost_bench_conceptmask_%j.err
#
# Runs scripts/celeba/analysis/benchmark_computational_cost_conceptmask.py:
# ConceptMask's row of the computational cost comparison, at the same full
# scale as the other methods (results/computational_cost_benchmark_full_
# scale.csv). No post_hoc_cbm dependency (neither cards.pipeline nor this
# script imports from it) and no CelebAMask-HQ dependency (uses the RAW
# official-val pool, not the HQ-overlap-excluded "clean" one -- see the
# benchmark script's own docstring), so this runs on Sol with no extra
# sibling-repo setup.

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh

set -euo pipefail

export CELEBA_ROOT=/data/hkerner/Datasets/CelebA/celeba   # confirmed 2026-09-17, matches other official-train jobs
export CARDS_RESULTS_DIR=results
export CARDS_COST_BENCH_CACHE_DIR=cost_bench_cache_conceptmask

mkdir -p logs "$CARDS_RESULTS_DIR" "$CARDS_COST_BENCH_CACHE_DIR"

cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/celeba/analysis/benchmark_computational_cost_conceptmask.py ]; then
    echo "ERROR: benchmark script not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi

uv run python scripts/celeba/analysis/benchmark_computational_cost_conceptmask.py
