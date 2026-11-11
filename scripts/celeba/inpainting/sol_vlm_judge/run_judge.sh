#!/bin/bash
#SBATCH -J vlm_judge
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# 3 parts x 8 shards (task = part*8 + shard; parts: 0 val, 1 retain, 2 real). Submit from this folder:
#   mkdir -p logs && sbatch run_judge.sh
# Smoke test (1 shard per part, 32 questions each):  sbatch --array=0,8,16 --time=00:30:00 --export=ALL,LIMIT=32 run_judge.sh
# Needs: ../sol_inpaint (manifest.csv) and ../sol_inpaint/fills/{run}, including fills/lama_s0 (uploaded from the laptop).

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1
export UV_PYTHON_PREFERENCE=only-managed
cd "$SLURM_SUBMIT_DIR"
[ -f judge.py ] || { echo "submit from the sol_vlm_judge folder"; exit 1; }

BUNDLE="${BUNDLE:-$SLURM_SUBMIT_DIR/../sol_inpaint}"
HQ="${HQ:-/data/hkerner/Datasets/CelebAMask-HQ/CelebA-HQ-img}"
[ -f "$BUNDLE/manifest.csv" ] && [ -d "$BUNDLE/fills/lama_s0" ] || { echo "missing $BUNDLE/manifest.csv or fills/lama_s0"; exit 1; }
PARTS=(val retain real); NSHARDS=8
T=$SLURM_ARRAY_TASK_ID; PART=${PARTS[$(( T / NSHARDS ))]}; SHARD=$(( T % NSHARDS ))
echo "task $T -> part $PART shard $SHARD/$NSHARDS"

uv run --locked python judge.py --part "$PART" --shard "$SHARD" --nshards "$NSHARDS" \
    --hq "$HQ" --fills "$BUNDLE/fills" --bundle "$BUNDLE" --out "$SLURM_SUBMIT_DIR/out" --limit "${LIMIT:-0}"
