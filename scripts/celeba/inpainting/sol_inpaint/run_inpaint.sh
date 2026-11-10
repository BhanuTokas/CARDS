#!/bin/bash
#SBATCH -J inpaint_gt
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-87
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# 11 conditions x 8 shards = 88 tasks (task = condition*8 + shard). Submit from this folder:  sbatch run_inpaint.sh
# --mem 64G: FLUX (12B transformer + T5) needs more host RAM while loading than the SD models.
# Smoke tests first (3 pairs each): sd2 = task 0, sdxl = task 24, flux = task 48
#   sbatch --array=0,24,48 --time=01:00:00 --export=ALL,LIMIT=3 run_inpaint.sh
# FLUX.1-Fill-dev is gated: accept the licence on its Hugging Face page with the account whose token is on Sol.

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1
export UV_PYTHON_PREFERENCE=only-managed
cd "$SLURM_SUBMIT_DIR"
[ -f manifest.csv ] && [ -d masks ] || { echo "submit from the sol_inpaint folder (manifest.csv + masks/ missing)"; exit 1; }

IMG_ROOT="${IMG_ROOT:-/data/hkerner/Datasets/CelebAMask-HQ/CelebA-HQ-img}"
OUT="${OUT:-$SLURM_SUBMIT_DIR/fills}"
# "model neg seed": primary = quality negative (SD) / none (FLUX), 3 seeds each; sensitivity = concept negative, 1 seed
CONDITIONS=(
  "sd2 quality 0" "sd2 quality 1" "sd2 quality 2"
  "sdxl quality 0" "sdxl quality 1" "sdxl quality 2"
  "flux none 0" "flux none 1" "flux none 2"
  "sd2 concept 0" "sdxl concept 0"
)
NSHARDS=8
T=$SLURM_ARRAY_TASK_ID
read -r MODEL NEG SEED <<< "${CONDITIONS[$(( T / NSHARDS ))]}"; SHARD=$(( T % NSHARDS ))
echo "task $T -> model $MODEL neg $NEG seed $SEED shard $SHARD/$NSHARDS"

uv run --locked python inpaint.py --model "$MODEL" --neg "$NEG" --seed "$SEED" --shard "$SHARD" --nshards "$NSHARDS" \
    --img-root "$IMG_ROOT" --bundle . --out "$OUT" --limit "${LIMIT:-0}"
