#!/bin/bash
#SBATCH -J funnybirds
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-10
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# FunnyBirds evaluation (scripts/funnybirds/evaluate.py), one run per array task, for $MODEL (default resnet50):
#   0 InputXGradient   1 IntegratedGradients   2 HnS (paper: L=196, alpha=1)   3 HnS with true part masks
#   4 HnS L=1024 alpha=1.5 (selected on training images)   5 HnS L=1024 alpha=2.5 (sensitivity)   6 EAC
#   7 TCAV (layer3)   8 PCBM (conventional, 26 variant CAVs)   9 PCBM CLIP-RN50 concepts   10 PCBM SigLIP concepts
# Each run takes well under 4 h, so submit on htc. From the repo root, after `git pull`:
#   mkdir -p logs
#   uv sync --locked --extra tcav --extra perception --extra pcbm     # once on the login node (array tasks share .venv)
#   (pcbm only for `requests`, which the framework's dataset module imports for its unused render server)
#   sbatch -p htc scripts/funnybirds/sol/run_funnybirds.sh
#   sbatch -p htc --array=0 --time=00:30:00 --export=ALL,NR_ITRS=5 scripts/funnybirds/sol/run_funnybirds.sh  # smoke
# Needs (defaults below, override by env): the dataset at $FUNNYBIRDS_ROOT, a clone of
# https://github.com/visinf/funnybirds-framework (commit 1350bbe) at $FUNNYBIRDS_FRAMEWORK, the authors' checkpoint(s)
# in $FUNNYBIRDS_CKPT_DIR, and for EAC segment-anything (commit dca509f) at $SEGMENT_ANYTHING + ViT-H weights at $SAM_CKPT.
# Results: $FUNNYBIRDS_OUT/<model>_<explainer>[_tag].json (+ per-image HnS/EAC logs); copy back for the tables.

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1
export UV_PYTHON_PREFERENCE=only-managed
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
[ -f scripts/funnybirds/evaluate.py ] || { echo "submit from the repo root"; exit 1; }

export FUNNYBIRDS_ROOT="${FUNNYBIRDS_ROOT:-/data/hkerner/Datasets/FunnyBirds}"
export FUNNYBIRDS_FRAMEWORK="${FUNNYBIRDS_FRAMEWORK:-/data/hkerner/$USER/funnybirds-framework}"
export FUNNYBIRDS_CKPT_DIR="${FUNNYBIRDS_CKPT_DIR:-/data/hkerner/$USER/FunnyBirds_models}"
export FUNNYBIRDS_OUT="${FUNNYBIRDS_OUT:-results/funnybirds}"
export SEGMENT_ANYTHING="${SEGMENT_ANYTHING:-/data/hkerner/$USER/segment-anything}"
export SAM_CKPT="${SAM_CKPT:-/data/hkerner/$USER/sam_weights/sam_vit_h_4b8939.pth}"
MODEL="${MODEL:-resnet50}"
NR_ITRS="${NR_ITRS:-2501}"   # framework default = all 500 test images

[ -f "$FUNNYBIRDS_ROOT/dataset_test.json" ] || { echo "dataset not found at $FUNNYBIRDS_ROOT"; exit 1; }
[ -f "$FUNNYBIRDS_FRAMEWORK/evaluation_protocols.py" ] || { echo "framework not found at $FUNNYBIRDS_FRAMEWORK"; exit 1; }
ls "$FUNNYBIRDS_CKPT_DIR"/*checkpoint_best.pth.tar > /dev/null || { echo "no checkpoints in $FUNNYBIRDS_CKPT_DIR"; exit 1; }

T=${SLURM_ARRAY_TASK_ID:-0}
case "$T" in
  0) ARGS=(--explainer InputXGradient) ;;
  1) ARGS=(--explainer IntegratedGradients) ;;
  2) ARGS=(--explainer HnS) ;;
  3) ARGS=(--explainer HnSTrueMask) ;;
  4) ARGS=(--explainer HnS --encoder ViT-B-16-SigLIP-512 --alpha 1.5 --tag L1024_a1.5) ;;
  5) ARGS=(--explainer HnS --encoder ViT-B-16-SigLIP-512 --alpha 2.5 --tag L1024_a2.5) ;;
  6) [ -f "$SAM_CKPT" ] && [ -d "$SEGMENT_ANYTHING/segment_anything" ] || { echo "missing SAM checkpoint or repo"; exit 1; }
     ARGS=(--explainer EAC) ;;
  7) ARGS=(--explainer TCAV) ;;
  8) ARGS=(--explainer PCBM) ;;
  9) ARGS=(--explainer PCBM_CLIP_RN50) ;;
  10) ARGS=(--explainer PCBM_SigLIP) ;;
  *) echo "unknown task $T"; exit 1 ;;
esac
echo "task $T model=$MODEL nr_itrs=$NR_ITRS args=${ARGS[*]} out=$FUNNYBIRDS_OUT"
uv run --locked --extra tcav --extra perception --extra pcbm python scripts/funnybirds/evaluate.py \
  --model "$MODEL" --nr_itrs "$NR_ITRS" "${ARGS[@]}"
