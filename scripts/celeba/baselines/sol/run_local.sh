#!/bin/bash
#SBATCH -J local_attr
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Local (per-image) attribution of the extra baselines for the official-train ResNet18 on CelebA, on the present and
# absent records of the paper's local experiment (see local_common.py). Array task T: record set = present for
# T < 8, absent otherwise; shard = T % 8. Submit from the repo root, one METHOD at a time:
#   mkdir -p logs
#   sbatch --export=ALL,METHOD=soft scripts/celeba/baselines/sol/run_local.sh            # CARDS + soft CARDS
#   sbatch --export=ALL,METHOD=eac  scripts/celeba/baselines/sol/run_local.sh            # EAC (SAM + 2,500 queries/image)
#   sbatch --array=0 --export=ALL,METHOD=sae_fit scripts/celeba/baselines/sol/run_local.sh  # features + SAE (one task)
#   sbatch --array=0,8 --export=ALL,METHOD=sae scripts/celeba/baselines/sol/run_local.sh    # after sae_fit (fast)
# Smoke test (one shard, 20 items):  sbatch --array=0 --time=00:30:00 --export=ALL,METHOD=eac,LOCAL_LIMIT=20 ...
# Then copy $OUT back and run score_local.py and plot_local_all.py (they need the existing local CSVs and the images).
# Needs, relative to the repo root: results/celeba_official_train_faithfulness_non_overlapping.csv,
# results/local_attribution_celeba_official_train_absent_pairs.csv, trained_models_new/celeba/ (ResNet18 checkpoint).
# EAC also needs segment-anything (git clone, commit dca509f) at $SAM_REPO and the ViT-H checkpoint at $SAM_CKPT.

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1
export UV_PYTHON_PREFERENCE=only-managed
cd "$SLURM_SUBMIT_DIR"
[ -f scripts/celeba/baselines/local_common.py ] || { echo "submit from the repo root"; exit 1; }

: "${METHOD:?set METHOD=soft|eac|sae_fit|sae}"
export CARDS_ARCH=resnet18
export CELEBA_ROOT="${CELEBA_ROOT:-/data/hkerner/Datasets/CelebA/celeba}"
export CELEBA_HQ_ROOT="${CELEBA_HQ_ROOT:-/data/hkerner/Datasets/CelebAMask-HQ}"
export CARDS_SAM_CACHE="${CARDS_SAM_CACHE:-/scratch/$USER/cards_sam_cache_celeba}"
SAM_REPO="${SAM_REPO:-/data/hkerner/$USER/segment-anything}"
SAM_CKPT="${SAM_CKPT:-/data/hkerner/$USER/sam_weights/sam_vit_h_4b8939.pth}"
OUT="${OUT:-results/baselines/local_resnet18}"
mkdir -p "$OUT"
NSHARDS=8
T=${SLURM_ARRAY_TASK_ID:-0}
export CARDS_LOCAL_SET=$([ "$T" -lt "$NSHARDS" ] && echo present || echo absent)
export SHARD=$(( T % NSHARDS )) NSHARDS
echo "METHOD=$METHOD set=$CARDS_LOCAL_SET shard=$SHARD/$NSHARDS out=$OUT"
B=scripts/celeba/baselines

case "$METHOD" in
  soft)    uv run --locked python $B/local_soft.py "$OUT" ;;
  eac)     [ -f "$SAM_CKPT" ] && [ -d "$SAM_REPO/segment_anything" ] || { echo "missing SAM checkpoint or repo"; exit 1; }
           PYTHONPATH="$SAM_REPO" uv run --locked python $B/local_eac.py "$SAM_CKPT" "$OUT" ;;
  sae_fit) uv run --locked python $B/extract_feats.py
           uv run --locked python $B/concept_sae.py "$OUT/sae_global.csv" ;;
  sae)     uv run --locked python $B/local_sae.py "$OUT/sae_global.pt" "$OUT" ;;
  *)       echo "unknown METHOD=$METHOD"; exit 1 ;;
esac
