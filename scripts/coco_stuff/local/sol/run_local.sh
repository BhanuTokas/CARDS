#!/bin/bash
#SBATCH -J coco_local
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# COCO-Stuff local (per-image) attribution for the ResNet18 black box: Hide and Seek, TCAV, EAC, ConceptSAE on the
# records of make_records.py, then scores and top-3/bottom-3 grids. One METHOD per submission; array tasks are shards.
# Submit from the repo root (uv sync --locked on the login node first), each step after the previous one:
#   mkdir -p logs
#   P=$(sbatch --parsable --export=ALL,METHOD=prep scripts/coco_stuff/local/sol/run_local.sh)
#   C=$(sbatch --parsable --dependency=afterok:$P --array=0-7  --export=ALL,METHOD=cards scripts/coco_stuff/local/sol/run_local.sh)
#   T=$(sbatch --parsable --dependency=afterok:$P              --export=ALL,METHOD=tcav  scripts/coco_stuff/local/sol/run_local.sh)
#   E=$(sbatch --parsable --dependency=afterok:$P --array=0-15 --export=ALL,METHOD=eac   scripts/coco_stuff/local/sol/run_local.sh)
#   S=$(sbatch --parsable --dependency=afterok:$P              --export=ALL,METHOD=sae   scripts/coco_stuff/local/sol/run_local.sh)
#   sbatch --dependency=afterok:$C:$T:$E:$S --export=ALL,METHOD=score scripts/coco_stuff/local/sol/run_local.sh
# Smoke test: same, with OUT pointing to a scratch folder and LOCAL_LIMIT=20 on cards/eac (the prep step is full size).
# Chosen pairs instead of the selection rule (e.g. the paper's example figures): export LOCAL_PAIRS="class:concept;..."
# and a separate OUT before the same chain (fewer shards suffice: --array=0-1 for cards, 0-3 for eac).
# The paper's figures ranked EVERY class-present train2017 image (plot_coco_stuff_local_attribution_viz.py), not the
# 100 + 100 records: add LOCAL_N_PER_SET=1000000 (all present + all absent images of each pair's class = the full class
# pool), skip eac (it needs ground-truth regions), use --array=0-15 for cards, make score depend on cards/tcav/sae only,
# and export LOCAL_PLOT_METHODS=cards,tcav_dd,sae (continuous layer3 TCAV: the sign score ties on large pools).
# Inputs (env, defaults in brackets): COCO_ROOT, BRODEN_ROOT [coco_common.py defaults]; CARDS_RESULTS_DIR
# [results/coco_stuff] with coco_stuff_faithfulness.csv + _detection_counts.csv; GT_MASKS_DIR [$CARDS_RESULTS_DIR/gt_masks];
# CARDS_FEAT_CACHE [$CARDS_RESULTS_DIR/baselines/feat_cache; extract_feats_coco.py runs if pooled.pt is missing];
# trained_models_new/coco_stuff/resnet18_coco_stuff.pt; SAM_REPO / SAM_CKPT for EAC. Outputs go to $OUT.

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1
export UV_PYTHON_PREFERENCE=only-managed
cd "$SLURM_SUBMIT_DIR"
[ -f scripts/coco_stuff/local/common.py ] || { echo "submit from the repo root"; exit 1; }

: "${METHOD:?set METHOD=prep|cards|tcav|eac|sae|score}"
export CARDS_ARCH=resnet18
export CARDS_RESULTS_DIR="${CARDS_RESULTS_DIR:-results/coco_stuff}"
export GT_MASKS_DIR="${GT_MASKS_DIR:-$CARDS_RESULTS_DIR/gt_masks}"
export CARDS_SAM_CACHE="${CARDS_SAM_CACHE:-/scratch/$USER/cards_sam_cache_coco}"
export COCO_LOCAL_OUT="${OUT:-results/coco_stuff_local/resnet18}"
SAM_REPO="${SAM_REPO:-/data/hkerner/$USER/segment-anything}"
SAM_CKPT="${SAM_CKPT:-/data/hkerner/$USER/sam_weights/sam_vit_h_4b8939.pth}"
export NSHARDS="${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}" SHARD="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$COCO_LOCAL_OUT"
echo "METHOD=$METHOD shard=$SHARD/$NSHARDS out=$COCO_LOCAL_OUT results=$CARDS_RESULTS_DIR"
[ -f trained_models_new/coco_stuff/resnet18_coco_stuff.pt ] || { echo "missing ResNet18 checkpoint"; exit 1; }
[ -f "$CARDS_RESULTS_DIR/coco_stuff_faithfulness_detection_counts.csv" ] || { echo "missing COCO ground truth in $CARDS_RESULTS_DIR"; exit 1; }
L=scripts/coco_stuff/local
RUN="uv run --locked --extra pcbm-training python"   # pcbm-training = pandas, which coco_common.py imports

case "$METHOD" in
  prep)  $RUN $L/make_records.py
         $RUN $L/local_cards.py cutoffs ;;
  cards) $RUN $L/local_cards.py score ;;
  tcav)  $RUN $L/local_tcav.py ;;
  eac)   [ -f "$SAM_CKPT" ] && [ -d "$SAM_REPO/segment_anything" ] || { echo "missing SAM checkpoint or repo"; exit 1; }
         [ -d "$GT_MASKS_DIR" ] || { echo "missing GT_MASKS_DIR=$GT_MASKS_DIR"; exit 1; }
         mkdir -p "$CARDS_SAM_CACHE"
         PYTHONPATH="$SAM_REPO" $RUN $L/local_eac.py "$SAM_CKPT" ;;
  sae)   export CARDS_FEAT_CACHE="${CARDS_FEAT_CACHE:-$CARDS_RESULTS_DIR/baselines/feat_cache}"
         [ -f "$CARDS_FEAT_CACHE/pooled.pt" ] || $RUN scripts/coco_stuff/baselines/extract_feats_coco.py
         $RUN $L/local_sae.py ;;
  score) $RUN $L/score_local.py "$COCO_LOCAL_OUT"
         $RUN $L/plot_local.py "$COCO_LOCAL_OUT" ;;
  *)     echo "unknown METHOD=$METHOD"; exit 1 ;;
esac
