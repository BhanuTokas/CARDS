#!/bin/bash
#SBATCH -J coco_shortcut_bl
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# ConceptSAE, ConceptSHAP and EAC on the 11 COCO-Stuff shortcut checkpoints. Array task id = rate / 10. Each task runs the
# METHODS env in order (default "feats sae shap"), then deletes its large feature maps. EAC is a separate, longer
# submission (METHODS=eac): rate 0 first, which fills the shared SAM cache, then the other 10 rates.
# Submit from the repo root after `uv sync --locked` on the login node:
#   mkdir -p logs
#   L=$(sbatch -p htc --parsable --array=0 --export=ALL,METHODS=labels scripts/coco_stuff/shortcut_baselines/sol/run.sh)
#   F=$(sbatch -p htc --parsable --dependency=afterok:$L --array=0-10 scripts/coco_stuff/shortcut_baselines/sol/run.sh)
#   E0=$(sbatch --parsable --dependency=afterok:$L --array=0 --time=24:00:00 --export=ALL,METHODS=eac scripts/coco_stuff/shortcut_baselines/sol/run.sh)
#   sbatch --dependency=afterok:$E0 --array=1-10 --time=24:00:00 --export=ALL,METHODS=eac scripts/coco_stuff/shortcut_baselines/sol/run.sh
#   smoke: sbatch -p htc --array=0 --time=01:00:00 --export=ALL,METHODS=eac,EAC_LIMIT=3,SHORTCUT_OUT=/scratch/$USER/coco_sc_smoke scripts/coco_stuff/shortcut_baselines/sol/run.sh
# EAC is resumable (per-image rows in eac.csv.images.csv): a task that hits its time limit continues when resubmitted.
# Then: uv run --locked --extra pcbm-training python scripts/coco_stuff/shortcut_baselines/summarize.py (CPU)
# Needs: resnet18_coco_stuff_shortcut_{0..100}.pt in $COCO_SHORTCUT_CKPT_DIR [trained_models_new/coco_stuff/shortcut];
# COCO_ROOT, BRODEN_ROOT [coco_common.py defaults]; CARDS_RESULTS_DIR [results/coco_stuff] with
# coco_stuff_faithfulness.csv + _detection_counts.csv; GT_MASKS_DIR [$CARDS_RESULTS_DIR/gt_masks] for EAC;
# segment-anything at $SAM_REPO and the ViT-H checkpoint at $SAM_CKPT.

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1
export UV_PYTHON_PREFERENCE=only-managed
cd "$SLURM_SUBMIT_DIR"
[ -f scripts/coco_stuff/shortcut_baselines/run_baseline.py ] || { echo "submit from the repo root"; exit 1; }

RATE=$(( ${SLURM_ARRAY_TASK_ID:-0} * 10 ))
export COCO_SHORTCUT_CKPT_DIR="${COCO_SHORTCUT_CKPT_DIR:-trained_models_new/coco_stuff/shortcut}"
export CARDS_RESULTS_DIR="${CARDS_RESULTS_DIR:-results/coco_stuff}"
export GT_MASKS_DIR="${GT_MASKS_DIR:-$CARDS_RESULTS_DIR/gt_masks}"
export CARDS_SAM_CACHE="${CARDS_SAM_CACHE:-/scratch/$USER/cards_sam_cache_coco}"   # shared with the other COCO EAC runs
export SHORTCUT_OUT="${SHORTCUT_OUT:-results/coco_stuff/shortcut_baselines}"
export SHORTCUT_FEAT_ROOT="${SHORTCUT_FEAT_ROOT:-/scratch/$USER/coco_shortcut_feats}"
SAM_REPO="${SAM_REPO:-/data/hkerner/$USER/segment-anything}"
SAM_CKPT="${SAM_CKPT:-/data/hkerner/$USER/sam_weights/sam_vit_h_4b8939.pth}"
METHODS="${METHODS:-feats sae shap}"
CK="$COCO_SHORTCUT_CKPT_DIR/resnet18_coco_stuff_shortcut_${RATE}.pt"
[ "$METHODS" = labels ] || [ -f "$CK" ] || { echo "missing $CK"; exit 1; }
[ -f "$CARDS_RESULTS_DIR/coco_stuff_faithfulness_detection_counts.csv" ] || { echo "missing COCO ground truth in $CARDS_RESULTS_DIR"; exit 1; }
echo "rate=$RATE methods=$METHODS ckpt=$CK out=$SHORTCUT_OUT results=$CARDS_RESULTS_DIR"
mkdir -p "$SHORTCUT_FEAT_ROOT"
RUN="uv run --locked --extra pcbm-training python scripts/coco_stuff/shortcut_baselines/run_baseline.py $RATE"   # pcbm-training = pandas
FEAT="$SHORTCUT_FEAT_ROOT/feat_${RATE}"

for M in $METHODS; do
  case "$M" in
    labels) $RUN labels ;;
    feats) [ -f "$FEAT/pooled.pt" ] && [ -f "$FEAT/map_train40k.pt" ] && { echo "features exist, skipping"; continue; }
           $RUN feats ;;
    sae|shap) $RUN $M ;;
    eac)   [ -f "$SAM_CKPT" ] && [ -d "$SAM_REPO/segment_anything" ] || { echo "missing SAM checkpoint or repo"; exit 1; }
           [ -d "$GT_MASKS_DIR" ] || { echo "missing GT_MASKS_DIR=$GT_MASKS_DIR"; exit 1; }
           mkdir -p "$CARDS_SAM_CACHE"
           PYTHONPATH="$SAM_REPO" $RUN eac "$SAM_CKPT" ;;
    *)     echo "unknown method $M"; exit 1 ;;
  esac
done
# feature maps are only needed by shap (~4 GB per checkpoint); keep the small pooled file
[[ " $METHODS " == *" shap "* ]] && rm -f "$FEAT"/map_*.pt
echo "done"
