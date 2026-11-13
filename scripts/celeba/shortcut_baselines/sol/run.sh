#!/bin/bash
#SBATCH -J shortcut_bl
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-21
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# ConceptSAE, ConceptSHAP and EAC on the 22 official-train shortcut checkpoints (2 tasks x 11 rates). Array task T:
# task = Attractive for T < 11, else Male; rate = (T % 11) * 10. Each task runs feats -> sae -> shap -> eac
# (METHODS env, default all four, in that order), then deletes its large feature maps.
# Submit from the repo root after `uv sync --locked` on the login node:
#   mkdir -p logs
#   sbatch -p htc scripts/celeba/shortcut_baselines/sol/run.sh
#   smoke: sbatch -p htc --array=0 --time=01:00:00 --export=ALL,METHODS=eac,EAC_LIMIT=3 scripts/celeba/shortcut_baselines/sol/run.sh
# Then: uv run --locked python scripts/celeba/shortcut_baselines/summarize.py  (CPU, needs the existing shortcut CSVs)
# Needs: the 22 checkpoints resnet18_official_train_{attractive,male}_shortcut_{rate}.pt in $CARDS_CKPT_DIR
# [trained_models_new/celeba], CelebA + CelebAMask-HQ, segment-anything at $SAM_REPO and the ViT-H checkpoint at $SAM_CKPT.

module purge
export PATH="$HOME/.local/bin:$PATH"   # uv
set -euo pipefail
export PYTHONNOUSERSITE=1
export UV_PYTHON_PREFERENCE=only-managed
cd "$SLURM_SUBMIT_DIR"
[ -f scripts/celeba/shortcut_baselines/run_baseline.py ] || { echo "submit from the repo root"; exit 1; }

T=${SLURM_ARRAY_TASK_ID:-0}
TASK=$([ "$T" -lt 11 ] && echo Attractive || echo Male); RATE=$(( (T % 11) * 10 ))
export CARDS_CKPT_DIR="${CARDS_CKPT_DIR:-trained_models_new/celeba}"
export CELEBA_ROOT="${CELEBA_ROOT:-/data/hkerner/Datasets/CelebA/celeba}"
export CELEBA_HQ_ROOT="${CELEBA_HQ_ROOT:-/data/hkerner/Datasets/CelebAMask-HQ}"
export CARDS_SAM_CACHE="${CARDS_SAM_CACHE:-/scratch/$USER/cards_sam_cache_celeba}"   # shared with the other CelebA EAC runs
export SHORTCUT_OUT="${SHORTCUT_OUT:-results/shortcut_baselines}"
export SHORTCUT_FEAT_ROOT="${SHORTCUT_FEAT_ROOT:-/scratch/$USER/shortcut_feats}"
SAM_REPO="${SAM_REPO:-/data/hkerner/$USER/segment-anything}"
SAM_CKPT="${SAM_CKPT:-/data/hkerner/$USER/sam_weights/sam_vit_h_4b8939.pth}"
METHODS="${METHODS:-feats sae shap eac}"
CK="$CARDS_CKPT_DIR/resnet18_official_train_${TASK,,}_shortcut_${RATE}.pt"
[ -f "$CK" ] || { echo "missing $CK"; exit 1; }
echo "task=$TASK rate=$RATE methods=$METHODS ckpt=$CK out=$SHORTCUT_OUT"
mkdir -p "$CARDS_SAM_CACHE" "$SHORTCUT_FEAT_ROOT"
RUN="uv run --locked python scripts/celeba/shortcut_baselines/run_baseline.py $TASK $RATE"
FEAT="$SHORTCUT_FEAT_ROOT/feat_${TASK,,}_${RATE}"

for M in $METHODS; do
  case "$M" in
    feats) [ -f "$FEAT/pooled_val.pt" ] && [ -f "$FEAT/map_train40k.pt" ] && { echo "features exist, skipping"; continue; }
           $RUN feats ;;
    sae|shap) $RUN $M ;;
    eac)   [ -f "$SAM_CKPT" ] && [ -d "$SAM_REPO/segment_anything" ] || { echo "missing SAM checkpoint or repo"; exit 1; }
           PYTHONPATH="$SAM_REPO" $RUN eac "$SAM_CKPT" ;;
    *)     echo "unknown method $M"; exit 1 ;;
  esac
done
# feature maps are only needed by sae/shap (~2.5 GB per checkpoint); keep the small pooled files
[[ " $METHODS " == *" shap "* ]] && rm -f "$FEAT"/map_*.pt
echo "done"
