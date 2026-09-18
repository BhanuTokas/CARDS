#!/bin/bash
#SBATCH --job-name=captioning_broden_masked_images
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 1-00:00:00   # unknown until measured -- 170 concepts x K=50 retrieved images each,
                         # localization + best-of-7 fill-strategy selection per image (SigLIP only,
                         # no VLM captioning models involved -- that happens in a separate later job)
#SBATCH -o logs/captioning_broden_masked_images_%j.out
#SBATCH -e logs/captioning_broden_masked_images_%j.err
#
# Builds the Broden-concept masked-image set + manifest for the
# captioning-bias track (build_captioning_broden_masked_images.py) --
# SigLIP-only (retrieval + patch-similarity localization + masking), no
# transformers/VLM deps, so a plain `uv run` (no `--extra captioning`)
# is enough, unlike generate_captions_with_logprobs.py's own submission
# script. Caption generation for these masked images (the actual VLM
# forward passes) is a SEPARATE later step, not part of this job.

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh

set -euo pipefail

# ---- fill these in for the real Sol filesystem layout ----
# TODO: confirm -- inferred from submit_caption_generation.sh's own
# CAPTIONING_COCO_ROOT=/data/hkerner/Datasets/COCO/val2014 (i.e. this is
# that same path's PARENT, assumed to also contain train2014/ alongside
# val2014/ -- not yet directly verified on Sol).
export CAPTIONING_COCO2014_ROOT=/data/hkerner/Datasets/COCO
export CARDS_RESULTS_DIR=results
export CARDS_CACHE_DIR=embedding_cache

mkdir -p logs "$CARDS_RESULTS_DIR" "$CARDS_CACHE_DIR"

# SLURM copies the submitted script into its own spool location and runs it
# from there, so `dirname "$0"` does NOT reliably point back at the repo --
# $SLURM_SUBMIT_DIR is the directory `sbatch` was invoked FROM. Submit this
# job with sbatch from the CARDS repo root.
cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/captioning/build/build_captioning_broden_masked_images.py ]; then
    echo "ERROR: scripts/captioning/build/build_captioning_broden_masked_images.py not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi
if [ ! -d ../post_hoc_cbm ]; then
    echo "ERROR: ../post_hoc_cbm not found next to $(pwd) -- this script imports BRODEN_CONCEPTS from it." >&2
    exit 1
fi

uv run python scripts/captioning/build/build_captioning_broden_masked_images.py
