#!/bin/bash
# One SLURM job = one (model_type, image_set) pair, matching this repo's own
# one-job-per-unit HPC convention (see notes/ftw_correlation_investigation.md's
# FTW split). Submit with, e.g.:
#   sbatch --export=MODEL_TYPE=blip,IMAGE_SET=original scripts/captioning/build/submit_caption_generation.sh
#   sbatch --export=MODEL_TYPE=llava,IMAGE_SET=masked scripts/captioning/build/submit_caption_generation.sh
# for each of the 5 models x 2 image sets (10 total jobs).
#
# `cd "$SLURM_SUBMIT_DIR"` (not `dirname "$0"`) because SLURM copies the
# submitted script into its own spool location before running it --
# confirmed the hard way during the FTW build (ftw_correlation_investigation.md).
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -p general
#SBATCH -q public
#SBATCH -t 1-00:00:00
#SBATCH -o logs/caption_gen_%x_%j.out
#SBATCH -e logs/caption_gen_%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
if [ ! -f pyproject.toml ]; then
    echo "ERROR: pyproject.toml not found in $SLURM_SUBMIT_DIR -- SLURM_SUBMIT_DIR is wrong." >&2
    exit 1
fi

: "${MODEL_TYPE:?Set MODEL_TYPE (vit_gpt2|blip|florence|llava|bakllava) via --export=MODEL_TYPE=...}"
: "${IMAGE_SET:?Set IMAGE_SET (original|masked) via --export=IMAGE_SET=...}"
MAX_LEN="${MAX_LEN:-200}"

mkdir -p logs
echo "model_type=$MODEL_TYPE image_set=$IMAGE_SET max_len=$MAX_LEN"

uv run --extra captioning python scripts/captioning/build/generate_captions_with_logprobs.py \
    --model_type "$MODEL_TYPE" --image_set "$IMAGE_SET" --max_len "$MAX_LEN"
