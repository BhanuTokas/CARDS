#!/bin/bash
#SBATCH --job-name=conceptmask_ftw_build
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 0-04:00:00   # unknown until a full-dataset timing is measured -- covers pool embedding + localization + best-of-family masking for ALL K x 19 concepts
#SBATCH -o logs/conceptmask_ftw_build_%j.out
#SBATCH -e logs/conceptmask_ftw_build_%j.err
#
# GPU stage of the two-job split. Covers BOTH real-compute encoder uses:
# (1) the one-time SigLIP embedding of the whole tile pool, and (2) per-
# concept localization (localize_concept) + best-of-family fill selection
# (encode_images) for every present tile -- direct feedback: "wouldn't
# concept masking also require GPU as we verify which type of masking
# works best?" -- correct, both need GPU, not just (1) (an earlier,
# corrected assumption -- see notes v10).
#
# Writes orig/masked GeoTIFF pairs + a manifest to FTW_MASKED_TILES_DIR.
# Run scripts/ftw/run/submit_ftw_hpc.sh (CPU-only, the ftw_tools CLI
# inference stage -- measured to get ZERO benefit from GPU) on that
# manifest AFTER this job completes.

module purge
export PATH="$HOME/.local/bin:$PATH"   # reach uv -- one-time login-node install: curl -LsSf https://astral.sh/uv/install.sh | sh
# `module purge` alone left a stale mamba-module PROJ database on the
# search path (observed: "CPLE_AppDefined in PROJ: ... DATABASE.LAYOUT.
# VERSION.MINOR = 3 whereas a number >= 6 is expected" from rasterio) --
# likely set by a system-wide hook rather than the module system itself,
# since purge didn't clear it. Our own code never reprojects/does EPSG
# lookups (only affine-preserving reads/writes), so this probably wasn't
# corrupting output, but unset it outright rather than rely on that.
unset PROJ_LIB PROJ_DATA GDAL_DATA

set -euo pipefail

# ---- must match submit_ftw_hpc.sh's FTW_MASKED_TILES_DIR exactly ----
export FTW_ROOT=/data/hkerner/FTW/ftw                                                                              # confirmed -- same nested "ftw" subfolder as the local dev machine's own layout
export FTW_MASKED_TILES_DIR=/data/hkerner/btokas/Datasets/ftw_masked_tiles                                         # confirmed
export FTW_EMBEDDING_CACHE=/data/hkerner/btokas/Datasets/FTW_embedding_cache/ftw_tiles_siglip.pt                   # confirmed
export FTW_K=50   # scale-up value for the full HPC run, confirmed 2026-09-06 (local pilot was K=10)

mkdir -p logs "$FTW_MASKED_TILES_DIR" "$(dirname "$FTW_EMBEDDING_CACHE")"

# SLURM copies the submitted script into its own spool location and runs
# it from there, so `dirname "$0"` does NOT reliably point back at the repo
# -- observed directly: it silently landed uv run outside any pyproject.toml,
# which fell back to system /usr/bin/python3 and then failed to find the
# (correct, relative) script path. $SLURM_SUBMIT_DIR is the directory
# `sbatch` was invoked FROM, which is what we actually want -- submit this
# job from the CARDS repo root.
cd "$SLURM_SUBMIT_DIR"
if [ ! -f scripts/ftw/build/build_ftw_masked_tiles.py ]; then
    echo "ERROR: scripts/ftw/build/build_ftw_masked_tiles.py not found in $(pwd)." >&2
    echo "Submit this job with sbatch from the CARDS repo root (SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR)." >&2
    exit 1
fi

# --extra ftw: rasterio lives in CARDS' own optional "ftw" dependency group
# in pyproject.toml -- a bare `uv run` only installs the default set and
# misses it (observed directly: ModuleNotFoundError: No module named
# 'rasterio' on the first Sol run).
uv run --extra ftw python scripts/ftw/build/build_ftw_masked_tiles.py
