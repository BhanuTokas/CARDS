#!/bin/bash
#SBATCH --job-name=tcav_negsrc
#SBATCH --account=grp_hkerner
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/data/hkerner/rnair21/CARDS/results/coco_stuff/logs/tcav_negsrc_%j.out
#SBATCH --error=/data/hkerner/rnair21/CARDS/results/coco_stuff/logs/tcav_negsrc_%j.err

# TCAV with Broden-negative and Grounding-DINO presence/absence CAV examples, all 3 black boxes.
# Submit from the repo root. LIMIT=3 sbatch ... runs a smoke test (3 concepts, *_smoke.csv).
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/rnair21/.cache/huggingface TORCH_HOME=/scratch/rnair21/.cache/torch XDG_CACHE_HOME=/scratch/rnair21/.cache
module load gcc/12.5.0
module load cuda-12.4.1-gcc-12.1.0
cd /data/hkerner/rnair21/CARDS
for BB in coco_stuff coco_stuff_vit_b_16 coco_stuff_convnext_tiny; do
  for V in random broden coco; do
    echo "=== $BB / $V ==="
    BACKBONE_NAME=$BB VARIANT=$V /scratch/rnair21/envs/cards_env/bin/python scripts/coco_stuff/run/run_tcav_coco_stuff_negsource_variants.py || echo "FAILED $BB $V"
  done
done
