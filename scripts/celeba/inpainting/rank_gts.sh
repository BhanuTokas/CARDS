#!/bin/bash
# Rank all methods against one or more GT CSVs:  scripts/celeba/inpainting/rank_gts.sh NAME [NAME ...]
# Reads $CARDS_INPAINT_WORK/gt_NAME.csv (build_inpaint_gts.py) and writes rank_NAME.txt next to it; prints the
# logit-scale pooled block. Run from the repo root with post_hoc_cbm on PYTHONPATH (PCBM checkpoints).
# Baseline score CSVs expected in the work dir: sae_scores.csv, cshap_scores.csv (scripts/celeba/baselines on the celeba_baselines branch).
# More methods: EXTRA2="label|csv|column;..." (CSV with task, concept_name columns), e.g. EAC, KernelSHAP-SAM, soft CARDS.
# Restrict concepts: KEEP=Concept_A,Concept_B
set -euo pipefail
WORK="${CARDS_INPAINT_WORK:-results/inpainting}"
PY="${PY:-python}"
HERE="$(dirname "$0")"
export EXTRA="ConceptSAE|$WORK/sae_scores.csv|score;ConceptSHAP orig|$WORK/cshap_scores.csv|shap_acc;ConceptSHAP signed|$WORK/cshap_scores.csv|shap_logit${EXTRA2:+;$EXTRA2}"
for n in "$@"; do
  "$PY" "$HERE/rank_methods.py" "$WORK/gt_$n.csv" "$WORK/random_cards_full.csv" > "$WORK/rank_$n.txt"
  echo "######## $n"; grep -E "scored" "$WORK/rank_$n.txt"
  sed -n '/=== scale logit/,/=== scale logodds/p' "$WORK/rank_$n.txt" | grep -v "=== scale logodds"
done
