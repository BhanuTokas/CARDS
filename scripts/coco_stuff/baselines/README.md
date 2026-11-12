# COCO-Stuff baselines (ConceptSAE, ConceptSHAP, EAC)

Ports of `scripts/celeba/baselines/` to the COCO-Stuff 20-class multi-label black boxes. Pick the black box with
`CARDS_ARCH=resnet18|vit|convnext` (default `resnet18`); everything else is the same for the three. Run from the repo
root (the checkout that holds `results/`, `weights/`, `trained_models_new/`).

| Method | Script | Output columns |
|---|---|---|
| Features + concept labels | `extract_feats_coco.py` | `$CARDS_FEAT_CACHE` (default `results/coco_stuff/baselines/feat_cache[_vit\|_convnext]`) |
| ConceptSAE | `concept_sae_coco.py OUT.csv` | `concept_name, class_name, score` (`weight_score` variant) |
| ConceptSHAP | `concept_shap_coco.py OUT.csv` | `shap_acc` (original), `shap_logit` (signed) |
| EAC | `eac_coco.py SAM_CKPT OUT.csv` | `score` |

- Concept presence labels = Grounding DINO detections from `coco_stuff_faithfulness.csv` (the images the ground truth is
  measured on); computed once and shared by all architectures (`results/coco_stuff/baselines/concept_labels.npz`).
- Set `GT_MASKS_DIR` (the `SAVE_MASKS` dir of `build/compute_coco_stuff_faithfulness_variants.py`) so EAC reuses the
  ground-truth concept masks instead of rerunning Grounding DINO. Set `CARDS_SAM_CACHE` so the three black boxes share SAM segments.
- Needs `PYTHONNOUSERSITE=1` on SOL (a stale `~/.local` OpenCV otherwise shadows the env's).
- Scoring: `EXTRA="ConceptSAE|OUT.csv|score;..." python scripts/coco_stuff/analysis/score_cohens_d_bootstrap_coco_stuff.py`
  (ResNet18) or `EXTRA="coco_stuff_vit_b_16:ConceptSAE|OUT.csv|score;..."` with `compare_variants_own_gt.py`.

```bash
export CARDS_ARCH=vit PYTHONNOUSERSITE=1 CARDS_SAM_CACHE=results/coco_stuff/baselines/sam_cache
export GT_MASKS_DIR=results/coco_stuff/gt_masks
D=results/coco_stuff/baselines/vit; mkdir -p $D
python scripts/coco_stuff/baselines/extract_feats_coco.py
python scripts/coco_stuff/baselines/concept_sae_coco.py  $D/sae.csv
python scripts/coco_stuff/baselines/concept_shap_coco.py $D/cshap.csv
python scripts/coco_stuff/baselines/eac_coco.py weights/sam_vit_h_4b8939.pth $D/eac.csv
```
Status: scripts have only been syntax-checked; `feats()` was verified to reproduce each model's output on CPU.
