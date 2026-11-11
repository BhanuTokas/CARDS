# CelebA baselines (official-train Attractive / Male classifiers)

Extra concept-attribution baselines for the CelebA experiments, for all three official-train black boxes
(ResNet18, ViT-B/16, ConvNeXt-Tiny), plus scorers against the paper's ground truth. CARDS, TCAV and the PCBM
variants live in `scripts/celeba/run/` and `scripts/celeba/build/`.

Run everything from the repo root. Pick the black box with `CARDS_ARCH=resnet18|vit|convnext` (default
`resnet18`; loaders in `black_boxes.py`, checkpoints in `trained_models_new/celeba/` or `$CARDS_CKPT_DIR`).
Image pool for all feature-based baselines = the clean official val split used by CARDS (official val minus every
CelebAMask-HQ image).

| Method | Script | Access | Output (task, concept_name, ...) |
|---|---|---|---|
| Features for SAE / ConceptSHAP | `extract_feats.py` | white-box | `$CARDS_FEAT_CACHE` (default `results/baselines/feat_cache[_vit\|_convnext]`) |
| ConceptSAE (TopK SAE, label-named latents) | `concept_sae.py OUT.csv` | white-box | column `score` |
| ConceptSHAP (Yeh et al. 2020) | `concept_shap.py OUT.csv` | white-box | `shap_acc` (original), `shap_logit` (signed) |
| EAC (Sun et al. 2023) | `celeba_eac.py SAM_CKPT OUT.csv` | grey-box (frozen head) | `score` |
| KernelSHAP over SAM segments | `celeba_kernelshap.py SAM_CKPT OUT.csv` | black-box | `score` |
| Soft CARDS (D - rho*A) | `celeba_soft.py FULL.csv`, then `soft_aggregate.py FULL.csv OUT.csv` | black-box | `soft` |

- EAC and KernelSHAP need Meta's `segment-anything` on `PYTHONPATH` and the SAM ViT-H checkpoint
  (`sam_vit_h_4b8939.pth`). Set `CARDS_SAM_CACHE=<dir>` to cache SAM masks so both methods and all three black boxes
  use identical segments. Both resume per image from `OUT.csv.images.csv`; `EAC_LIMIT=n` limits images for a smoke run.
- `extract_feats.py` must run before `concept_sae.py` and `concept_shap.py` (same `CARDS_ARCH`).
- ViT feature maps are the 14x14 patch tokens average-pooled to 7x7, so all three models give ConceptSHAP a 49-patch grid.

## Scoring
- `score_paper_gt.py OUT.csv`: every method against the paper's raw blur ground truth (`previously_used` and the
  clean `non_overlapping` image set), per task and pooled, with bootstrap CIs. Add baseline outputs with
  `EXTRA="label|csv|column;..."`. For ResNet18 it also loads CARDS, Cohen's d, TCAV and the PCBMs (needs
  `post_hoc_cbm` on `PYTHONPATH`); for ViT / ConvNeXt it reads main's per-model result files from
  `$CARDS_RESULTS_DIR` (default `results`).
- `score_control_gts.py OUT.csv` (ResNet18): the same methods against the random-mask-control-adjusted ground truths
  (permissive, strict, strict_excl) of `score_methods_against_strict_control_official_train.py`.

Example (ViT, all baselines):
```bash
export CARDS_ARCH=vit CARDS_SAM_CACHE=results/baselines/sam_cache PYTHONPATH=../segment-anything
D=results/baselines/vit; mkdir -p $D
python scripts/celeba/baselines/extract_feats.py
python scripts/celeba/baselines/concept_sae.py $D/sae.csv
python scripts/celeba/baselines/concept_shap.py $D/cshap.csv
python scripts/celeba/baselines/celeba_eac.py sam_vit_h_4b8939.pth $D/eac.csv
python scripts/celeba/baselines/celeba_soft.py $D/soft_full.csv && python scripts/celeba/baselines/soft_aggregate.py $D/soft_full.csv $D/soft.csv
EXTRA="ConceptSAE|$D/sae.csv|score;ConceptSHAP signed|$D/cshap.csv|shap_logit;EAC|$D/eac.csv|score;CARDS soft|$D/soft.csv|soft" \
  python scripts/celeba/baselines/score_paper_gt.py $D/paper_gt_scores.csv
```
