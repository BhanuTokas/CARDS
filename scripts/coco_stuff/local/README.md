# COCO-Stuff local (per-image) attribution: every method, ResNet18

The COCO counterpart of the CelebA local experiment (`scripts/celeba/baselines/local_*.py`): per-image scores of
Hide and Seek, TCAV, EAC and ConceptSAE on (concept, class) pairs with a per-image ground truth, then specificity /
fidelity scores and top-3 / bottom-3 grids for every method.

| Step | Script | Output in `$COCO_LOCAL_OUT` (default `results/coco_stuff_local/resnet18`) |
|---|---|---|
| Pairs and records | `make_records.py` | `pairs.csv`, `records.csv` |
| Hide and Seek | `local_cards.py cutoffs`, then `local_cards.py score` (sharded) | `cards_cutoffs.json`, `cards_*.csv` |
| TCAV (layer3; layer4 for reference) | `local_tcav.py` | `tcav.csv`, `tcav_cavs.pt` |
| EAC | `local_eac.py SAM_CKPT` (sharded, resumable) | `eac_*.csv` |
| ConceptSAE | `local_sae.py` | `sae_resnet18.pt`, `sae_global.csv`, `sae.csv` |
| Scores | `score_local.py` | `local_merged.csv`, `local_scores.csv`, `local_pair_scores.csv` |
| Grids | `plot_local.py` | `grids/{concept}_{class}.png` |

## Design

- **Records.**
  - Present: the concept was detected in the image by the ground-truth pipeline and the class is annotated in it.
    Ground truth = that image's `delta_p` in `coco_stuff_faithfulness.csv`.
  - Absent: the class is in the image but the concept was not detected. Ground truth = 0.
  - Up to 100 of each per pair.
- **Pairs.** Fixed before scoring. Among pairs with at least 100 present images, each class contributes three: the largest
  mean ground truth, the smallest, and the most images. The three cases of the existing COCO local figures are added too.
- **Hide and Seek.** Uses the COCO headline configuration: SigLIP, de-meaned and orthogonalized queries, z-score
  α = 1.5 from the top-50 val2017 retrievals, and best-of-7 fill. The score is the class logit drop.
- **TCAV.** Uses layer3 because the layer4 score is constant per (concept, class) for ResNet18, since only average
  pooling and the linear head follow layer4; `tcav4_dd` shows this. CAVs come from 40 Broden positives against 6
  random sets of 25 COCO images, and the classifier is seeded.
- **EAC.** Same procedure as `eac_coco.py`, on the saved ground-truth concept regions (`GT_MASKS_DIR`). Absent records
  score 0, because EAC has no region for an undetected concept.
  - **Caveat:** EAC is given the ground-truth detector's regions, so its present-vs-absent metrics (AUROC, AP,
    hit rates) are close to perfect by construction.
  - Compare EAC only on ρ, the effect fidelity among present records.
- **ConceptSAE.** The same SAE as `concept_sae_coco.py`, refitted and saved. Score = z_j(x) · (w_class · d_j).
- **Metrics.** Each metric is described in `score_local.py`'s docstring.
  - Pooled and within-pair Spearman ρ.
  - Hit@3/5/10.
  - AUROC and AP of |score|, present vs absent.

## Sol

Run `uv sync --locked` on the login node first. Then submit the dependency chain in the header of
`sol/run_local.sh`, in this order: prep, then cards / tcav / eac / sae in parallel, then score + plot. All steps fit
on `-p htc`.
