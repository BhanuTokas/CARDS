# COCO-Stuff shortcut experiment: the new baselines

The COCO-Stuff counterpart of `scripts/celeba/shortcut_baselines/` (on `main`). It runs ConceptSAE, ConceptSHAP (original
and signed) and EAC on the 11 ResNet18 shortcut checkpoints of `build/train_coco_stuff_shortcut_classifiers.py`. Then it
adds them to the paper's COCO-Stuff shortcut table and figure.

| Step | Script | Output in `$SHORTCUT_OUT` (default `results/coco_stuff/shortcut_baselines`) |
|---|---|---|
| Per checkpoint | `run_baseline.py RATE METHOD [SAM_CKPT]` | `rate_{rate}/{sae,shap,eac}.csv` |
| Table and figures | `summarize.py` | `shortcut_vs_gap.csv`, `shortcut_all_methods_mean_abs.csv`, `shortcut_{all_methods,new_baselines}.{png,pdf}` |

- **Wrapper.**
  - `run_baseline.py` replaces `coco_common.load_black_box` with a loader for the shortcut checkpoint. Architecture,
    20-way head and preprocess are the same as the main model.
  - It runs the scripts in `scripts/coco_stuff/baselines/` unchanged.
  - Every method sees clean images, as in the existing CARDS, TCAV and PCBM shortcut runs.
- **Metric** (the paper's COCO-Stuff column):
  - Per rate: mean |score| over every (concept, class) row.
  - Correlated with the inverted accuracy gap (clean − shortcut test accuracy), over the 11 rates and over 0–90%.
  - Reports Pearson r (the paper's) and Spearman ρ.
- **The paper's five methods** come from the values in `analysis/plot_coco_stuff_shortcut_gap_vs_methods.py`. They
  reproduce the paper's column: .82, −.42, −.28, −.62, −.63.
- **Sol.** `sol/run.sh` runs in this order:
  1. `labels`: the shared concept-label cache, run once.
  2. `feats sae shap`: an 11-task array.
  3. EAC: rate 0 first, which fills the shared SAM cache, then rates 10–100. EAC is resumable.

  The header of `sol/run.sh` has the exact commands.
