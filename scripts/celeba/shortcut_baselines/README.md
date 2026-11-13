# Shortcut experiment: extra baselines

Adds ConceptSAE, ConceptSHAP (original and signed) and EAC to the official-train shortcut experiment. That experiment
has 22 single-task ResNet18s: Attractive and Male, each trained with a coloured-patch shortcut injected into
0, 10, …, 100% of the training images. The masking hybrid, TCAV and PCBM ×3 already have results in `results/`.

| File | Role |
|---|---|
| `run_baseline.py TASK RATE METHOD [SAM_CKPT]` | Runs `scripts/celeba/baselines/{extract_feats,concept_sae,concept_shap,celeba_eac}.py` unchanged on one shortcut checkpoint |
| `sol/run.sh` | Sol array, one task per checkpoint: feats → sae → shap → eac |
| `summarize.py` | Every method side by side: mean \|score\| over the 26 concepts per task and rate, and its correlation with the inverted accuracy gap (all 11 rates, and 0–90%) |

## How the checkpoints are adapted

- **Head.** The baseline scripts expect the 4-way Attractive/Male head. `run_baseline.py` gives them the shortcut
  model's 2-way head copied into both task slots, and keeps only the rows of the checkpoint's own task.
- **Images.** All methods see clean images, with no shortcut cue, as the existing shortcut runs do.
- **Design and settings.** Each baseline's design and settings are those of `scripts/celeba/baselines/`.
- **Reproduction check.** `summarize.py` reproduces the stored correlations of the five existing methods
  (`celeba_official_train_shortcut_inverted_gap_vs_all_methods_pearson.csv`).

## Sol

```bash
uv sync --locked && mkdir -p logs
sbatch -p htc scripts/celeba/shortcut_baselines/sol/run.sh
uv run --locked python scripts/celeba/shortcut_baselines/summarize.py
```
