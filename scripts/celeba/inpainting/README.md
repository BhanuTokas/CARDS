# Inpainting ground truth for CelebA (Attractive / Male, official-train ResNet18)

Replaces the blur fill of the CelebAMask-HQ faithfulness ground truth with inpainting, adds a random-shape
control per inpainter, checks how much each fill actually removes the concept, and ranks every method against
each resulting ground truth. Built on the strict-control ground truth (`scripts/celeba/ground_truth/`, records in
`results/celeba_official_train_faithfulness_non_overlapping.csv`).

Run everything from the repo root. Paths are set in `paths.py` (environment overrides: `CARDS_INPAINT_WORK`,
`CARDS_INPAINT_BUNDLE`, `CARDS_INPAINT_FILLS`, `CARDS_LAMA_WEIGHTS`, `CARDS_LAMA_OUT`, `CARDS_CLIP_L14`,
`CARDS_INPAINT_REVIEW`; the baselines use `CARDS_FEAT_CACHE`). Fills, masks and weights are not in git.

## Definitions
- Concept fill (`m0`): the concept's CelebAMask-HQ region, dilated 8 px, inpainted and pasted back inside the mask only.
- Random-shape fills (`m1`-`m5`): equal-area smooth random shapes outside the dilated region, same inpainter.
- Adjusted GT per (task, concept) = mean over images of [logit drop for `m0`] - [mean logit drop for `m1`-`m5`].
- Validity (pre-set): all 5 random shapes placed, fill fraction <= 0.8, >= 95% of a concept's images valid.

## Pipeline
| Step | Script | Output (work dir unless noted) |
|---|---|---|
| Blur random-shape control | `blob_control.py OUT.csv` | blur GT + control |
| LaMa GT + control | `lama_gt.py lama_full.csv` (needs big-lama.pt) | `lama_full.csv`, `$CARDS_LAMA_OUT` |
| Export masks for other inpainters | `export_masks.py` | `$CARDS_INPAINT_BUNDLE/{manifest.csv,masks/}` |
| Diffusion fills (Sol) | `sol_inpaint/` (`run_inpaint.sh`, SD-2 / SDXL / FLUX.1-Fill, neutral, quality-negative and concept-negative prompts) | `fills/{model}_{neg}_s{seed}/` |
| LaMa fills in the same format | `lama_fills.py` | `fills/lama_s0/` |
| Score fills with the black box | `score_fills.py FILLS/RUN fills_RUN.csv` | `fills_RUN.csv` |
| Build GT CSVs, inpainter agreement | `build_inpaint_gts.py` | `gt_*.csv` |
| Removal check (CLIP ViT-L/14 probes; diagnostic only) | `removal_check.py`, `removal_an.py`, `removal_runs.py`, `removal_runs_an.py` | `removal_*.csv` |
| VLM judge (Sol; diagnostic only) | `sol_vlm_judge/` (Qwen2.5-VL-7B yes/no retention + realism) | `out/*.csv` |
| CARDS nulls / moved masks | `hq_nulls.py`, `random_cards.py` | `random_cards_full.csv` |
| Rank methods | `rank_gts.sh NAME ...` (wraps `rank_methods.py`) | `rank_NAME.txt` |
| Visual review | `review_sheets.py`, `compare_sheets.py` | `$CARDS_INPAINT_REVIEW` |

The extra baselines ranked here (ConceptSAE, ConceptSHAP, EAC, KernelSHAP-SAM, soft CARDS) are produced by
`scripts/celeba/baselines/` on the `celeba_baselines` branch; `rank_gts.sh` only reads their output CSVs.

## Notes
- The removal detector and VLM judge are diagnostics. They order inpainters by how much they remove; they never
  select fills.
- Sol jobs (`sol_inpaint/`, `sol_vlm_judge/`) are standalone uv projects; copy the folder (plus the exported
  bundle for `sol_inpaint`) to Sol and follow the header of the `.sh` file.
