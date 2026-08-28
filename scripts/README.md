# Scripts

`run_attribution.py` is the general-purpose Hydra CLI entrypoint (wraps `cards.pipeline.run`) and is
the only script kept at this top level. Everything else is organized by track/purpose below.

Note: `notes/pcbm_correlation_investigation.md` and `notes/cub_correlation_investigation.md` reference
these scripts by their **historical** paths (flat under `scripts/`, since that's where they lived when
each note entry was written). This README reflects the current, reorganized layout — use it to find
where a script actually lives now.

## `imagenet/` — ImageNet + Broden track

The original investigation track: native ImageNet-1000 classification + the NetDissect Broden concept
bank, ground truth via masking-based faithfulness.

- `run_cards_imagenet.py`, `run_tcav.py`, `train_pcbm_surrogate.py` — the three methods run against this track.
- `run_broden_faithfulness.py` — builds the masking-based faithfulness ground truth.
- `score_all_methods_against_faithfulness.py` — scores all three methods against it.
- `broden_purity_check.py`, `broden_label_flags.py` — Broden concept-bank data-quality diagnostics.
- `build_imagenet_slice.py` — builds the local ImageNet class subset used for this track.

## `cub/` — CUB-200-2011 track

The larger, later track (`notes/cub_correlation_investigation.md`, v33 onward).

- **`build/`** — concept-bank / CAV construction: `build_cub_part_concept_bank.py`,
  `fit_cub_part_cavs.py`, `fit_cub_official_cavs_siglip.py`.
- **`ground_truth/`** — the masking-based faithfulness ground truth builders:
  `run_cub_faithfulness.py` (8-part bank), `run_cub_attribute_faithfulness.py` (87-attribute bank,
  class-stratified sampling), `run_cub_mask_approximation_check.py`.
- **`run/`** — the "official" per-method runs producing the full score matrices used throughout the
  investigation: `run_cards_cub.py` / `run_cards_cub_attributes.py` (also the shared source of
  `PREFIX_TEMPLATES`, imported by most `ablate/` and `analysis/` scripts), the `run_tcav_cub*.py`
  family, and the `train_pcbm_*_cub*.py` family (surrogate/CAV-based and concepts-only variants).
- **`ablate/`** — CARDS hyperparameter ablation grids (K, demean, phrasing, encoder, retrieval
  strategy, orthogonalization, normalization, top-concept-identification variants).
  `ablate_cards_cub_attributes.py` is also the shared source of `ALT_PREFIX_TEMPLATES` /
  `ENCODER_CONFIGS`, imported by several sibling ablation and analysis scripts.
- **`analysis/`** — diagnostics and evaluation: angle-diagnostic checks, the blur/zero-fill/hue-shift
  masking-strategy comparison, top-concept identification and recall@N analyses, the multi-concept
  masking additivity test, and `score_all_methods_against_cub_faithfulness.py` (the main Part 1/Part 2
  scoring script).
- **`plots/`** — visualization scripts (predicted-vs-actual scatter grid, masking-strategy-bias bars,
  winner's-curse comparison).

## Cross-script imports

A handful of scripts import shared constants directly from two "hub" files rather than duplicating
them:
- `run_cards_cub_attributes.py` (in `run/`) — `PREFIX_TEMPLATES`.
- `ablate_cards_cub_attributes.py` (in `ablate/`) — `ALT_PREFIX_TEMPLATES`, `ENCODER_CONFIGS`.

Any script that needs either adds both `cub/run/` and `cub/ablate/` to `sys.path` at the top (look for
the `sys.path.insert(0, str(Path(__file__).parent.parent / "run"))` / `.../ "ablate"` lines) — this
works from any `cub/*/` subfolder regardless of which one the importing script itself lives in.
