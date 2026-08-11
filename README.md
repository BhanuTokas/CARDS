# CARDS

CARDS: Concept Attribution via Retrieved Distribution Shift

A black-box concept attribution method. Instead of generating counterfactual
images, CARDS retrieves real images from a candidate pool whose CLIP-space
distribution shift aligns with a target concept, and measures how a
black-box classifier's outputs differ across that shift — in two modes,
population-level (global) and single-instance (local).

See the design doc (shared separately) for the full method spec, validation
plan, and ablations. This README covers project setup only.

## Setup

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/):

```
uv sync
```

If `uv` isn't on your `PATH` (e.g. it's only installed inside a conda
`base` env), either activate that env first or invoke `uv` via its full
path / `conda run -n base uv ...` instead — same commands otherwise.

This creates an isolated `.venv` in the project directory (Python 3.11,
downloaded by uv itself — kept separate from any conda envs) and installs
everything from `pyproject.toml` / `uv.lock`, including a CUDA build of
torch pinned to the `pytorch-cu124` index (see `[tool.uv.sources]` in
`pyproject.toml`).

Run anything inside the environment with `uv run`, e.g.:

```
uv run pytest
uv run python scripts/run_attribution.py
```

Datasets live outside the repo at `../Datasets` (sibling to `CARDS/`), matching
how other projects on this machine are laid out — see `configs/dataset/*.yaml`
for the exact subfolder each dataset expects (e.g. `CUB_200_2011`,
`broden_concepts`, both already present there; `CIFAR10`/`CIFAR100`/
`MetaDataset` still need to be fetched).

## Project layout

```
src/cards/
  concepts/       Step 1 — concept prompt construction (prompt ensembling -> t_c)
  encoders/        CLIP / OpenCLIP-H / SigLIP, behind a shared protocol (Step 6 ablation)
  retrieval/        Step 2 (top-k/bottom-k) + Step 3 (confound-matched / stratified retrieval)
  directions/         Step 4 (diff-of-means direction) + Step 5 (Löwdin orthogonalization)
  attribution/          Step 6 (global TCAV-style / local CCE-style scoring) + Step 7 (normalization)
  data/                   Dataset loaders: CIFAR-10/100, CUB, CCE MetaDataset, Broden
  utils/                   Seeding etc.

configs/            Hydra configs — see "Configs with Hydra" below
scripts/            CLI entrypoints (Hydra-wired)
tests/               pytest
```

Every module under `src/cards` currently contains stub function signatures
with `NotImplementedError` bodies, referencing the specific design-doc step
each one implements. Steps 1–4 are the ones ready to fill in first.

## Configs with Hydra

This project uses [Hydra](https://hydra.cc/) to manage the experiment config
because the ablation plan (Section 3 of the design doc) crosses several
largely-independent axes — encoder, retrieval strategy, dataset, score
normalization — and Hydra lets you swap any one of them from the command
line without editing files or writing bespoke argparse plumbing.

**The mental model**: `configs/config.yaml` is the top-level config. Its
`defaults:` list picks one option from each *config group* — a subdirectory
like `configs/encoder/` that holds one YAML file per alternative
(`clip.yaml`, `open_clip_h.yaml`, `siglip.yaml`). At runtime Hydra merges the
selected file from each group into one config object (`cfg` in
`scripts/run_attribution.py`), which behaves like a nested dict/object
(`cfg.encoder.model_name`, `cfg.k`, etc. — backed by
[OmegaConf](https://omegaconf.readthedocs.io/)).

Run with defaults:

```
uv run python scripts/run_attribution.py
```

Override one group from the CLI (swap the encoder, keep everything else):

```
uv run python scripts/run_attribution.py encoder=open_clip_h
```

Override a plain value:

```
uv run python scripts/run_attribution.py k=100
```

Sweep — run the multi-run flag `-m` to launch one job per combination
(this is how the Section 3 ablation grid gets executed in practice):

```
uv run python scripts/run_attribution.py -m encoder=clip,open_clip_h,siglip retrieval=matched,naive
```

That example launches 6 jobs (3 encoders × 2 retrieval strategies), each
with its own output directory under `multirun/<date>/<time>/`. Each
single-run job's config snapshot + logs go under `outputs/<date>/<time>/`
(see `output_dir` in `configs/config.yaml`).

Config groups currently defined:

- `configs/encoder/` — `clip` (default), `open_clip_h`, `siglip`
- `configs/retrieval/` — `matched` (default), `stratified`, `naive`
- `configs/dataset/` — `cifar10` (default), `cifar100`, `cub`, `metadataset`, `broden`
- `configs/normalization/` — `variance` (default), `embedding_distance`

Note: `hydra.job.chdir` is explicitly set to `false` in `config.yaml` — by
default Hydra changes the process's working directory to the run's output
folder, which would break the `../Datasets/...` relative paths used in
`configs/dataset/*.yaml`.
