# CARDS

[![CI](https://github.com/BhanuTokas/CARDS/actions/workflows/ci.yml/badge.svg)](https://github.com/BhanuTokas/CARDS/actions/workflows/ci.yml)

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
  models/                 Pluggable black-box model protocol for Step 6/7 (see BlackBoxModel)
  data/                    Dataset loaders: CIFAR-10/100, CUB, CCE MetaDataset, Broden
  validation/               Design-doc Section 2 validation checks (currently: Broden retrieval purity)
  pipeline.py                 Orchestrates Steps 1-7 end to end, driven by configs/config.yaml
  utils/                        Seeding etc.

configs/            Hydra configs — see "Configs with Hydra" below
scripts/            CLI entrypoints (Hydra-wired attribution pipeline + standalone validation scripts)
tests/               pytest
```

Steps 1–7 are implemented and tested; `scripts/run_attribution.py` wires them
together end to end (Steps 1–5 always run, Steps 6–7 run once a real
black-box model is configured under `configs/model/` — `none` is the
default, which just skips scoring). `cards/data/datasets.py`'s loaders are
implemented for CIFAR-10/100, CUB, and Broden; MetaDataset's loader assumes
an ImageFolder-style layout that hasn't been checked against a real release
copy yet.

**Black-box models (Steps 6–7)**: `configs/model/pcbm_cub.yaml` wires in a
trained [Post-hoc CBM](https://github.com/mertyg/post-hoc-cbm) CUB species
classifier via `cards.models.posthoc_cbm.PosthocCBMBlackBox` — verified
end to end (`model.forward` correctly identifies held-out species by
argmax, and a full attribution run on `red`/`black` for `Scarlet_Tanager`
gives biologically sensible signs: black wings help the score, plain red
hurts it, matching that species' actual field markings). Needs
`uv sync --extra pcbm` (adds `pytorchcv`, for the `resnet18_cub` backbone)
and `../post_hoc_cbm` checked out as a sibling directory:

```
uv sync --extra pcbm
uv run python scripts/run_attribution.py concepts='[red,black]' dataset=cub model=pcbm_cub model.target_class=Scarlet_Tanager
```

`model.target_class` is any of the checkpoint's 200 CUB species names
(defaults to `Scarlet_Tanager`, the design doc's own worked example). New
black-box models plug in the same way: implement `cards.models.base.
BlackBoxModel` (`preprocess()` + `__call__()`) and add a `configs/model/
*.yaml` with a Hydra `_target_`.

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
  (`broden` isn't usable as a retrieval pool through `run_attribution.py` —
  it's ground-truth pairs, not a pool; see `cards.data.datasets.load_broden`
  and the validation scripts below instead)
- `configs/normalization/` — `variance` (default), `embedding_distance`
- `configs/model/` — `none` (default, skips Steps 6-7), `pcbm_cub`

Note: `hydra.job.chdir` is explicitly set to `false` in `config.yaml` — by
default Hydra changes the process's working directory to the run's output
folder, which would break the `../Datasets/...` relative paths used in
`configs/dataset/*.yaml`.

**Embedding cache**: candidate pool embeddings are cached to disk under
`cache_dir` (default `embedding_cache/`, override with `cache_dir=...`),
keyed on the `(dataset, pool_source, encoder)` combination — retrieval
strategy, normalization, and model don't affect the pool, so a `-m` sweep
over those axes reuses one cached encoding pass per distinct dataset+encoder
instead of paying for it on every job. Deleting `embedding_cache/` (or a
single stale `.pt` file inside it) forces a recompute; the cache also
self-invalidates if the underlying `(dataset, split)` pool contents change.

## Validation scripts

Standalone scripts (plain `argparse`, not Hydra-wired — they're one-off
checks, not part of the ablation grid) for the design doc's Section 2
validation experiments:

- **`scripts/broden_purity_check.py`** — Section 2 item 5: precision@k /
  negative-recall@k / average precision of CARDS' Step 1-2 CLIP retrieval
  against Broden's ground-truth positives/negatives, per concept. Useful for
  setting a per-concept reliability gate (mirroring CCE's 0.7-accuracy
  filter) before trusting attribution scores for a given concept.
- **`scripts/broden_label_flags.py`** — surfaces the ground-truth positives
  CLIP ranks least like the concept, and the negatives it ranks most like
  the concept, as candidates for manual label-quality review.
- **`scripts/cub_concept_bank_accuracies.py`** — recovers human-readable
  names for the anonymous 112-concept bank baked into `../post_hoc_cbm`'s
  trained CUB PCBM checkpoint (the standard Concept Bottleneck Models
  312→112 CUB attribute filtering) and reports each concept's train/test CAV
  accuracy, for auditing that checkpoint before reusing it as CARDS' Step
  6-7 black-box model.

Both Broden scripts default to checking all 170 concepts, which takes a few
minutes (CLIP has to encode every image in each concept's positive/negative
set). Pass `--concepts` to check a subset instead, e.g.
`--concepts dog air_conditioner`.

Reusable logic behind the two Broden scripts lives in
`cards.validation.broden_purity`, tested independently of any real CLIP
model or dataset in `tests/test_broden_purity.py`.

### Finding: the Broden concept dataset has real label corruption

Initial validation found that concepts with notably low retrieval-purity AP
are usually mislabeled, not just visually hard — e.g. a `knob`-labeled image
is actually a windmill, a `street_s`-labeled image is actually a dam. 9/9
checked low-AP concepts showed this; two high-AP controls (`dog`, `bus`)
didn't. Traced as far as possible, this is a real data-quality issue in the
pre-packaged, third-party Broden concept dataset itself (likely originating
from the CCE paper's experimental setup), not a bug anywhere in this repo.

Full writeup, evidence, and root-cause tracing: **[docs/broden_label_corruption.md](docs/broden_label_corruption.md)**.

All three scripts write into `results/` by default (`--output` overrides
this). `results/` already has the full-170-concept / full-112-concept
outputs checked in from initial validation, so a reader can inspect the
findings directly without rerunning anything.
