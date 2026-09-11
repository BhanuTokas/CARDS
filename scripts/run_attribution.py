"""Hydra entrypoint -- wires configs/config.yaml to cards.pipeline.run.

Usage (from the repo root):
    uv run python scripts/run_attribution.py concepts='[red_wing]' dataset=cub
    uv run python scripts/run_attribution.py concepts='[red_wing,black_wing]' dataset=cub encoder=open_clip_h
    uv run python scripts/run_attribution.py -m concepts='[red_wing]' retrieval=matched,naive  # sweep

    # CelebA masking hybrid, best-known config (see configs/dataset/celeba.yaml
    # for the caveats behind this exact combination):
    uv run python scripts/run_attribution.py dataset=celeba encoder=siglip model=celeba_attractive_young scoring_mode=masking_hybrid orthogonalize=true concepts='["a person with arched eyebrows"]'

`concepts` has no default and must be set on the CLI -- there's no
universally sensible concept to run by default. See cards.pipeline for the
actual Steps 1-7 orchestration this delegates to.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from cards.pipeline import run


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
