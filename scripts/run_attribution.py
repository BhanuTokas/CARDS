"""Hydra entrypoint stub — wires configs/config.yaml to the cards pipeline.

Usage (from the repo root):
    uv run python scripts/run_attribution.py
    uv run python scripts/run_attribution.py encoder=open_clip_h retrieval=naive
    uv run python scripts/run_attribution.py -m encoder=clip,open_clip_h,siglip  # sweep
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from cards.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))
    raise NotImplementedError("wire up cards.concepts/retrieval/directions/attribution here")


if __name__ == "__main__":
    main()
