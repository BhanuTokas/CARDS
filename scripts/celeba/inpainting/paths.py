
"""Locations for the inpainting ground-truth pipeline. Every path can be overridden with an environment variable;
defaults are relative to the repo root (run every script from there). CelebA / CelebAMask-HQ roots come from
scripts/celeba/ground_truth/run_celeba_official_train_faithfulness.py, as for the existing ground truth."""
import os
from pathlib import Path


def _p(var: str, default) -> Path:
    return Path(os.environ.get(var, str(default))).expanduser()


WORK = _p("CARDS_INPAINT_WORK", "results/inpainting")           # analysis CSVs (inputs and outputs)
BUNDLE = _p("CARDS_INPAINT_BUNDLE", WORK / "bundle")            # manifest.csv + masks/ (export_masks.py)
FILLS = _p("CARDS_INPAINT_FILLS", WORK / "fills")               # {run}/{hq_idx}_{concept}_m{k}.jpg
LAMA_WEIGHTS = _p("CARDS_LAMA_WEIGHTS", "models/big-lama.pt")    # big-lama TorchScript (Sanster/IOPaint release)
LAMA_OUT = _p("CARDS_LAMA_OUT", WORK / "lama_out")              # 512 px LaMa concept fills written by lama_gt.py
CLIP_L14 = _p("CARDS_CLIP_L14", "~/.cache/clip/ViT-L-14.pt")     # OpenAI CLIP ViT-L/14 TorchScript (removal detector)
REVIEW = _p("CARDS_INPAINT_REVIEW", WORK / "review")            # visual review sheets
