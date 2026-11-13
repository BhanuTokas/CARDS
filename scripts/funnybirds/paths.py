"""Locations and shared constants for the FunnyBirds experiments (Hesse et al., ICCV 2023).

Every path can be overridden with an environment variable, so the same scripts run on this machine and on Sol:
  FUNNYBIRDS_ROOT       the unzipped dataset (contains train/, test/, test_interventions/, classes.json, parts.json)
  FUNNYBIRDS_FRAMEWORK  a checkout of https://github.com/visinf/funnybirds-framework (used unmodified)
  FUNNYBIRDS_CKPT_DIR   folder with the authors' model checkpoints (*_checkpoint_best.pth.tar)
  FUNNYBIRDS_OUT        where results are written
  SEGMENT_ANYTHING      a checkout of https://github.com/facebookresearch/segment-anything (EAC only)
  SAM_CKPT              SAM ViT-H weights, sam_vit_h_4b8939.pth (EAC only)
Defaults assume sibling folders of the repo checkout: ../Datasets/FunnyBirds, ../funnybirds-framework,
../Datasets/FunnyBirds_models (set the variables when running from a worktree or on Sol).
"""
import os
from pathlib import Path

_PROJECTS = Path(__file__).resolve().parents[2].parent

ROOT = Path(os.environ.get("FUNNYBIRDS_ROOT", _PROJECTS / "Datasets" / "FunnyBirds"))
FRAMEWORK = Path(os.environ.get("FUNNYBIRDS_FRAMEWORK", _PROJECTS / "funnybirds-framework"))
CKPT_DIR = Path(os.environ.get("FUNNYBIRDS_CKPT_DIR", _PROJECTS / "Datasets" / "FunnyBirds_models"))
OUT = Path(os.environ.get("FUNNYBIRDS_OUT", "results/funnybirds"))
SEGMENT_ANYTHING = Path(os.environ.get("SEGMENT_ANYTHING", _PROJECTS / "segment-anything"))
SAM_CKPT = Path(os.environ.get("SAM_CKPT", _PROJECTS / "sam_weights" / "sam_vit_h_4b8939.pth"))

CHECKPOINTS = {
    "resnet50": "resnet50_final_0_checkpoint_best.pth.tar",
    "vgg16": "vgg16_final_1_checkpoint_best.pth.tar",
    "vit_b_16": "vit_base_patch16_224_final_1_checkpoint_best.pth.tar",
}

# Part names as the framework's protocols use them (colour names with digits stripped), and the part-map colours.
PARTS = ["beak", "eye", "foot", "tail", "wing"]
PART_COLORS = {
    "eye": [(255, 255, 253), (255, 255, 254)],
    "beak": [(255, 255, 0)],
    "foot": [(255, 0, 1), (255, 0, 2)],
    "wing": [(0, 255, 1), (0, 255, 2)],
    "tail": [(0, 0, 255)],
}

# HnS text queries, fixed before looking at any result (same "a <subject> with ..." form as the CelebA queries).
QUERY_TEXT = {
    "beak": "a bird with a beak",
    "eye": "a bird with eyes",
    "foot": "a bird with feet",
    "tail": "a bird with a tail",
    "wing": "a bird with wings",
}
