"""Run one of the CelebA baselines (scripts/celeba/baselines/) on one official-train shortcut checkpoint.

usage: run_baseline.py TASK RATE_PCT METHOD [SAM_CKPT]
  TASK in {Attractive, Male}; RATE_PCT in {0, 10, ..., 100}; METHOD in {feats, sae, shap, eac}.

The shortcut classifiers (train_{attractive,male}_shortcut_classifiers_official_train.py) are single-task ResNet18s with
a 2-way head: resnet18_official_train_{task}_shortcut_{rate}.pt in $CARDS_CKPT_DIR. The baseline scripts expect the
4-way Attractive/Male head of black_boxes.py ([0:2] Attractive, [2:4] Male). This wrapper gives them the shortcut model
with its 2-way head copied into BOTH task slots, so every script runs unchanged, and keeps only the rows of TASK. (A
copy rather than an empty slot: an empty slot would make ConceptSHAP's completeness divide by zero.) The scripts are
not edited; they are run with runpy after black_boxes.load is replaced.

Every method sees CLEAN images (no shortcut cue), as the existing shortcut runs (CARDS, TCAV, PCBM) do.
Outputs in $SHORTCUT_OUT/{task}_{rate}/ (default results/shortcut_baselines): sae.csv, shap.csv, eac.csv in the
scripts' own formats; features in $SHORTCUT_FEAT_ROOT/{task}_{rate} (default the same folder; large, safe to delete
after sae and shap)."""
import os
import runpy
import sys
from pathlib import Path

TASK, RATE, METHOD = sys.argv[1], int(sys.argv[2]), sys.argv[3]
assert TASK in ("Attractive", "Male") and RATE in range(0, 101, 10) and METHOD in ("feats", "sae", "shap", "eac")
OUT = Path(os.environ.get("SHORTCUT_OUT", "results/shortcut_baselines")) / f"{TASK.lower()}_{RATE}"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["CARDS_ARCH"] = "resnet18"
os.environ["CARDS_FEAT_CACHE"] = str(Path(os.environ.get("SHORTCUT_FEAT_ROOT", str(OUT.parent))) / f"feat_{TASK.lower()}_{RATE}")

# imports below must come after the env settings above (black_boxes reads them on import)
import torch
from torch import nn
from torchvision.models import resnet18

B = Path("scripts/celeba/baselines")
sys.path.insert(0, str(B))
import black_boxes


def load_shortcut(arch: str = "resnet18") -> nn.Module:
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(black_boxes.CKPT_DIR / f"resnet18_official_train_{TASK.lower()}_shortcut_{RATE}.pt",
                                 map_location="cpu"))
    fc4 = nn.Linear(m.fc.in_features, 4)
    with torch.no_grad():
        fc4.weight.copy_(m.fc.weight.repeat(2, 1)); fc4.bias.copy_(m.fc.bias.repeat(2))
    m.fc = fc4
    return m.eval()


black_boxes.load = load_shortcut
script, args = {"feats": ("extract_feats.py", []), "sae": ("concept_sae.py", [OUT / "sae.csv"]),
                "shap": ("concept_shap.py", [OUT / "shap.csv"]),
                "eac": ("celeba_eac.py", [sys.argv[4] if len(sys.argv) > 4 else "", OUT / "eac.csv"])}[METHOD]
print(f"{TASK} rate {RATE}%: {script} -> {OUT}  (features in {os.environ['CARDS_FEAT_CACHE']})", flush=True)
sys.argv = [str(B / script), *map(str, args)]
runpy.run_path(str(B / script), run_name="__main__")
