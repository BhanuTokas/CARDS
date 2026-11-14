"""Run one of the COCO-Stuff baselines (scripts/coco_stuff/baselines/) on one shortcut checkpoint.

usage: run_baseline.py RATE_PCT METHOD [SAM_CKPT]
  RATE_PCT in {0, 10, ..., 100}; METHOD in {labels, feats, sae, shap, eac}.

The shortcut classifiers (build/train_coco_stuff_shortcut_classifiers.py) are ResNet18s with the same 20-way BCE head as
the main COCO-Stuff black box: resnet18_coco_stuff_shortcut_{rate}.pt in $COCO_SHORTCUT_CKPT_DIR
[trained_models_new/coco_stuff/shortcut]. This wrapper replaces coco_common.load_black_box by a loader of that checkpoint
(same preprocess as the main model) and runs the baseline script unchanged with runpy; nothing in baselines/ is edited.

Every method sees CLEAN images (no shortcut cue), as the existing COCO shortcut runs (CARDS, TCAV, PCBM) do.
  labels: builds the shared, model-independent concept_labels.npz once (run it before the array, so the 11 tasks do
          not write it concurrently).
Outputs in $SHORTCUT_OUT/rate_{rate}/ (default results/coco_stuff/shortcut_baselines): sae.csv, shap.csv, eac.csv in the
scripts' own formats; features in $SHORTCUT_FEAT_ROOT/feat_{rate} (default next to them; large, safe to delete after sae
and shap)."""
import os
import runpy
import sys
from pathlib import Path

RATE, METHOD = int(sys.argv[1]), sys.argv[2]
assert RATE in range(0, 101, 10) and METHOD in ("labels", "feats", "sae", "shap", "eac")
OUT = Path(os.environ.get("SHORTCUT_OUT", "results/coco_stuff/shortcut_baselines")) / f"rate_{RATE}"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = Path(os.environ.get("COCO_SHORTCUT_CKPT_DIR", "trained_models_new/coco_stuff/shortcut")) / \
    f"resnet18_coco_stuff_shortcut_{RATE}.pt"
os.environ["CARDS_ARCH"] = "resnet18"
os.environ["CARDS_FEAT_CACHE"] = str(Path(os.environ.get("SHORTCUT_FEAT_ROOT", str(OUT.parent))) / f"feat_{RATE}")

# imports below must come after the env settings above (coco_common reads them on import)
import torch
from torch import nn
from torchvision.models import resnet18

B = Path("scripts/coco_stuff/baselines")
sys.path.insert(0, str(B))
import coco_common


def load_shortcut():
    """(shortcut ResNet18, the main model's preprocess): the same architecture and head as BACKBONES['coco_stuff']."""
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(coco_common.CLASSES))
    m.load_state_dict(torch.load(CKPT, map_location="cpu"))
    return m.to(coco_common.DEVICE).eval(), coco_common.BACKBONES[coco_common.BACKBONE["resnet18"]].preprocess


if METHOD == "labels":
    y = coco_common.concept_labels(coco_common.pool_paths(), coco_common.concept_list())
    print(f"concept labels {y.shape} in {coco_common.OUT_DIR / 'concept_labels.npz'}")
    sys.exit(0)

def stub_groundingdino():
    """eac_coco.main() imports GroundingDINO unconditionally, but with GT_MASKS_DIR set it uses it only for images
    without a saved ground-truth mask (none expected: EAC's images are concept-positives of the same faithfulness CSV the
    masks were saved for). GroundingDINO is not in the uv env, so register placeholders: the imports and the transform
    setup succeed, and actually needing GroundingDINO raises instead of silently differing from the ground truth."""
    import types

    def needed(*_a, **_k):
        raise RuntimeError("an EAC image has no saved mask in GT_MASKS_DIR, so eac_coco.py needs GroundingDINO, which "
                           "is not installed in this environment; check GT_MASKS_DIR")

    mods = {n: types.ModuleType(n) for n in ("groundingdino", "groundingdino.util", "groundingdino.util.box_ops",
                                              "groundingdino.util.inference", "groundingdino.datasets",
                                              "groundingdino.datasets.transforms")}
    mods["groundingdino.util.inference"].load_model = mods["groundingdino.util.inference"].predict = needed
    for name in ("Compose", "RandomResize", "ToTensor", "Normalize"):  # built once at startup, used only with GroundingDINO
        setattr(mods["groundingdino.datasets.transforms"], name, lambda *_a, **_k: None)
    for n, m in mods.items():
        if "." in n:
            parent, child = n.rsplit(".", 1)
            setattr(mods[parent], child, m)
    sys.modules.update(mods)


assert CKPT.exists(), f"missing {CKPT}"
coco_common.load_black_box = load_shortcut
if METHOD == "eac":
    assert os.environ.get("GT_MASKS_DIR"), "EAC needs GT_MASKS_DIR (GroundingDINO is replaced by placeholders)"
    stub_groundingdino()
script, args = {"feats": ("extract_feats_coco.py", []), "sae": ("concept_sae_coco.py", [OUT / "sae.csv"]),
                "shap": ("concept_shap_coco.py", [OUT / "shap.csv"]),
                "eac": ("eac_coco.py", [*sys.argv[3:4], OUT / "eac.csv"])}[METHOD]
print(f"rate {RATE}%: {script} on {CKPT} -> {OUT}  (features in {os.environ['CARDS_FEAT_CACHE']})", flush=True)
sys.argv = [str(B / script), *map(str, args)]
runpy.run_path(str(B / script), run_name="__main__")
