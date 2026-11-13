"""Run the FunnyBirds evaluation (all eight metrics) for one model and one explainer, using the framework's own
protocol functions unchanged (funnybirds-framework/evaluation_protocols.py), and save every per-threshold value.

Explainers:
  InputXGradient, IntegratedGradients   the framework's Captum wrappers (reference rows; IxG reproduces the
                                         framework README's ResNet-50 numbers)
  HnS                                    Hide and Seek with SigLIP pseudo-masks (hns_explainer.py)
  HnSTrueMask                            HnS with the true part masks (oracle localization; isolates the fill)
  EAC                                    Explain Any Concept (eac_explainer.py; needs segment-anything + SAM_CKPT)
  TCAV                                   local TCAV at layer3, part CAVs (tcav_explainer.py)
  PCBM, PCBM_CLIP_RN50, PCBM_SigLIP      post-hoc CBM, conventional / CLIP-RN50 / SigLIP concepts (pcbm_explainer.py)

Final row as in the framework: the threshold maximizing mean(CSDC, PC, DC) + D picks CSDC/PC/DC/D. Also reported:
  D_parts  distractibility with background objects left out of every explanation (HnS never scores background
           objects, so plain D favours it; D_parts compares all methods on bird parts only), at its own best
           threshold by the same rule.
Derived as in the paper: Com = (mean(CSDC, PC, DC) + D) / 2, Cor = SD, Con = TS, mX = mean(Com, Cor, Con).

Run from the repo root:
  python scripts/funnybirds/evaluate.py --model resnet50 --explainer HnS [--nr_itrs N] [--no-demean] [--no-orth] [--encoder ViT-B-16-SigLIP-512] [--alpha 1.5]
Writes $FUNNYBIRDS_OUT/<model>_<explainer>[_tag].json (+ _hns_log.csv or _eac_log.csv).
"""
import argparse
import base64
import csv
import json
import os
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from paths import (
    CHECKPOINTS,
    CKPT_DIR,
    FRAMEWORK,
    OUT,
    PARTS,
    QUERY_TEXT,
    ROOT,
    SAM_CKPT,
    SEGMENT_ANYTHING,
)

base64.decodestring = base64.decodebytes  # removed in Python 3.9; the framework's dataset module imports it
sys.path.insert(0, str(FRAMEWORK))  # framework packages (models, datasets, explainers) take precedence
from captum.attr import InputXGradient, IntegratedGradients
from evaluation_protocols import (
    accuracy_protocol,
    background_independence_protocol,
    controlled_synthetic_data_check_protocol,
    deletion_check_protocol,
    distractibility_protocol,
    preservation_check_protocol,
    single_deletion_protocol,
    target_sensitivity_protocol,
)
from explainers.explainer_wrapper import CaptumAttributionExplainer
from models.model_wrapper import StandardModel, ViTModel
from models.resnet import resnet50
from models.vgg import vgg16
from models.ViT.ViT_new import vit_base_patch16_224

N_CAL = 50  # calibration images for the z-score cutoffs (same draw as localization_check.py)

# CSDC calls FunnyBirds.get_minimal_sufficient_part_sets(class) once per threshold per image (80 x 500 times), and
# each call brute-forces every part subset against all 50 classes. Its result depends only on the class (classes.json
# and parts.json are fixed), so memoize it per class; returns a fresh copy each time, so values are unchanged.
import copy

from datasets.funny_birds import FunnyBirds

_msps_original = FunnyBirds.get_minimal_sufficient_part_sets
_msps_cache: dict[tuple[str, int], list] = {}


def _msps_memo(self, class_idx):
    key = (self.root_dir, int(class_idx))
    if key not in _msps_cache:
        _msps_cache[key] = _msps_original(self, class_idx)
    return copy.deepcopy(_msps_cache[key])


FunnyBirds.get_minimal_sufficient_part_sets = _msps_memo


class PartsOnly:
    """Wraps an explainer so explanations never include background objects (for D_parts)."""

    def __init__(self, inner):
        self.inner = inner

    def get_p_thresholds(self):
        return self.inner.get_p_thresholds()

    def get_important_parts(self, image, part_map, target, colors_to_part, thresholds, with_bg=False):
        return self.inner.get_important_parts(image, part_map, target, colors_to_part, thresholds=thresholds,
                                              with_bg=False)


class CachedModel(torch.nn.Module):
    """The model as the protocols see it, with outputs memoized by input content.

    PC, DC and D score the same rendered intervention image once per threshold (80 thresholds, mostly the same
    part set), so most protocol forward passes repeat. Eval-mode outputs are deterministic, so reusing them changes
    no number. Calls with gradients or batches (IxG/IG, EAC, accuracy) bypass the cache; explainers get the raw
    model.
    """

    def __init__(self, model):
        super().__init__()
        self.model, self._cache, self.hits, self.misses = model, {}, 0, 0

    def forward(self, x):
        if x.shape[0] != 1 or x.requires_grad:
            return self.model(x)
        key = zlib.crc32(x.detach().cpu().numpy().tobytes()) ^ (hash(tuple(x.shape)) & 0xFFFFFFFF)
        if key in self._cache:
            self.hits += 1
            return self._cache[key].to(x.device)
        self.misses += 1
        with torch.no_grad():
            out = self.model(x)
        self._cache[key] = out.detach()
        return out


def load_model(name, device):
    if name == "resnet50":
        model = StandardModel(resnet50(num_classes=50))
    elif name == "vgg16":
        model = StandardModel(vgg16(num_classes=50))
    elif name == "vit_b_16":
        model = ViTModel(vit_base_patch16_224(num_classes=50))
    else:
        raise ValueError(name)
    sd = torch.load(CKPT_DIR / CHECKPOINTS[name], map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd)
    return model.to(device).eval()


def build_hns(model, mask_source, demean, orth, device, encoder="ViT-B-16-SigLIP", alpha=1.0):
    from hns_explainer import HnSExplainer
    from localization_check import part_masks, sim_maps
    from omegaconf import OmegaConf
    from PIL import Image

    from cards.attribution.localization import concept_zscore_cutoff
    from cards.concepts.prompts import (
        GENERIC_REFERENCE_CONCEPTS,
        build_concept_query,
        compute_text_center,
        demean_query,
    )
    from cards.pipeline import instantiate_encoder, orthogonalize_queries

    cfg = OmegaConf.create({"seed": 0, "device": device, "encoder": {
        "name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
        "model_name": encoder, "pretrained": "webli", "device": device}})
    enc = instantiate_encoder(cfg)
    center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, enc)
    q = {p: build_concept_query(QUERY_TEXT[p], enc) for p in PARTS}
    if demean:
        q = {p: demean_query(v, center) for p, v in q.items()}
    if orth:
        q = orthogonalize_queries(q)
    q = {p: v.to(device) for p, v in q.items()}
    cut = {}
    if mask_source == "hns":
        params = json.loads((ROOT / "dataset_train.json").read_text())
        idx = list(range(len(params))); random.Random(0).shuffle(idx)
        maps = {p: [] for p in PARTS}
        for i in idx[:N_CAL]:
            c = str(params[i]["class_idx"])
            img = Image.open(ROOT / "train" / c / f"{i:06d}.png").convert("RGB")
            masks = part_masks(ROOT / "train_part_map" / c / f"{i:06d}.png")
            sm = sim_maps(enc, img, q)
            for p in PARTS:
                if masks[p].any():
                    maps[p].append(sm[p])
        cut = {p: concept_zscore_cutoff(v, alpha) for p, v in maps.items()}
        print("cutoffs:", {p: round(v, 4) for p, v in cut.items()}, flush=True)
    return HnSExplainer(model, enc, q, cut, mask_source=mask_source, device=device), cut


def best_threshold(csdc, pc, dc, d):
    """The framework's rule: maximize mean(CSDC, PC, DC) + D over thresholds."""
    return max(csdc, key=lambda t: csdc[t] / 3 + pc[t] / 3 + dc[t] / 3 + d[t])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50", choices=list(CHECKPOINTS))
    ap.add_argument("--explainer", required=True,
                    choices=["InputXGradient", "IntegratedGradients", "HnS", "HnSTrueMask", "EAC", "TCAV", "PCBM",
                             "PCBM_CLIP_RN50", "PCBM_SigLIP"])
    ap.add_argument("--nr_itrs", type=int, default=2501, help="framework default: all test images")
    ap.add_argument("--no-demean", action="store_true")
    ap.add_argument("--no-orth", action="store_true")
    ap.add_argument("--encoder", default="ViT-B-16-SigLIP", help="SigLIP checkpoint (L=196; -512 gives L=1024)")
    ap.add_argument("--alpha", type=float, default=1.0, help="z-score cutoff multiplier (paper: 1.0)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    random.seed(0); torch.manual_seed(0)
    device = f"cuda:{a.gpu}"
    args = argparse.Namespace(data=str(ROOT), gpu=a.gpu, nr_itrs=a.nr_itrs, batch_size=32, seed=0)
    model = load_model(a.model, device)

    cutoffs = {}
    if a.explainer == "InputXGradient":
        explainer = CaptumAttributionExplainer(InputXGradient(model))
    elif a.explainer == "IntegratedGradients":
        explainer = CaptumAttributionExplainer(IntegratedGradients(model),
                                               baseline=torch.zeros((1, 3, 256, 256)).to(device))
    elif a.explainer == "TCAV":
        from tcav_explainer import TCAVExplainer
        explainer = TCAVExplainer(model, device=device)
    elif a.explainer.startswith("PCBM"):
        from pcbm_explainer import PCBMExplainer
        variant = {"PCBM": "conventional", "PCBM_CLIP_RN50": "clip_rn50", "PCBM_SigLIP": "siglip"}[a.explainer]
        explainer = PCBMExplainer(model, variant, device=device)
    elif a.explainer == "EAC":
        sys.path.insert(0, str(SEGMENT_ANYTHING))
        from eac_explainer import EACExplainer
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        sam = sam_model_registry["default"](checkpoint=str(SAM_CKPT)).to(device)
        explainer = EACExplainer(model, SamAutomaticMaskGenerator(sam), device=device)
    else:
        explainer, cutoffs = build_hns(model, "hns" if a.explainer == "HnS" else "true",
                                       not a.no_demean, not a.no_orth, device, a.encoder, a.alpha)

    pm = CachedModel(model)  # protocols' model; explainers above hold the raw model
    steps = [
        ("accuracy", lambda: accuracy_protocol(model, args)),
        ("csdc", lambda: controlled_synthetic_data_check_protocol(pm, explainer, args)),
        ("ts", lambda: target_sensitivity_protocol(pm, explainer, args)),
        ("sd", lambda: single_deletion_protocol(pm, explainer, args)),
        ("pc", lambda: preservation_check_protocol(pm, explainer, args)),
        ("dc", lambda: deletion_check_protocol(pm, explainer, args)),
        ("d", lambda: distractibility_protocol(pm, explainer, args)),
        ("d_parts", lambda: distractibility_protocol(pm, PartsOnly(explainer), args)),
        ("bi", lambda: background_independence_protocol(pm, args)),
    ]
    t0 = time.time(); res = {}
    for i, (name, fn) in enumerate(steps):
        print(f"[{time.strftime('%H:%M:%S')}] step {i + 1}/{len(steps)}: {name} ...", flush=True)
        res[name] = fn()
        print(f"[{time.strftime('%H:%M:%S')}] step {i + 1}/{len(steps)}: {name} done "
              f"({(time.time() - t0) / 60:.1f} min total; model cache {pm.hits} hits / {pm.misses} misses)",
              flush=True)

    t = best_threshold(res["csdc"], res["pc"], res["dc"], res["d"])
    tp = best_threshold(res["csdc"], res["pc"], res["dc"], res["d_parts"])
    final = {"accuracy": res["accuracy"], "CSDC": res["csdc"][t], "PC": res["pc"][t], "DC": res["dc"][t],
             "D": res["d"][t], "BI": res["bi"], "SD": res["sd"], "TS": res["ts"], "threshold": float(t)}
    final["Com"] = (np.mean([final["CSDC"], final["PC"], final["DC"]]) + final["D"]) / 2
    final["Cor"], final["Con"] = final["SD"], final["TS"]
    final["mX"] = np.mean([final["Com"], final["Cor"], final["Con"]])
    final |= {"D_parts": res["d_parts"][tp], "threshold_parts": float(tp)}
    final["Com_parts"] = (np.mean([res["csdc"][tp], res["pc"][tp], res["dc"][tp]]) + final["D_parts"]) / 2
    final["mX_parts"] = np.mean([final["Com_parts"], final["Cor"], final["Con"]])
    final = {k: float(v) for k, v in final.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{a.model}_{a.explainer}" + (f"_{a.tag}" if a.tag else "")
    per_t = {k: {str(float(t)): float(v) for t, v in res[k].items()} for k in ("csdc", "pc", "dc", "d", "d_parts")}
    meta = {"model": a.model, "explainer": a.explainer, "nr_itrs": a.nr_itrs, "demean": not a.no_demean,
            "orthogonalize": not a.no_orth, "encoder": a.encoder, "alpha": a.alpha, "cutoffs": cutoffs, "minutes": (time.time() - t0) / 60,
            "framework": str(FRAMEWORK), "data": str(ROOT), "host": os.environ.get("COMPUTERNAME", "")}
    if hasattr(explainer, "fidelity"):  # PCBM surrogate: held-out training-image agreement with the black box
        meta |= {"pcbm_fidelity": explainer.fidelity, "pcbm_lam": explainer.lam}
    (OUT / f"{stem}.json").write_text(json.dumps({"final": final, "per_threshold": per_t, "meta": meta}, indent=1))
    if hasattr(explainer, "log") and explainer.log:
        with open(OUT / f"{stem}_{'eac' if a.explainer == 'EAC' else 'hns'}_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, list(explainer.log[0])); w.writeheader(); w.writerows(explainer.log)
    print("\nFINAL", stem)
    print("  ".join(f"{k} {v:.4f}" for k, v in final.items()))


if __name__ == "__main__":
    main()
