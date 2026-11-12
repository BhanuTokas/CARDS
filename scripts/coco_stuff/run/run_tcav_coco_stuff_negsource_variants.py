"""COCO-Stuff TCAV with alternative CAV example sources (103 balanced concepts, any black box), CelebA-style sign counts.

Same CAV fit (LinearSVC C=0.1 on pooled hook-layer activations) and sizes (N_CAV_POS positives, N_CAV_NEG negatives) as
run_tcav_coco_stuff_model_variants.py; only the CAV examples change. Sign counts differ from that script, following CelebA:
the sign count of class k is computed on 40 random val2017 images that CONTAIN class k (per-class sets, 20 x 40 images),
not on one shared random set. Gradients depend only on class and image, so they are computed once and reused per concept.

  VARIANT=random   positives = Broden positive exemplars, negatives = random train2017 images (the original TCAV recipe)
  VARIANT=broden   positives = Broden positive exemplars, negatives = Broden negative exemplars of that concept
  VARIANT=coco     positives = train2017 images where Grounding DINO detected the concept (presence),
                   negatives = train2017 images where it did not (absence), both from the 54,766-image pool
                   (detection is model-independent, so results/coco_stuff/coco_stuff_faithfulness.csv serves all models)

The original run (Broden positives, random train2017 negatives) is untouched. Positives of the coco variant can overlap
the ground-truth images of the concept; the CAV never sees delta_p. `LIMIT` runs the first n concepts (smoke test).

Output: results/coco_stuff/model_variants/{backbone}_tcav_{variant}_perclass_scores.csv (concept_name, class_name, mean_sign_count)
Score with compare_variants_own_gt.py via EXTRA="backbone:TCAV-Broden|path|mean_sign_count;..."
Usage (repo root): BACKBONE_NAME=coco_stuff_vit_b_16 VARIANT=coco python scripts/coco_stuff/run/run_tcav_coco_stuff_negsource_variants.py"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from PIL import Image

import run_tcav_coco_stuff_model_variants as base  # noqa: E402
from cards.data.coco_stuff import COCO_STUFF_20_CLASSES  # noqa: E402
from cards.data.datasets import list_broden_concepts, load_broden, load_coco_stuff  # noqa: E402
from cards.models.backbones import BACKBONES  # noqa: E402


def _val_multilabel(coco_root: Path, classes: list[str]) -> dict[str, np.ndarray]:
    """val2017 image path -> binary vector over the 20 classes (images with >=1 class)."""
    import json
    from collections import defaultdict
    data = json.load(open(coco_root / "annotations" / "instances_val2017.json"))
    cid = {c["name"]: c["id"] for c in data["categories"]}
    idx = {cid[c]: i for i, c in enumerate(classes) if c in cid}
    per: dict[int, set] = defaultdict(set)
    for a in data["annotations"]:
        if a["category_id"] in idx:
            per[a["image_id"]].add(idx[a["category_id"]])
    fn = {im["id"]: im["file_name"] for im in data["images"]}
    out = {}
    for i, ks in per.items():
        v = np.zeros(len(classes), dtype=np.int8); v[list(ks)] = 1
        out[str(coco_root / "val2017" / fn[i])] = v
    return out


def _class_grads(paths: list[str], k: int, model, hook_layer, backbone: str, preprocess) -> torch.Tensor:
    """Pooled gradient [n, D] of logit k w.r.t. the hooked layer on the given images (concept-independent)."""
    x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(base.DEVICE)
    store: list[torch.Tensor] = []

    def hook(m, i, o):
        o.retain_grad(); store.append(o)

    h = hook_layer.register_forward_hook(hook)
    with torch.enable_grad():
        logits = model(x)
    h.remove()
    logits[:, k].sum().backward()
    g = base._pool(store[0].grad, backbone).detach()
    model.zero_grad(set_to_none=True)
    return g


def main() -> None:
    backbone = os.environ.get("BACKBONE_NAME", "coco_stuff")
    variant = os.environ.get("VARIANT", "")
    if variant not in ("random", "broden", "coco"):
        raise SystemExit("set VARIANT=random, broden or coco")
    limit = int(os.environ.get("LIMIT", "0")) or None
    spec = BACKBONES[backbone]
    base.OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = spec.load_native().to(base.DEVICE).eval()
    hook_layer = model.get_submodule(spec.hook_layer)

    balanced = base._load_balanced_concepts(base.COUNTS_CSV, base.MIN_FAITH_IMGS)
    concepts = [c for c in list_broden_concepts(base.BRODEN_ROOT)
                if c not in base.BRODEN_DROPPED_CONCEPTS and c in balanced]
    print(f"{backbone} / {variant}: {len(concepts)} concepts", flush=True)
    if limit:
        concepts = concepts[:limit]

    # Per-class sign-count images: 40 random val2017 images containing class k; gradients cached once per class.
    train_all = [str(p) for p, lbl in load_coco_stuff(base.COCO_ROOT, COCO_STUFF_20_CLASSES, split="train")]
    val_labels = _val_multilabel(base.COCO_ROOT, COCO_STUFF_20_CLASSES)
    grads: list[torch.Tensor] = []
    for k, cls in enumerate(COCO_STUFF_20_CLASSES):
        cands = sorted(p for p, v in val_labels.items() if v[k])
        rk = np.random.default_rng(base.SEED + 1000 + k)
        pick = [cands[i] for i in rk.permutation(len(cands))[:base.N_VAL_SAMPLES]]
        grads.append(_class_grads(pick, k, model, hook_layer, backbone, spec.preprocess))
        print(f"  class {cls}: {len(pick)} of {len(cands)} val images containing it", flush=True)

    detected: dict[str, set[str]] = {}
    pool: list[str] = []
    if variant == "coco":
        pool = sorted(str(p) for p, lbl in load_coco_stuff(base.COCO_ROOT, COCO_STUFF_20_CLASSES, split="train")
                      if lbl >= 0)  # same pool the ground truth was built on
        pairs = pd.read_csv(base.FAITH_CSV, usecols=["concept_name", "image_path"]).drop_duplicates()
        for c, g in pairs.groupby("concept_name"):
            detected[c] = set(g["image_path"])
        print(f"pool {len(pool)} images; detections for {len(detected)} concepts", flush=True)
        missing = sum(1 for p in list(detected.get("sky", []))[:50] if p not in set(pool))
        if missing:
            raise SystemExit("detection image paths do not match pool paths; check path format")

    rows = []
    for ci, concept in enumerate(concepts):
        r = np.random.default_rng(base.SEED + ci)
        if variant == "broden":
            pos_all, neg_all = load_broden(base.BRODEN_ROOT, concept)
            pos_all, neg_all = sorted(map(str, pos_all)), sorted(map(str, neg_all))
        elif variant == "random":
            pos_all, _ = load_broden(base.BRODEN_ROOT, concept)
            pos_all = sorted(map(str, pos_all))
            neg_all = train_all  # random train2017 images, as in the original run (not filtered)
        else:
            pos_all = sorted(detected.get(concept, []))
            neg_all = [p for p in pool if p not in detected.get(concept, set())]
        pos = [pos_all[i] for i in r.permutation(len(pos_all))[:base.N_CAV_POS]]
        neg = [neg_all[i] for i in r.permutation(len(neg_all))[:base.N_CAV_NEG]]
        cav = base._fit_cav([Image.open(p).convert("RGB") for p in pos], [Image.open(p).convert("RGB") for p in neg],
                            model, hook_layer, backbone, spec.preprocess, base.DEVICE)
        if cav is None:
            sc = np.full(len(COCO_STUFF_20_CLASSES), 0.5, dtype=np.float32); status = "skip"
        else:
            cav_t = torch.tensor(cav, dtype=torch.float32, device=base.DEVICE)
            sc = np.array([float((torch.sign(g @ cav_t) > 0).float().mean()) for g in grads], dtype=np.float32)
            status = "ok"
        rows += [(concept, cls, float(sc[k])) for k, cls in enumerate(COCO_STUFF_20_CLASSES)]
        if ci % 10 == 0:
            print(f"  [{ci + 1}/{len(concepts)}] {concept:<22s} pos={len(pos)} neg={len(neg)} {status} "
                  f"mean_sign={float(sc.mean()):.3f}", flush=True)

    out = base.OUT_DIR / f"{backbone}_tcav_{variant}_perclass_scores{'_smoke' if limit else ''}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["concept_name", "class_name", "mean_sign_count"]); w.writerows(rows)
    print(f"Saved {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
