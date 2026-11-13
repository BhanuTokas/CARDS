"""Local (per-image) TCAV on the COCO-Stuff local records (common.py), ResNet18, at layer3 and layer4.

layer3 is the score that is compared: after layer4 ResNet18 has only average pooling and the linear fc, so the class
logit's gradient at layer4 is the same for every image and the layer4 local score is one constant per
(concept, class). It is computed as well (tcav4_dd) to show that.

CAVs (Kim et al. 2018): per concept and layer, a linear classifier (sklearn SGDClassifier, alpha=0.01 as captum's
default, random_state=0) separating the layer's activations of 40 Broden positives of the concept (the concept
examples of the existing COCO TCAV runs) from each of 6 random sets of 25 class-present train2017 images; CAV = its
unit-norm weight vector. Per image and class: directional derivatives d_k = grad_layer(class logit) . CAV_k, k = 1..6;
  tcav_sign = mean_k [d_k > 0] - 0.5   (standard local TCAV score, centred so "no effect" is 0)
  tcav_dd   = mean_k d_k              (continuous directional derivative)

argv: none. Writes $COCO_LOCAL_OUT/tcav.csv (every record) and tcav_cavs.pt."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, str(Path(__file__).parent))
import common as C

from cards.data.coco_stuff import COCO_STUFF_20_CLASSES
from cards.data.datasets import load_broden, load_coco_stuff
from cards.models.backbones import BACKBONES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = ["layer3", "layer4"]
N_RANDOM, N_PER_RANDOM, N_POS = 6, 25, 40
STAGES = ["layer1", "layer2", "layer3", "layer4"]


def split(m, layer):
    """(pre, post): pre(x) = activations at layer, post(a) = logits."""
    k = STAGES.index(layer)

    def pre(x):
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        for s in STAGES[:k + 1]:
            x = getattr(m, s)(x)
        return x

    def post(a):
        for s in STAGES[k + 1:]:
            a = getattr(m, s)(a)
        return m.fc(torch.flatten(m.avgpool(a), 1))
    return pre, post


@torch.no_grad()
def acts(pre, prep, paths):
    out = []
    for s in range(0, len(paths), 64):
        x = torch.stack([prep(Image.open(p).convert("RGB")) for p in paths[s:s + 64]]).to(DEVICE)
        out.append(pre(x).flatten(1).cpu())
    return torch.cat(out).numpy()


def fit_cavs(model, prep, concepts):
    pool = sorted(str(p) for p, lbl in load_coco_stuff(C.COCO_ROOT, COCO_STUFF_20_CLASSES, split="train") if lbl >= 0)
    rnd = np.random.default_rng(C.SEED).choice(pool, N_RANDOM * N_PER_RANDOM, replace=False).tolist()
    randoms = [rnd[i * N_PER_RANDOM:(i + 1) * N_PER_RANDOM] for i in range(N_RANDOM)]
    pos = {}
    for i, c in enumerate(concepts):
        p = sorted(str(x) for x in load_broden(C.BRODEN_ROOT, c)[0])
        np.random.default_rng(C.SEED + 1 + i).shuffle(p); pos[c] = p[:N_POS]
    cavs = {}
    for layer in LAYERS:
        pre, _ = split(model, layer)
        ra = [acts(pre, prep, r) for r in randoms]; cavs[layer] = {}
        for c in concepts:
            ca = acts(pre, prep, pos[c]); v = []
            for r in ra:
                X = np.concatenate([ca, r]); y = np.r_[np.ones(len(ca)), np.zeros(len(r))]
                w = SGDClassifier(alpha=0.01, max_iter=1000, tol=1e-3, random_state=0).fit(X, y).coef_[0]
                v.append(w / np.linalg.norm(w))
            cavs[layer][c] = torch.tensor(np.stack(v), dtype=torch.float32)
            print(f"  {layer} CAVs {c} ({len(pos[c])} positives)", flush=True)
    return cavs


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    spec = BACKBONES["coco_stuff"]; model = spec.load_native().to(DEVICE).eval(); prep = spec.preprocess
    recs = C.records(); concepts = sorted({r["concept_name"] for r in recs})
    cav_path = C.OUT / "tcav_cavs.pt"
    cavs = torch.load(cav_path) if cav_path.exists() else fit_cavs(model, prep, concepts)
    torch.save(cavs, cav_path)
    cavs = {L: {c: v.to(DEVICE) for c, v in d.items()} for L, d in cavs.items()}
    cls_idx = {k: i for i, k in enumerate(COCO_STUFF_20_CLASSES)}
    by_image = C.groups(recs, by=("image",))
    with open(C.OUT / "tcav.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow([*C.BASE_COLS, "tcav_sign", "tcav_dd", "tcav4_sign", "tcav4_dd"])
        for b in range(0, len(by_image), 32):
            chunk = by_image[b:b + 32]
            x = torch.stack([prep(Image.open(C.image_path(n)).convert("RGB")) for (n,), _ in chunk]).to(DEVICE)
            classes = sorted({r["class_name"] for _, rs in chunk for r in rs})
            res = {}
            for layer in LAYERS:
                pre, post = split(model, layer)
                a = pre(x).detach().requires_grad_(True); logits = post(a)
                g = {k: torch.autograd.grad(logits[:, cls_idx[k]].sum(), a, retain_graph=True)[0].flatten(1)
                     for k in classes}  # per-image gradients (images are independent in eval mode)
                for i, (_, rs) in enumerate(chunk):
                    for r in rs:
                        d = cavs[layer][r["concept_name"]] @ g[r["class_name"]][i]
                        res.setdefault(id(r), []).extend([float((d > 0).float().mean() - 0.5), float(d.mean())])
            for _, rs in chunk:
                for r in rs:
                    w.writerow([*C.base(r), *res[id(r)]])
            if (b // 32) % 20 == 0:
                print(f"  {b + len(chunk)}/{len(by_image)} images", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
