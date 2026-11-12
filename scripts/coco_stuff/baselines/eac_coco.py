# ruff: noqa: SIM115
"""Explain Any Concept (EAC, Sun et al. 2023) on COCO-Stuff ResNet18; port of scripts/celeba/baselines/celeba_eac.py.
Run from the repo root with Meta's segment-anything and GroundingDINO importable and weights/ populated (the same files
scripts/coco_stuff/build/compute_coco_stuff_faithfulness.py uses). argv[1] = SAM checkpoint, argv[2] = output CSV.
env EAC_LIMIT=n images (smoke test), N_PER_CONCEPT (default 20), MC (default 50000).

EAC procedure identical to CelebA: SAM ViT-H automatic segments at the model's 224 px input, black fill in raw pixel
space, PIE surrogate = Linear(n_segments, 512) + the model's frozen fc, BCE on the class probability, SGD lr .008 /
momentum .9 / batch 100 / 50 epochs, 2500 Bernoulli(.5) subsets, MC Shapley coalitions. One surrogate per class (20),
targets P(class) = sigmoid(logit).

Concept regions: CelebA reads face-part masks from CelebAMask-HQ; COCO has none, so a concept's region in an image is
the SAME Grounding DINO + SAM union mask the ground truth masked (all NMS'd instances of the concept, box thresholds
0.30/0.30), recomputed here for the chosen images. A segment adds its value to a concept in proportion to the share of
its pixels inside that region. Score (concept, class) = mean over the chosen images where the concept was detected.
Images: N_PER_CONCEPT concept-positive pool images per concept (positives = Grounding DINO detections, as in the GT).
env GT_MASKS_DIR=<dir> reads the masks saved by compute_coco_stuff_faithfulness_variants.py (SAVE_MASKS) instead of
rerunning Grounding DINO; images without a saved file fall back to the rerun. The masks are model-independent, so the same
directory serves every black box.
Resumable: per-image values go to OUT.csv.images.csv. SAM segments cached in $CARDS_SAM_CACHE when set."""
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, "scripts/coco_stuff/build")
from coco_common import CLASSES, DEVICE, FEAT_CACHE, concept_list, load_black_box
from compute_coco_stuff_faithfulness import (  # noqa: E402
    BOX_THRESHOLD, GDINO_CKPT, GDINO_CONFIG, NMS_IOU, SAM_CKPT, TEXT_THRESHOLD, _match_phrase,
)

N_PER_CONCEPT = int(os.environ.get("N_PER_CONCEPT", "20")); MC = int(os.environ.get("MC", "50000"))
N_SAMPLES, SEED, EPOCHS, BATCH, LR, MOMENTUM = 2500, 0, 50, 100, 0.008, 0.9
NC = len(CLASSES)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SAM_CACHE = Path(os.environ["CARDS_SAM_CACHE"]) if os.environ.get("CARDS_SAM_CACHE") else None


def sam_masks(gen, key, raw):
    f = SAM_CACHE / f"{key}.npz" if SAM_CACHE else None
    if f is not None and f.exists():
        z = np.load(f); return np.unpackbits(z["m"], axis=-1, count=int(z["w"])).astype(bool)
    m = np.stack([x["segmentation"] for x in gen.generate(raw)])
    if f is not None:
        SAM_CACHE.mkdir(parents=True, exist_ok=True); np.savez_compressed(f, m=np.packbits(m, axis=-1), w=m.shape[-1])
    return m


GT_MASKS = Path(os.environ["GT_MASKS_DIR"]) if os.environ.get("GT_MASKS_DIR") else None


def saved_regions(image_path):
    """concept -> union mask saved by compute_coco_stuff_faithfulness_variants.py (SAVE_MASKS); None if absent."""
    f = GT_MASKS / f"{Path(image_path).stem}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    m = np.unpackbits(z["masks"], axis=-1, count=int(z["w"])).astype(bool)
    return dict(zip(z["concepts"].tolist(), m))


def concept_regions(img, gdino, predictor, concepts, caption, gtf, box_ops, predict):
    """concept -> union SAM mask (H,W) of all NMS'd GDINO instances; identical to the ground-truth pipeline."""
    from torchvision.ops import nms
    arr = np.array(img); h, w = arr.shape[:2]
    t, _ = gtf(img, None)
    boxes, logits, phrases = predict(model=gdino, image=t.to(DEVICE), caption=caption, box_threshold=BOX_THRESHOLD,
                                     text_threshold=TEXT_THRESHOLD, device=DEVICE)
    if len(boxes) == 0:
        return {}
    babs = box_ops.box_cxcywh_to_xyxy(boxes.cpu()) * torch.tensor([w, h, w, h], dtype=torch.float32)
    raw = defaultdict(list)
    for bx, lg, ph in zip(babs, logits.tolist(), phrases):
        c = _match_phrase(ph, concepts)
        if c is not None: raw[c].append((bx, float(lg)))
    predictor.set_image(arr); out = {}
    for c, pairs in raw.items():
        bt = torch.stack([b for b, _ in pairs]); keep = nms(bt, torch.tensor([s for _, s in pairs]), NMS_IOU)
        u = np.zeros((h, w), bool)
        for i in keep:
            u |= predictor.predict(box=bt[i].numpy(), multimask_output=False)[0][0]
        if u.any() and not u.all(): out[c] = u
    return out


def fit_values(subsets, probs, w, b):
    """Segment values (NC, n_segments): one PIE surrogate per class."""
    n_seg, d = subsets.shape[1], w.shape[1]
    torch.manual_seed(0)
    layer = torch.nn.Linear(n_seg, d)
    weight = layer.weight.detach().T[None].repeat(NC, 1, 1).to(DEVICE).requires_grad_()
    bias = layer.bias.detach()[None].repeat(NC, 1).to(DEVICE).requires_grad_()
    opt = torch.optim.SGD([weight, bias], lr=LR, momentum=MOMENTUM)
    x = torch.from_numpy(subsets.astype(np.float32)).to(DEVICE); y = probs.to(DEVICE)

    def surrogate(bt):
        h = torch.einsum("bn,tnd->tbd", bt, weight) + bias[:, None]
        return torch.sigmoid(torch.einsum("tbd,td->bt", h, w) + b)  # (B, NC)

    gen = torch.Generator().manual_seed(0)
    for _ in range(EPOCHS):
        order = torch.randperm(len(x), generator=gen)
        for s in range(0, len(x), BATCH):
            idx = order[s:s + BATCH].to(DEVICE)
            pred = surrogate(x[idx]).clamp(1e-6, 1 - 1e-6)
            loss = torch.nn.functional.binary_cross_entropy(pred, y[idx], reduction="sum") / len(idx)
            opt.zero_grad(); loss.backward(); opt.step()
    values = torch.zeros(NC, n_seg, device=DEVICE); rng = torch.Generator(device=DEVICE).manual_seed(0)
    with torch.no_grad():
        for i in range(n_seg):
            tot = torch.zeros(NC, device=DEVICE)
            for s in range(0, MC, 5000):
                c = (torch.rand(5000, n_seg, generator=rng, device=DEVICE) < 0.5).float()
                c[:, i] = 1; a = surrogate(c); c[:, i] = 0
                tot += (a - surrogate(c)).sum(0)
            values[:, i] = tot / MC
    return values.cpu().numpy()


@torch.no_grad()
def model_probs(model, raw, masks, subsets):
    img = torch.from_numpy(raw).permute(2, 0, 1)[None].float().to(DEVICE) / 255
    flat = torch.from_numpy(masks.reshape(len(masks), -1)).float().to(DEVICE); out = []
    for s in range(0, len(subsets), 250):
        keep = (torch.from_numpy(subsets[s:s + 250].astype(np.float32)).to(DEVICE) @ flat > 0).float().view(-1, 1, *raw.shape[:2])
        out.append(torch.sigmoid(model((img * keep - MEAN.to(DEVICE)) / STD.to(DEVICE))))
    return torch.cat(out).cpu()


def main():
    sam_ckpt, out_path = sys.argv[1] if len(sys.argv) > 2 else str(SAM_CKPT), sys.argv[-1]
    limit = int(os.environ.get("EAC_LIMIT", "0"))
    from groundingdino.util import box_ops
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T
    from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

    d = torch.load(FEAT_CACHE / "pooled.pt", weights_only=False)
    paths, concepts, Y = d["paths"], list(d["concepts"]), d["labels"].numpy()
    rng = np.random.default_rng(SEED); chosen = set()
    for n in range(len(concepts)):
        pos = np.where(Y[:, n])[0]; chosen.update(rng.choice(pos, min(N_PER_CONCEPT, len(pos)), replace=False).tolist())
    chosen = sorted(chosen)[:limit] if limit else sorted(chosen)
    print(f"{len(chosen)} images", flush=True)
    model, _ = load_black_box(); w, b = model.fc.weight.detach(), model.fc.bias.detach()
    gtf = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(DEVICE)
    sgen, spred = SamAutomaticMaskGenerator(sam), SamPredictor(sam)
    gdino = None  # loaded only if some image has no saved mask file
    caption = " . ".join(concepts)
    per = defaultdict(list); start = time.time()
    img_path = Path(out_path + ".images.csv"); done = set()
    if img_path.exists():
        for r in csv.DictReader(open(img_path)):
            done.add(int(r["img"]))
            if r["concept_name"]: per[r["concept_name"], r["class_name"]].append(float(r["value"]))
    new = not img_path.exists(); img_f = open(img_path, "a", newline=""); img_w = csv.writer(img_f)
    if new: img_w.writerow(["img", "concept_name", "class_name", "value"])
    print(f"resuming with {len(done)} images done", flush=True)
    for n, i in enumerate(chosen):
        if i in done: continue
        pil = Image.open(paths[i]).convert("RGB")
        regions = saved_regions(paths[i]) if GT_MASKS else None
        if regions is None:
            if GT_MASKS: print(f"  no saved masks for {paths[i]}; rerunning Grounding DINO", flush=True)
            if gdino is None: gdino = load_model(str(GDINO_CONFIG), str(GDINO_CKPT)).to(DEVICE).eval()
            regions = concept_regions(pil, gdino, spred, concepts, caption, gtf, box_ops, predict)
        regions = {c: r for c, r in regions.items() if c in concepts}
        if not regions:
            img_w.writerow([i, "", "", 0.0]); img_f.flush(); continue
        raw = np.asarray(pil.resize((224, 224), Image.Resampling.BILINEAR))
        masks = sam_masks(sgen, Path(paths[i]).stem, raw)
        subsets = np.random.default_rng(SEED + i).random((N_SAMPLES, len(masks))) < 0.5
        values = fit_values(subsets, model_probs(model, raw, masks, subsets), w, b)  # (NC, n)
        area = masks.sum((1, 2)).clip(min=1)
        for c, reg in regions.items():
            r224 = np.asarray(Image.fromarray(reg).resize((224, 224), Image.Resampling.NEAREST))
            frac = (masks & r224).sum((1, 2)) / area
            for k, cls in enumerate(CLASSES):
                v = float(values[k] @ frac); per[c, cls].append(v); img_w.writerow([i, c, cls, v])
        img_f.flush()
        if (n + 1) % 20 == 0: print(f"{n + 1}/{len(chosen)} images in {time.time() - start:.0f}s", flush=True)
    with open(out_path, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["concept_name", "class_name", "n_images", "score"])
        for (c, cls), v in sorted(per.items()):
            wr.writerow([c, cls, len(v), np.mean(v)])
    print("done", flush=True)


if __name__ == "__main__":
    main()
