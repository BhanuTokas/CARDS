"""Per-black-box faithfulness ground truth for COCO-Stuff (ViT-B/16, ConvNeXt-Tiny, or any BACKBONES entry).

Why this exists: compute_coco_stuff_faithfulness.py measures delta_p with the ResNet18 only. The ViT / ConvNeXt
attribution results (run_cards_coco_stuff_model_variants.py, TCAV and PCBM variants) were scored against that
ResNet18 ground truth, i.e. against masking effects on a different model. This script produces the matching ground
truth for each black box.

Grounding DINO + SAM run ONCE per image, with the same thresholds, NMS, per-concept instance union and black_mask /
black_bbox strategies as the original script, and every requested black box is evaluated on those identical masked
images. So all ground truths share the same concept regions and differ only in the model.

Usage (repo root, GPU, weights/ populated):
    BACKBONES_TO_RUN=coco_stuff_vit_b_16,coco_stuff_convnext_tiny \
        python scripts/coco_stuff/build/compute_coco_stuff_faithfulness_variants.py
Optional env:
    SHARD=i/n          process every n-th image starting at i (parallel jobs); output gets a _shard{i}of{n} suffix
    LIMIT=n            first n pool images only (smoke test)
    SAVE_MASKS=dir     also save per-image concept union masks (bit-packed .npz), reusable by EAC
Outputs: results/coco_stuff/coco_stuff_faithfulness_{backbone}[_shardIofN].csv, same columns as the ResNet18 file.
Rows are streamed, so a killed job keeps what it wrote; concatenate shards afterwards (header once)."""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

from compute_coco_stuff_faithfulness import (  # noqa: E402
    BOX_THRESHOLD, BRODEN_DROPPED_CONCEPTS, BRODEN_ROOT, COCO_ROOT, DEVICE, GDINO_CKPT, GDINO_CONFIG, NMS_IOU,
    RESULTS_DIR, SAM_CKPT, SAM_TYPE, TEXT_THRESHOLD, _load_coco_multilabel, _match_phrase,
)

from cards.data.coco_stuff import COCO_STUFF_20_CLASSES
from cards.data.datasets import list_broden_concepts, load_coco_stuff
from cards.models.backbones import BACKBONES

FIELDS = [
    "concept_name", "image_path", "class_name", "class_label", "p_orig", "logit_orig",
    "p_masked_black_mask", "delta_p_black_mask", "logit_masked_black_mask", "delta_logit_black_mask",
    "p_masked_black_bbox", "delta_p_black_bbox", "logit_masked_black_bbox", "delta_logit_black_bbox",
]


@torch.no_grad()
def logits_of(model, preprocess, images: list[Image.Image]) -> torch.Tensor:
    x = torch.stack([preprocess(im) for im in images]).to(DEVICE)
    return torch.cat([model(x[s:s + 32]) for s in range(0, len(x), 32)]).cpu()


def main() -> None:
    names = os.environ.get("BACKBONES_TO_RUN", "coco_stuff_vit_b_16,coco_stuff_convnext_tiny").split(",")
    for n in names:
        if n not in BACKBONES:
            raise ValueError(f"unknown backbone {n!r}; available: {list(BACKBONES)}")
    shard_i, shard_n = (int(v) for v in os.environ.get("SHARD", "0/1").split("/"))
    suffix = f"_shard{shard_i}of{shard_n}" if shard_n > 1 else ""
    limit = int(os.environ.get("LIMIT", "0"))
    save_masks = Path(os.environ["SAVE_MASKS"]) if os.environ.get("SAVE_MASKS") else None
    if save_masks:
        save_masks.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    from groundingdino.util import box_ops
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T
    from segment_anything import SamPredictor, sam_model_registry

    gtf = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    gdino = load_model(str(GDINO_CONFIG), str(GDINO_CKPT)).to(DEVICE).eval()
    sam_predictor = SamPredictor(sam_model_registry[SAM_TYPE](checkpoint=str(SAM_CKPT)).to(DEVICE))
    models = {n: (BACKBONES[n].load_native().to(DEVICE).eval(), BACKBONES[n].preprocess) for n in names}
    print(f"black boxes: {names}  shard {shard_i}/{shard_n}", flush=True)

    concepts = [c for c in list_broden_concepts(BRODEN_ROOT) if c not in BRODEN_DROPPED_CONCEPTS]
    caption = " . ".join(concepts)
    pairs = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="train")
    image_paths = [p for p, lbl in pairs if lbl >= 0]
    if limit:
        image_paths = image_paths[:limit]
    image_paths = image_paths[shard_i::shard_n]
    multilabel = _load_coco_multilabel(COCO_ROOT, COCO_STUFF_20_CLASSES, "train")
    print(f"{len(image_paths)} images", flush=True)

    files = {n: open(RESULTS_DIR / f"coco_stuff_faithfulness_{n}{suffix}.csv", "w", newline="") for n in names}
    writers = {n: csv.DictWriter(files[n], fieldnames=FIELDS) for n in names}
    for w in writers.values():
        w.writeheader()

    for img_idx, image_path in enumerate(image_paths):
        pil = Image.open(image_path).convert("RGB")
        arr = np.array(pil); h, w = arr.shape[:2]
        t, _ = gtf(pil, None)
        with torch.no_grad():
            boxes, box_logits, phrases = predict(model=gdino, image=t.to(DEVICE), caption=caption,
                                                 box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, device=DEVICE)
        if len(boxes) == 0:
            continue
        babs = box_ops.box_cxcywh_to_xyxy(boxes.cpu()) * torch.tensor([w, h, w, h], dtype=torch.float32)
        raw: dict[str, list] = defaultdict(list)
        for bx, lg, ph in zip(babs, box_logits.tolist(), phrases):
            c = _match_phrase(ph, concepts)
            if c is not None:
                raw[c].append((bx, float(lg)))
        if not raw:
            continue
        concept_boxes = {}
        for c, prs in raw.items():
            bt = torch.stack([b for b, _ in prs]); keep = nms(bt, torch.tensor([s for _, s in prs]), NMS_IOU)
            concept_boxes[c] = [bt[i] for i in keep]

        sam_predictor.set_image(arr)
        labels = multilabel.get(str(image_path), np.zeros(len(COCO_STUFF_20_CLASSES), dtype=np.int8))
        names_c, bm_imgs, bb_imgs, saved = [], [], [], {}
        for c, bxs in concept_boxes.items():
            union = np.zeros((h, w), bool)
            for bx in bxs:
                union |= sam_predictor.predict(box=bx.numpy(), multimask_output=False)[0][0]
            if not union.any() or union.all():
                continue
            a = arr.copy(); a[union] = 0
            bbox = np.zeros((h, w), bool)
            for bx in bxs:
                x1, y1, x2, y2 = [int(v) for v in bx.clamp(min=0).tolist()]
                bbox[y1:min(y2, h), x1:min(x2, w)] = True
            b = arr.copy(); b[bbox] = 0
            names_c.append(c); bm_imgs.append(Image.fromarray(a)); bb_imgs.append(Image.fromarray(b)); saved[c] = union
        if not names_c:
            continue
        if save_masks:
            np.savez_compressed(save_masks / f"{Path(image_path).stem}.npz",
                                concepts=np.array(list(saved)), masks=np.packbits(np.stack(list(saved.values())), axis=-1),
                                w=w)
        for n, (model, pre) in models.items():
            lo = logits_of(model, pre, [pil])[0]
            lbm = logits_of(model, pre, bm_imgs); lbb = logits_of(model, pre, bb_imgs)
            po, pbm, pbb = torch.sigmoid(lo), torch.sigmoid(lbm), torch.sigmoid(lbb)
            for i, c in enumerate(names_c):
                for k, cls in enumerate(COCO_STUFF_20_CLASSES):
                    writers[n].writerow({
                        "concept_name": c, "image_path": str(image_path), "class_name": cls, "class_label": int(labels[k]),
                        "p_orig": float(po[k]), "logit_orig": float(lo[k]),
                        "p_masked_black_mask": float(pbm[i, k]), "delta_p_black_mask": float(po[k] - pbm[i, k]),
                        "logit_masked_black_mask": float(lbm[i, k]), "delta_logit_black_mask": float(lo[k] - lbm[i, k]),
                        "p_masked_black_bbox": float(pbb[i, k]), "delta_p_black_bbox": float(po[k] - pbb[i, k]),
                        "logit_masked_black_bbox": float(lbb[i, k]), "delta_logit_black_bbox": float(lo[k] - lbb[i, k]),
                    })
        if (img_idx + 1) % 100 == 0:
            for f in files.values():
                f.flush()
            print(f"[{img_idx + 1}/{len(image_paths)}]", flush=True)
    for f in files.values():
        f.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
