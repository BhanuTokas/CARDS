"""Ground-truth faithfulness for COCO-Stuff: Grounding DINO + SAM + ResNet18.

Runs on COCO train2017 images that contain >=1 of the 20 target classes.
For each image:
  1. Grounding DINO (all 143 Broden concepts batched, dot-separated) ->
     per-concept bounding boxes. Each box is assigned to exactly ONE concept
     (the phrase returned by GDINO, which is already argmax-token for that
     box under dot-separation -- avoids multi-token fusion artifacts).
     NMS (IoU=0.5) is applied per concept to suppress duplicate detections
     of the same instance while keeping genuinely distinct instances.
  2. SAM (box prompt per surviving instance) -> one mask per instance.
     All instance masks for the same concept are unioned into one combined
     mask, so ALL instances of a concept are masked simultaneously.
  3. mask_region (blur fill) + ResNet18 -> delta_p / delta_logit for all 20
     classes. The original-image logits are computed once per image and
     shared across all concepts detected in that image.

Outputs (in results/coco_stuff/):
  coco_stuff_faithfulness.csv
    concept_name, image_path, class_name, class_label,
    p_orig, p_masked, delta_p, logit_orig, logit_masked, delta_logit
  coco_stuff_faithfulness_detection_counts.csv
    concept_name, n_images, n_instances, n_total_images,
    image_detection_rate, mean_instances_per_image

Prerequisites:
  pip install groundingdino-py segment-anything
  Download weights into weights/ (relative to CARDS root):
    gdino_swint_ogc.pth            -- GroundingDINO SwinT backbone
    GroundingDINO_SwinT_OGC.py     -- matching config
    sam_vit_h_4b8939.pth           -- SAM ViT-H checkpoint
  Train the ResNet18 classifier first:
    uv run python scripts/coco_stuff/build/train_coco_stuff_classifier.py

Thresholds (calibrated via calibrate_gdino_thresholds.py on 1k train2017 images):
  BOX_THRESHOLD  = 0.30
  TEXT_THRESHOLD = 0.30  (mean_IoU=0.6708)
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet18
from torchvision.ops import nms

from cards.data.coco_stuff import COCO_STUFF_20_CLASSES
from cards.data.datasets import list_broden_concepts, load_coco_stuff
from cards.models.coco_stuff_black_box import _coco_stuff_preprocess

# ---------------------------------------------------------------------------
# Paths -- override to match your cluster layout
# ---------------------------------------------------------------------------
COCO_ROOT    = Path("/scratch/rnair21/LLMConceptEditing_old/coco")
BRODEN_ROOT  = Path("/scratch/rnair21/LLMConceptEditing_old/broden_concepts")

BRODEN_DROPPED_CONCEPTS: set[str] = {
    "air_conditioner", "apron", "awning", "bathroom_s", "beak", "bedroom_s",
    "blotchy", "bumper", "can", "candlestick", "cap", "clock", "dining_room_s",
    "drinking_glass", "eye", "eyebrow", "faucet", "figurine", "handle",
    "inside_arm", "jar", "knob", "loudspeaker", "mouse", "mouth",
    "outside_arm", "street_s",
}
CKPT_PATH    = Path("trained_models_new/coco_stuff/resnet18_coco_stuff.pt")
GDINO_CONFIG = Path("weights/GroundingDINO_SwinT_OGC.py")
GDINO_CKPT   = Path("weights/groundingdino_swint_ogc.pth")
SAM_CKPT     = Path("weights/sam_vit_h_4b8939.pth")
SAM_TYPE     = "vit_h"
RESULTS_DIR  = Path("results/coco_stuff")

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BOX_THRESHOLD  = 0.30
TEXT_THRESHOLD = 0.30
NMS_IOU        = 0.50
SEED           = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_coco_multilabel(coco_root: Path, classes: list[str], split: str) -> dict[str, np.ndarray]:
    """Return image_path (str) → binary label vector (len = len(classes))."""
    ann_path = coco_root / "annotations" / f"instances_{split}2017.json"
    with open(ann_path) as f:
        data = json.load(f)

    cat_name_to_id = {cat["name"]: cat["id"] for cat in data["categories"]}
    cat_id_to_idx: dict[int, int] = {}
    for idx, cls in enumerate(classes):
        coco_id = cat_name_to_id.get(cls)
        if coco_id is not None:
            cat_id_to_idx[coco_id] = idx
    target_cat_ids = set(cat_id_to_idx)

    image_classes: dict[int, set] = defaultdict(set)
    for ann in data["annotations"]:
        if ann["category_id"] in target_cat_ids:
            image_classes[ann["image_id"]].add(cat_id_to_idx[ann["category_id"]])

    id_to_fname = {img["id"]: img["file_name"] for img in data["images"]}
    image_dir = coco_root / f"{split}2017"
    result: dict[str, np.ndarray] = {}
    for img_id, cls_set in image_classes.items():
        vec = np.zeros(len(classes), dtype=np.int8)
        for idx in cls_set:
            vec[idx] = 1
        result[str(image_dir / id_to_fname[img_id])] = vec
    return result


def _load_resnet(ckpt: Path, n_classes: int, device: str) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device).eval()


def _match_phrase(phrase: str, concepts: list[str]) -> str | None:
    """Map a GDINO phrase back to the original Broden concept name.

    GDINO returns lowercased phrase strings derived from the dot-separated
    caption tokens. Try exact match first, then contained-in match (handles
    minor tokenisation differences such as extra whitespace or partial spans).
    """
    p = phrase.lower().strip()
    for c in concepts:
        if c.lower() == p:
            return c
    for c in concepts:
        c_lower = c.lower()
        if c_lower in p or p in c_lower:
            return c
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Grounding DINO --------------------------------------------------
    from groundingdino.util.inference import load_model, predict
    from groundingdino.util import box_ops
    import groundingdino.datasets.transforms as T

    gdino_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    gdino = load_model(str(GDINO_CONFIG), str(GDINO_CKPT)).to(DEVICE).eval()
    print(f"Grounding DINO: {GDINO_CKPT.name}", flush=True)

    # --- SAM -------------------------------------------------------------
    from segment_anything import sam_model_registry, SamPredictor

    sam = sam_model_registry[SAM_TYPE](checkpoint=str(SAM_CKPT)).to(DEVICE)
    sam_predictor = SamPredictor(sam)
    print(f"SAM ({SAM_TYPE}): {SAM_CKPT.name}", flush=True)

    # --- ResNet18 --------------------------------------------------------
    resnet = _load_resnet(CKPT_PATH, len(COCO_STUFF_20_CLASSES), DEVICE)
    preprocess = _coco_stuff_preprocess()
    print(f"ResNet18: {CKPT_PATH}", flush=True)

    # --- Concepts & image pool ------------------------------------------
    concepts = [c for c in list_broden_concepts(BRODEN_ROOT) if c not in BRODEN_DROPPED_CONCEPTS]
    caption  = " . ".join(concepts)   # single batched GDINO query
    print(f"Concepts: {len(concepts)} Broden concepts ({len(BRODEN_DROPPED_CONCEPTS)} dropped)", flush=True)

    all_pairs   = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="train")
    image_paths = [p for p, lbl in all_pairs if lbl >= 0]
    n_total     = len(image_paths)
    print(f"Pool: {n_total} COCO train2017 images with >=1 target class", flush=True)

    multilabel_map = _load_coco_multilabel(COCO_ROOT, COCO_STUFF_20_CLASSES, "train")
    print(f"Multi-label map: {len(multilabel_map)} images", flush=True)

    # --- Bookkeeping -----------------------------------------------------
    image_counts: Counter    = Counter()   # concept -> n distinct images detected in
    instance_counts: Counter = Counter()   # concept -> n total NMS instances (reference)
    records: list[dict] = []

    for img_idx, image_path in enumerate(image_paths):
        image_pil = Image.open(image_path).convert("RGB")
        image_np  = np.array(image_pil)
        h, w      = image_np.shape[:2]

        # --- GDINO -------------------------------------------------------
        image_t, _ = gdino_transform(image_pil, None)
        with torch.no_grad():
            boxes, box_logits, phrases = predict(
                model=gdino,
                image=image_t.to(DEVICE),
                caption=caption,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=DEVICE,
            )

        if len(boxes) == 0:
            if (img_idx + 1) % 500 == 0:
                print(f"[{img_idx+1}/{n_total}] no detections", flush=True)
            continue

        # Convert normalised [cx,cy,w,h] -> absolute [x1,y1,x2,y2]
        scale     = torch.tensor([w, h, w, h], dtype=torch.float32)
        boxes_abs = box_ops.box_cxcywh_to_xyxy(boxes.cpu()) * scale

        # Group all GDINO boxes by concept, then apply NMS per concept to
        # suppress duplicate detections of the same instance while keeping
        # genuinely distinct instances.
        concept_boxes_raw: dict[str, list[tuple[torch.Tensor, float]]] = defaultdict(list)
        for box, logit, phrase in zip(boxes_abs, box_logits.tolist(), phrases):
            concept = _match_phrase(phrase, concepts)
            if concept is None:
                continue
            concept_boxes_raw[concept].append((box, float(logit)))

        if not concept_boxes_raw:
            continue

        # NMS per concept
        concept_boxes: dict[str, list[torch.Tensor]] = {}
        for concept, pairs in concept_boxes_raw.items():
            boxes_t  = torch.stack([b for b, _ in pairs])           # (N, 4)
            scores_t = torch.tensor([s for _, s in pairs])          # (N,)
            keep     = nms(boxes_t, scores_t, NMS_IOU)
            concept_boxes[concept] = [boxes_t[i] for i in keep]

        # --- Shared: original-image logits (computed once per image) -----
        pix_orig = preprocess(image_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit_orig = resnet(pix_orig)[0]   # (20,)
        p_orig = torch.sigmoid(logit_orig)

        # --- SAM: set image once, union all instance masks per concept ---
        sam_predictor.set_image(image_np)
        img_labels = multilabel_map.get(str(image_path), np.zeros(len(COCO_STUFF_20_CLASSES), dtype=np.int8))

        for concept_name, boxes in concept_boxes.items():
            # --- Strategy 1: black_mask — SAM union mask filled with black ---
            union_mask = np.zeros((h, w), dtype=bool)
            for box in boxes:
                masks, _, _ = sam_predictor.predict(
                    box=box.numpy(),
                    multimask_output=False,
                )
                union_mask |= masks[0]

            if not union_mask.any() or union_mask.all():
                continue

            img_arr_bm = np.array(image_pil).copy()
            img_arr_bm[union_mask] = 0
            pix_bm = preprocess(Image.fromarray(img_arr_bm)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logit_bm = resnet(pix_bm)[0]
            p_bm = torch.sigmoid(logit_bm)

            # --- Strategy 2: black_bbox — union of NMS bbox rectangles filled with black ---
            bbox_mask = np.zeros((h, w), dtype=bool)
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box.clamp(min=0).tolist()]
                x2 = min(x2, w); y2 = min(y2, h)
                bbox_mask[y1:y2, x1:x2] = True

            img_arr_bb = np.array(image_pil).copy()
            img_arr_bb[bbox_mask] = 0
            pix_bb = preprocess(Image.fromarray(img_arr_bb)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logit_bb = resnet(pix_bb)[0]
            p_bb = torch.sigmoid(logit_bb)

            image_counts[concept_name]    += 1
            instance_counts[concept_name] += len(boxes)

            for cls_idx, cls_name in enumerate(COCO_STUFF_20_CLASSES):
                records.append({
                    "concept_name":              concept_name,
                    "image_path":                str(image_path),
                    "class_name":                cls_name,
                    "class_label":               int(img_labels[cls_idx]),
                    "p_orig":                    float(p_orig[cls_idx]),
                    "p_masked_black_mask":        float(p_bm[cls_idx]),
                    "delta_p_black_mask":         float(p_orig[cls_idx] - p_bm[cls_idx]),
                    "logit_masked_black_mask":    float(logit_bm[cls_idx]),
                    "delta_logit_black_mask":     float(logit_orig[cls_idx] - logit_bm[cls_idx]),
                    "p_masked_black_bbox":        float(p_bb[cls_idx]),
                    "delta_p_black_bbox":         float(p_orig[cls_idx] - p_bb[cls_idx]),
                    "logit_masked_black_bbox":    float(logit_bb[cls_idx]),
                    "delta_logit_black_bbox":     float(logit_orig[cls_idx] - logit_bb[cls_idx]),
                    "logit_orig":                float(logit_orig[cls_idx]),
                })

        if (img_idx + 1) % 100 == 0:
            print(
                f"[{img_idx+1:>6d}/{n_total}]  "
                f"concepts active: {len(image_counts)}  "
                f"records: {len(records)}",
                flush=True,
            )

    # --- Save faithfulness CSV ------------------------------------------
    faith_path = RESULTS_DIR / "coco_stuff_faithfulness.csv"
    fields = [
        "concept_name", "image_path", "class_name", "class_label",
        "p_orig", "logit_orig",
        "p_masked_black_mask", "delta_p_black_mask", "logit_masked_black_mask", "delta_logit_black_mask",
        "p_masked_black_bbox", "delta_p_black_bbox", "logit_masked_black_bbox", "delta_logit_black_bbox",
    ]
    with open(faith_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved {faith_path}  ({len(records):,} records)", flush=True)

    # --- Save detection counts CSV --------------------------------------
    counts_path = RESULTS_DIR / "coco_stuff_faithfulness_detection_counts.csv"
    with open(counts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "concept_name", "n_images", "n_instances",
            "n_total_images", "image_detection_rate", "mean_instances_per_image",
        ])
        for concept in concepts:
            n_img = image_counts[concept]
            n_ins = instance_counts[concept]
            mean_ins = n_ins / n_img if n_img > 0 else 0.0
            writer.writerow([concept, n_img, n_ins, n_total, f"{n_img / n_total:.4f}", f"{mean_ins:.2f}"])
    print(f"Saved {counts_path}", flush=True)

    # --- Detection summary ----------------------------------------------
    detected = [c for c in concepts if image_counts[c] > 0]
    unused   = [c for c in concepts if image_counts[c] == 0]
    print(f"\nDetected: {len(detected)}/{len(concepts)} concepts", flush=True)

    print(f"\nAll concepts by image count (n_images | n_instances | mean/image):", flush=True)
    for concept in sorted(concepts, key=lambda c: -image_counts[c]):
        n_img = image_counts[concept]
        n_ins = instance_counts[concept]
        mean_ins = n_ins / n_img if n_img > 0 else 0.0
        print(f"  {concept:<30s}  images={n_img:>6d}  instances={n_ins:>7d}  mean/img={mean_ins:.2f}", flush=True)

    print(f"\nZero-detection concepts ({len(unused)}) -- likely textures/abstracts:", flush=True)
    for c in sorted(unused):
        print(f"  {c}", flush=True)


if __name__ == "__main__":
    main()
