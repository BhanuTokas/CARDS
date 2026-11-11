"""Calibrate GroundingDINO box/text thresholds against COCO val2017 GT.

Pipeline:
  1. Finds exact-match overlap between COCO 80 class names and the 143
     active Broden concepts (underscore→space, case-insensitive).
     Prints count and names.
  2. Samples N_SAMPLE images from val2017 that contain ≥1 overlapping
     class (fixed seed). Caches the image-id list so every ablation run
     reuses the same images.
  3. Sweeps all (BOX_THRESHOLD, TEXT_THRESHOLD) combinations. For each:
       - runs GDINO with the overlapping concept names as a dot-caption
       - matches predicted phrases back to concept names
       - computes per-GT-box max IoU vs predicted boxes of the same concept
       - reports mean IoU, recall@IOU_MATCH_THRESH, precision@IOU_MATCH_THRESH
  4. Saves results to ABLATION_CSV. Prints top-5 combos by mean IoU and
     the single best recommendation.

Run on an interactive GPU node:
    module load gcc/12.5.0 && module load cuda-12.4.1-gcc-12.1.0
    cd /scratch/rnair21/CARDS
    export PYTHONNOUSERSITE=1
    /scratch/rnair21/envs/cards_env/bin/python \\
        scripts/coco_stuff/build/calibrate_gdino_thresholds.py
"""

from __future__ import annotations

import csv
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import torch
from PIL import Image

from cards.data.datasets import list_broden_concepts

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COCO_ROOT    = Path("/scratch/rnair21/LLMConceptEditing_old/coco")
BRODEN_ROOT  = Path("/scratch/rnair21/LLMConceptEditing_old/broden_concepts")
GDINO_CONFIG = Path("weights/GroundingDINO_SwinT_OGC.py")
GDINO_CKPT   = Path("weights/groundingdino_swint_ogc.pth")
RESULTS_DIR  = Path("results/coco_stuff")
SAMPLE_IDS_PATH = RESULTS_DIR / "gdino_calib_sample_ids.json"
ABLATION_CSV    = RESULTS_DIR / "gdino_threshold_ablation.csv"

BRODEN_DROPPED_CONCEPTS: set[str] = {
    "air_conditioner", "apron", "awning", "bathroom_s", "beak", "bedroom_s",
    "blotchy", "bumper", "can", "candlestick", "cap", "clock", "dining_room_s",
    "drinking_glass", "eye", "eyebrow", "faucet", "figurine", "handle",
    "inside_arm", "jar", "knob", "loudspeaker", "mouse", "mouth",
    "outside_arm", "street_s",
}

# ---------------------------------------------------------------------------
# Ablation settings
# ---------------------------------------------------------------------------
N_SAMPLE         = 1000
SEED             = 42
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
IOU_MATCH_THRESH = 0.5

BOX_THRESHOLDS  = [0.30, 0.35, 0.40, 0.45]
TEXT_THRESHOLDS = [0.25, 0.30, 0.35, 0.40]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    return name.lower().replace("_", " ").strip()


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(a[0], b[0]);  y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]);  y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _xywh_to_xyxy(box: list[float]) -> np.ndarray:
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def _find_overlap(coco_names: list[str], broden_concepts: list[str]) -> list[str]:
    """Return Broden concept names whose normalised form matches a COCO class."""
    broden_norm_map: dict[str, str] = {_normalize(c): c for c in broden_concepts}
    matched: list[str] = []
    for cname in coco_names:
        normed = _normalize(cname)
        if normed in broden_norm_map:
            matched.append(broden_norm_map[normed])
    return matched


def _match_phrase(phrase: str, concepts: list[str]) -> str | None:
    """Map a GDINO phrase back to a concept name (exact then partial)."""
    p = _normalize(phrase)
    norm_map = {_normalize(c): c for c in concepts}
    if p in norm_map:
        return norm_map[p]
    for normed, orig in norm_map.items():
        if normed in p or p in normed:
            return orig
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Broden 143 ──────────────────────────────────────────────────────────
    broden_143 = [
        c for c in list_broden_concepts(BRODEN_ROOT)
        if c not in BRODEN_DROPPED_CONCEPTS
    ]
    print(f"Broden active concepts: {len(broden_143)}", flush=True)

    # ── COCO train2017 annotations ───────────────────────────────────────────
    ann_path = COCO_ROOT / "annotations" / "instances_train2017.json"
    with open(ann_path) as f:
        coco = json.load(f)

    coco_cat_names: list[str] = [cat["name"] for cat in coco["categories"]]
    coco_catid_to_name: dict[int, str] = {
        cat["id"]: cat["name"] for cat in coco["categories"]
    }

    # ── Overlap ──────────────────────────────────────────────────────────────
    overlapping = _find_overlap(coco_cat_names, broden_143)
    overlap_norm_to_broden: dict[str, str] = {_normalize(c): c for c in overlapping}
    overlap_catid_to_concept: dict[int, str] = {
        cid: overlap_norm_to_broden[_normalize(name)]
        for cid, name in coco_catid_to_name.items()
        if _normalize(name) in overlap_norm_to_broden
    }
    overlap_cat_ids = set(overlap_catid_to_concept)

    print(f"\nOverlapping concept names (COCO ∩ Broden 143): {len(overlapping)}", flush=True)
    for c in sorted(overlapping):
        print(f"  {c}", flush=True)

    # ── Sample 1000 images ───────────────────────────────────────────────────
    if SAMPLE_IDS_PATH.exists():
        sample_ids: list[int] = json.loads(SAMPLE_IDS_PATH.read_text())
        print(
            f"\nLoaded {len(sample_ids)} cached sample IDs from {SAMPLE_IDS_PATH}",
            flush=True,
        )
    else:
        imgs_with_overlap: set[int] = set()
        for ann in coco["annotations"]:
            if ann["category_id"] in overlap_cat_ids:
                imgs_with_overlap.add(ann["image_id"])
        eligible = sorted(imgs_with_overlap)
        rng = np.random.default_rng(SEED)
        sample_ids = rng.choice(
            eligible, size=min(N_SAMPLE, len(eligible)), replace=False
        ).tolist()
        SAMPLE_IDS_PATH.write_text(json.dumps(sample_ids))
        print(
            f"\nSampled {len(sample_ids)} / {len(eligible)} eligible images "
            f"→ cached at {SAMPLE_IDS_PATH}",
            flush=True,
        )

    sample_id_set = set(sample_ids)

    # ── GT boxes per (image_id, concept) ─────────────────────────────────────
    gt_by_image: dict[int, dict[str, list[np.ndarray]]] = {
        iid: {} for iid in sample_ids
    }
    for ann in coco["annotations"]:
        if ann["image_id"] not in sample_id_set:
            continue
        if ann["category_id"] not in overlap_cat_ids:
            continue
        concept = overlap_catid_to_concept[ann["category_id"]]
        gt_by_image[ann["image_id"]].setdefault(concept, []).append(
            _xywh_to_xyxy(ann["bbox"])
        )

    img_id_to_fname: dict[int, str] = {
        img["id"]: img["file_name"] for img in coco["images"]
    }
    total_gt = sum(
        len(boxes)
        for img_dict in gt_by_image.values()
        for boxes in img_dict.values()
    )
    print(f"Total GT boxes across {len(sample_ids)} images: {total_gt}", flush=True)

    # ── GDINO caption: all overlapping concepts ───────────────────────────────
    caption = " . ".join(overlapping)

    # ── Load GDINO (once) ─────────────────────────────────────────────────────
    from groundingdino.util.inference import load_model, predict
    from groundingdino.util import box_ops
    import groundingdino.datasets.transforms as T

    gdino_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    gdino = load_model(str(GDINO_CONFIG), str(GDINO_CKPT)).to(DEVICE).eval()
    print(f"\nGrounding DINO loaded ({GDINO_CKPT.name})  device={DEVICE}", flush=True)

    # ── Ablation ──────────────────────────────────────────────────────────────
    combos = list(product(BOX_THRESHOLDS, TEXT_THRESHOLDS))
    print(
        f"\nAblating {len(combos)} threshold combinations "
        f"(box={BOX_THRESHOLDS}  text={TEXT_THRESHOLDS})\n",
        flush=True,
    )
    header = (
        f"{'box':>6}  {'text':>6}  {'mean_IoU':>8}  "
        f"{'rec@.5':>7}  {'prec@.5':>8}  {'n_pred':>7}  {'n_gt':>6}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    ablation_rows: list[dict] = []

    for box_thresh, text_thresh in combos:
        # Per-GT-box IoU (for recall) and per-pred-box IoU (for precision)
        gt_ious: list[float] = []    # one per GT box
        pred_ious: list[float] = []  # one per predicted box
        n_pred_total = 0

        for img_id in sample_ids:
            fname   = img_id_to_fname[img_id]
            img_path = COCO_ROOT / "train2017" / fname
            try:
                image_pil = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            w, h  = image_pil.size
            scale = torch.tensor([w, h, w, h], dtype=torch.float32)

            image_t, _ = gdino_transform(image_pil, None)
            with torch.no_grad():
                boxes, _, phrases = predict(
                    model=gdino,
                    image=image_t.to(DEVICE),
                    caption=caption,
                    box_threshold=box_thresh,
                    text_threshold=text_thresh,
                    device=DEVICE,
                )

            # group predictions by matched concept
            pred_by_concept: dict[str, list[np.ndarray]] = {}
            if len(boxes) > 0:
                boxes_abs = (
                    box_ops.box_cxcywh_to_xyxy(boxes.cpu()) * scale
                ).numpy()
                for box, phrase in zip(boxes_abs, phrases):
                    concept = _match_phrase(phrase, overlapping)
                    if concept is None:
                        continue
                    pred_by_concept.setdefault(concept, []).append(box)
            n_pred_total += sum(len(v) for v in pred_by_concept.values())

            for concept, gt_boxes in gt_by_image[img_id].items():
                preds = pred_by_concept.get(concept, [])

                # recall: for each GT box, best IoU with any prediction
                for gt in gt_boxes:
                    best = max((_box_iou(gt, p) for p in preds), default=0.0)
                    gt_ious.append(best)

                # precision: for each prediction, best IoU with any GT box
                for pred in preds:
                    best = max((_box_iou(pred, gt) for gt in gt_boxes), default=0.0)
                    pred_ious.append(best)

        mean_iou  = float(np.mean(gt_ious))  if gt_ious  else 0.0
        recall    = float(np.mean([v >= IOU_MATCH_THRESH for v in gt_ious]))  if gt_ious  else 0.0
        precision = float(np.mean([v >= IOU_MATCH_THRESH for v in pred_ious])) if pred_ious else 0.0
        n_gt      = len(gt_ious)

        row = dict(
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            mean_iou=mean_iou,
            recall_at_50=recall,
            precision_at_50=precision,
            n_predictions=n_pred_total,
            n_gt_boxes=n_gt,
        )
        ablation_rows.append(row)

        print(
            f"{box_thresh:>6.2f}  {text_thresh:>6.2f}  "
            f"{mean_iou:>8.4f}  {recall:>7.4f}  {precision:>8.4f}  "
            f"{n_pred_total:>7d}  {n_gt:>6d}",
            flush=True,
        )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(ABLATION_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_rows)
    print(f"\nSaved ablation results → {ABLATION_CSV}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Top 5 by mean IoU ===", flush=True)
    for row in sorted(ablation_rows, key=lambda r: r["mean_iou"], reverse=True)[:5]:
        print(
            f"  box={row['box_threshold']:.2f}  text={row['text_threshold']:.2f}  "
            f"mean_IoU={row['mean_iou']:.4f}  "
            f"rec@.5={row['recall_at_50']:.4f}  "
            f"prec@.5={row['precision_at_50']:.4f}",
            flush=True,
        )

    best = max(ablation_rows, key=lambda r: r["mean_iou"])
    print(
        f"\nRecommended thresholds → "
        f"BOX_THRESHOLD={best['box_threshold']:.2f}  "
        f"TEXT_THRESHOLD={best['text_threshold']:.2f}  "
        f"(mean_IoU={best['mean_iou']:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
