"""Rebuild BAM images (held-out val or train split), following BAM's construction script.

BAM (Yang and Kim, 2019) publishes models and a construction script, not images.
That script sets no random seed, so object sizes and positions cannot be matched
exactly. The held-out objects and scenes can be: they are indices 900-999 of the
sorted object and scene lists. Run in the BAM environment (needs pycocotools).

Two deliberate details, both inherited from the original script:
  * Grayscale COCO images are skipped. There, `I * mask` raises and a bare
    `except` moves on, which changes which 1000 objects are selected.
  * Pure black object pixels become transparent.

The original passes 'stop_sign' to `getCatIds`, but the COCO category is
'stop sign'. See --stop-sign-name.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

NUM_IMAGES_PER_CLASS = 1000
VAL_START = 900
OBJ_NAMES = [
    "backpack", "bird", "dog", "elephant", "kite", "pizza", "stop_sign",
    "toilet", "truck", "zebra",
]
SCENE_NAMES = [
    "bamboo_forest", "bedroom", "bowling_alley", "bus_interior", "cockpit",
    "corn_field", "laundromat", "runway", "ski_slope", "track/outdoor",
]
RESIZE_LO, RESIZE_HI = np.sqrt(3), np.sqrt(2)


def coco_image_path(coco_root: Path, image_id: int) -> Path:
    """COCO 2017 images are the 2014 files under their old names."""
    for split in ("train2014", "val2014"):
        path = coco_root / split / f"COCO_{split}_{image_id:012d}.jpg"
        if path.exists():
            return path
    raise FileNotFoundError(f"COCO image {image_id}")


def extract_crop(coco: COCO, coco_root: Path, img: dict, cat_ids: list) -> Image.Image | None:
    """Return the largest annotation as an RGBA crop, or None if BAM would skip it."""
    with Image.open(coco_image_path(coco_root, img["id"])) as handle:
        if handle.mode != "RGB":
            return None
        pixels = np.asarray(handle)

    anns = coco.loadAnns(coco.getAnnIds(imgIds=img["id"], catIds=cat_ids, iscrowd=None))
    best, best_area = None, 0
    for ann in anns:
        if ann["area"] > best_area:
            best, best_area = ann, ann["area"]
    if best is None:
        return None

    x, y, w, h = best["bbox"]
    y0, y1, x0, x1 = int(y), int(y + h), int(x), int(x + w)
    mask = coco.annToMask(best).astype(np.uint8)
    crop = (pixels * mask[..., None])[y0:y1, x0:x1, :]
    if crop.size == 0:
        return None
    alpha = np.where(crop.any(axis=-1), 255, 0).astype(np.uint8)
    rgba = np.dstack([crop, alpha])
    rgba[alpha == 0] = 0
    return Image.fromarray(rgba)


def select_objects(coco: COCO, coco_root: Path, obj_name: str, coco_name: str) -> list:
    """Select BAM's 1000 objects for a class, sorted by file name as BAM does."""
    cat_ids = coco.getCatIds(catNms=[coco_name])
    imgs = coco.loadImgs(coco.getImgIds(catIds=cat_ids))
    kept = []
    for img in imgs:
        crop = extract_crop(coco, coco_root, img, cat_ids)
        if crop is None:
            continue
        kept.append((img["file_name"], crop))
        if len(kept) == NUM_IMAGES_PER_CLASS:
            break
    if len(kept) < NUM_IMAGES_PER_CLASS:
        raise RuntimeError(f"{obj_name}: only {len(kept)} usable objects")
    kept.sort(key=lambda item: item[0])
    print(f"{obj_name}: selected {len(kept)}, {len(cat_ids)} category ids", flush=True)
    return kept


def paste(scene: Image.Image, crop: Image.Image, rng: np.random.Generator):
    """Resize and place one object exactly as BAM does."""
    scene_w, scene_h = scene.size
    obj_w, obj_h = crop.size
    if obj_w / scene_w > obj_h / scene_h:
        new_w = int(rng.integers(int(scene_w / RESIZE_LO), int(scene_w / RESIZE_HI)))
        new_h = int(new_w / obj_w * obj_h)
    else:
        new_h = int(rng.integers(int(scene_h / RESIZE_LO), int(scene_h / RESIZE_HI)))
        new_w = int(new_h / obj_h * obj_w)
    row = int(rng.integers(0, scene_h - new_h))
    col = int(rng.integers(0, scene_w - new_w))

    resized = crop.resize((new_w, new_h), Image.Resampling.BILINEAR)
    composite = scene.copy()
    composite.paste(resized, (col, row), resized)
    alpha = Image.new("L", scene.size, 0)
    alpha.paste(resized.getchannel("A"), (col, row))
    return composite.convert("RGB"), alpha, (row, col, row + new_h, col + new_w)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, required=True,
                        help="Directory holding train2014/ and val2014/.")
    parser.add_argument("--annotations", type=Path, required=True,
                        help="instances_train2017.json")
    parser.add_argument("--miniplaces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["val", "train"], default="val",
                        help="val = indices 900-999 (held out); train = 0-899.")
    parser.add_argument("--composite-only", action="store_true",
                        help="Skip scene-only images and object masks.")
    parser.add_argument("--objects", nargs="*", default=OBJ_NAMES)
    parser.add_argument("--stop-sign-name", default="stop sign",
                        help="Pass 'stop_sign' for the published script's literal "
                             "behaviour, which matches no COCO category.")
    args = parser.parse_args()

    indices = range(VAL_START, NUM_IMAGES_PER_CLASS) if args.split == "val" else range(VAL_START)
    subdirs = ("composite",) if args.composite_only else ("composite", "scene_only", "object_mask")
    for sub in subdirs:
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    coco = COCO(str(args.annotations))
    rng = np.random.default_rng(args.seed)
    rows = []
    for obj_name in args.objects:
        coco_name = args.stop_sign_name if obj_name == "stop_sign" else obj_name
        objects = select_objects(coco, args.coco_root, obj_name, coco_name)
        for scene_label, scene_name in enumerate(SCENE_NAMES):
            scene_dir = args.miniplaces / "images" / "train" / scene_name[0] / scene_name
            scene_files = sorted(p for p in scene_dir.iterdir() if p.suffix == ".jpg")
            scene_short = scene_name.split("/")[0]
            for index in indices:
                coco_file, crop = objects[index]
                with Image.open(scene_files[index]) as handle:
                    scene = handle.convert("RGBA")
                composite, alpha, box = paste(scene, crop, rng)
                stem = f"{obj_name}-{scene_short}-{index:04d}"
                composite.save(args.out / "composite" / f"{stem}.jpg", format="jpeg")
                if not args.composite_only:
                    scene.convert("RGB").save(args.out / "scene_only" / f"{stem}.jpg", format="jpeg")
                    alpha.save(args.out / "object_mask" / f"{stem}.png")
                rows.append({
                    "file": f"{stem}.jpg",
                    "object": obj_name,
                    "scene": scene_short,
                    "object_label": OBJ_NAMES.index(obj_name),
                    "scene_label": scene_label,
                    "coco_file": coco_file,
                    "scene_file": scene_files[index].name,
                    "row0": box[0], "col0": box[1], "row1": box[2], "col1": box[3],
                    "object_fraction": float((np.asarray(alpha) > 0).mean()),
                })
        print(f"{obj_name}: wrote {len(rows)} images so far", flush=True)

    with open(args.out / "manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"done: {len(rows)} images in {args.out}")


if __name__ == "__main__":
    main()
