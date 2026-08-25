"""Phase 1 (notes/pcbm_correlation_investigation.md, CARDS vs. TCAV vs.
PCBM plan): pulls a small, class-targeted local slice of ILSVRC/imagenet-1k
from Hugging Face for prototyping, instead of the full ~150GB dataset.

ImageNet-1k's parquet shards are near-uniformly class-shuffled (verified:
987 of 1000 labels present in the first 4,358-row train shard, most
common label appearing only 12 times) -- so getting a few hundred images
per specific class means scanning many shards and filtering, not
downloading a handful and hoping. This script does that: streams shards
one at a time, extracts rows matching TARGET_CLASSES, writes them to
Datasets/imagenet_slice/{train,val}/<class_name>/, and deletes each raw
shard immediately after processing so peak disk usage stays small
regardless of how many shards get scanned.

Val is exhaustive (all 14 shards, ~6.5GB, giving the full ~50/class
Broden-standard split, used only for held-out evaluation). Train is
budget-capped (stops once every target class has >= TRAIN_TARGET_PER_CLASS
images, or MAX_TRAIN_SHARDS is hit, whichever first) -- the achieved
per-class count is reported honestly, not assumed to hit the target.
"""

from __future__ import annotations

import io
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

HF_CACHE = Path(r"C:\Users\btokas\Projects\Datasets\hf_cache")
OUTPUT_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\imagenet_slice")
os.environ["HF_HUB_CACHE"] = str(HF_CACHE)

REPO_ID = "ILSVRC/imagenet-1k"
N_VAL_SHARDS = 14
N_TRAIN_SHARDS_TOTAL = 294
TRAIN_TARGET_PER_CLASS = 150
MAX_TRAIN_SHARDS = 50

# (label_idx, wnid, class_name) -- hand-picked from IMAGENET2012_CLASSES,
# matched against NetDissect Broden's c_object.csv vocabulary (car,
# ashcan, ... match Broden concept names directly; others are the closest
# clean ImageNet-1k class for a Broden object concept, e.g. "chair" ->
# rocking_chair/folding_chair, "cat"/"dog" -> specific breeds since
# ImageNet has no generic cat/dog class). Includes the car/motorbike pair
# central to the v19/v20 diagnosis (sports_car, convertible vs.
# motor_scooter, tricycle) for continuity with that investigation.
TARGET_CLASSES = [
    (817, "n04285008", "sports_car"),
    (511, "n03100240", "convertible"),
    (670, "n03791053", "motor_scooter"),
    (870, "n04482393", "tricycle"),
    (555, "n03345487", "fire_engine"),
    (765, "n04099969", "rocking_chair"),
    (559, "n03376595", "folding_chair"),
    (532, "n03201208", "dining_table"),
    (526, "n03179701", "desk"),
    (703, "n03891251", "park_bench"),
    (883, "n04522168", "vase"),
    (721, "n03938244", "pillow"),
    (898, "n04557648", "water_bottle"),
    (409, "n02708093", "analog_clock"),
    (923, "n07579787", "plate"),
    (659, "n03775546", "mixing_bowl"),
    (412, "n02747177", "ashcan"),
    (495, "n03018349", "china_cabinet"),
    (619, "n03637318", "lampshade"),
    (790, "n04204238", "shopping_basket"),
    (281, "n02123045", "tabby_cat"),
    (284, "n02123597", "siamese_cat"),
    (285, "n02124075", "egyptian_cat"),
    (207, "n02099601", "golden_retriever"),
    (208, "n02099712", "labrador_retriever"),
]
LABEL_TO_NAME = {idx: name for idx, _, name in TARGET_CLASSES}


def process_shard(filename: str, split: str, counts: dict[str, int]) -> int:
    """Downloads one shard, extracts matching rows, writes images, deletes
    the raw shard. Returns the number of matching rows found."""
    path = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename))
    table = pq.read_table(path, columns=["image", "label"])
    labels = table.column("label").to_pylist()
    images = table.column("image").to_pylist()
    n_matched = 0
    for label, image_struct in zip(labels, images):
        if label not in LABEL_TO_NAME:
            continue
        class_name = LABEL_TO_NAME[label]
        out_dir = OUTPUT_ROOT / split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = counts.get(class_name, 0)
        img = Image.open(io.BytesIO(image_struct["bytes"])).convert("RGB")
        img.save(out_dir / f"{existing:04d}.jpg", quality=92)
        counts[class_name] = existing + 1
        n_matched += 1
    # Delete the raw shard immediately -- only the filtered, matching
    # images are worth keeping locally; the full shard was only ever a
    # scanning cost, per this script's docstring.
    path.unlink()
    return n_matched


def run_val() -> dict[str, int]:
    counts: dict[str, int] = {}
    for i in range(N_VAL_SHARDS):
        filename = f"data/validation-{i:05d}-of-{N_VAL_SHARDS:05d}.parquet"
        n = process_shard(filename, "val", counts)
        print(f"[val {i + 1}/{N_VAL_SHARDS}] {filename}: {n} matching rows, "
              f"running totals: {counts}", flush=True)
    return counts


def run_train() -> dict[str, int]:
    counts: dict[str, int] = {}
    for i in range(MAX_TRAIN_SHARDS):
        filename = f"data/train-{i:05d}-of-{N_TRAIN_SHARDS_TOTAL:05d}.parquet"
        n = process_shard(filename, "train", counts)
        min_count = min((counts.get(name, 0) for _, _, name in TARGET_CLASSES), default=0)
        print(f"[train {i + 1}/{MAX_TRAIN_SHARDS}] {filename}: {n} matching rows, "
              f"min per-class count so far: {min_count}", flush=True)
        if min_count >= TRAIN_TARGET_PER_CLASS:
            print(f"All classes reached {TRAIN_TARGET_PER_CLASS}+ after {i + 1} shards -- stopping early.", flush=True)
            break
    return counts


if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=== val (exhaustive, all 14 shards) ===", flush=True)
    val_counts = run_val()
    print("\n=== train (budget-capped) ===", flush=True)
    train_counts = run_train()

    manifest = {
        "target_classes": [{"label_idx": idx, "wnid": wnid, "name": name} for idx, wnid, name in TARGET_CLASSES],
        "val_counts": val_counts,
        "train_counts": train_counts,
    }
    with open(OUTPUT_ROOT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== final counts ===")
    for _, _, name in TARGET_CLASSES:
        print(f"{name}: train={train_counts.get(name, 0)} val={val_counts.get(name, 0)}")

    shutil.rmtree(HF_CACHE, ignore_errors=True)
    print("\nCleaned up HF shard cache. Done.")
