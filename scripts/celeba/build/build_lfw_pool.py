"""Pulls the full LFW (Labeled Faces in the Wild) image set from Hugging
Face (`logasja/lfw`, single parquet shard, 13,233 images -- the
canonical LFW count) for the cross-dataset ConceptMask generalization
check, prompted directly ("test if ConceptMask still works if we use a
different dataset for querying"). The official LFW host
(vis-www.cs.umass.edu) is unreachable from this environment (DNS
failure, confirmed directly) -- this HF mirror is a drop-in substitute,
same images.

Follows `scripts/imagenet/build_imagenet_slice.py`'s own established
pattern (hf_hub_download + pyarrow, not the full `datasets` library) --
LFW is small enough (one ~470MB parquet) to pull as a single shard, no
budget-capping/streaming needed.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

HF_CACHE = Path(r"C:\Users\btokas\Projects\Datasets\hf_cache")
OUTPUT_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\LFW")
os.environ["HF_HUB_CACHE"] = str(HF_CACHE)

REPO_ID = "logasja/lfw"
FILENAME = "data/train-00000-of-00001.parquet"


def main():
    images_dir = OUTPUT_ROOT / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID}/{FILENAME} ...", flush=True)
    path = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=FILENAME))
    table = pq.read_table(path, columns=["image", "label"])
    images = table.column("image").to_pylist()
    print(f"Loaded {len(images)} rows, extracting images ...", flush=True)

    n_saved = 0
    for i, image_struct in enumerate(images):
        img = Image.open(io.BytesIO(image_struct["bytes"])).convert("RGB")
        img.save(images_dir / f"{i:05d}.jpg", quality=95)
        n_saved += 1
        if (i + 1) % 2000 == 0:
            print(f"  [{i + 1}/{len(images)}] saved", flush=True)

    path.unlink()
    print(f"\nSaved {n_saved} images to {images_dir}", flush=True)


if __name__ == "__main__":
    main()
