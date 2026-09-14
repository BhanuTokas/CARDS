"""Pulls the UTKFace image set from Hugging Face (`nlphuji/utk_faces`,
a zip archive, not a parquet shard -- the canonical mirror of
susanqq.github.io/UTKFace's own official release) for the cross-dataset
ConceptMask generalization check, prompted directly ("Let's try both
LFW and UTK if possible?"). The official UTKFace release is Google-
Drive-hosted, not a scriptable direct URL -- this HF mirror avoids that
friction entirely, same images.

Downloads the zip via `hf_hub_download` (works for any file in a
dataset repo, not just parquet -- no `datasets`/pyarrow dependency
needed for this one) and extracts it flat into `Datasets/UTKFace/images/`.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_CACHE = Path(r"C:\Users\btokas\Projects\Datasets\hf_cache")
OUTPUT_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\UTKFace")
os.environ["HF_HUB_CACHE"] = str(HF_CACHE)

REPO_ID = "nlphuji/utk_faces"
FILENAME = "utk_faces_images.zip"


def main():
    images_dir = OUTPUT_ROOT / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID}/{FILENAME} ...", flush=True)
    path = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=FILENAME))
    print(f"Downloaded to {path} ({path.stat().st_size / 1e6:.1f} MB), extracting ...", flush=True)

    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png")) and not n.startswith("__MACOSX")]
        print(f"  {len(names)} image entries found in archive", flush=True)
        for i, name in enumerate(names):
            data = zf.read(name)
            out_name = f"{i:05d}_{Path(name).name}"
            (images_dir / out_name).write_bytes(data)
            if (i + 1) % 5000 == 0:
                print(f"  [{i + 1}/{len(names)}] extracted", flush=True)

    path.unlink()
    n_saved = len(list(images_dir.glob("*.*")))
    print(f"\nSaved {n_saved} images to {images_dir}", flush=True)


if __name__ == "__main__":
    main()
