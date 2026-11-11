"""Explain Any Concept (EAC) on BAM, step 1 of 3: SAM segments and coalition samples.

EAC (Sun et al., NeurIPS 2023; github.com/Jerry00917/samshap) treats each SAM
segment of an image as a concept, removes segments by setting their pixels to
black, fits a per-image linear surrogate on random segment subsets, and scores
segments by their mean marginal effect on the class probability.

This step picks N_PER_PAIR held-out composites per (object, scene) pair, runs
SamAutomaticMaskGenerator (ViT-H, library defaults, as in EAC) on each image at
the models' 224 px input size, names every segment "object" if at least half
of it lies on BAM's object mask and "scene" otherwise, and draws N_SAMPLES
Bernoulli(0.5) segment subsets per image (EAC's PIE training set).

Needs Meta's segment-anything code on PYTHONPATH and --sam-checkpoint
(sam_vit_h_4b8939.pth). Writes <bam-root>/baselines/eac/segments/<stem>.npz
(masks, names, subsets) and eac_manifest.csv. Next: run_eac_bam_tf.py.
"""

import argparse
import csv
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

N_PER_PAIR = 2
N_SAMPLES = 2500
INPUT_SIZE = (224, 224)
SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam-root", type=Path, default=Path(os.environ.get("BAM_ROOT", "../Datasets/BAM")))
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.bam_root / "baselines" / "eac"
    (out_dir / "segments").mkdir(parents=True, exist_ok=True)

    with open(args.bam_root / "val" / "manifest.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[row["object"], row["scene"]].append(row)
    rng = np.random.default_rng(SEED)
    chosen = [r for pair in sorted(by_pair) for r in rng.choice(by_pair[pair], N_PER_PAIR, replace=False)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["default"](checkpoint=str(args.sam_checkpoint)).to(device)
    generator = SamAutomaticMaskGenerator(sam)
    manifest, start = [], time.time()
    for n, row in enumerate(chosen):
        stem = Path(row["file"]).stem
        image = np.asarray(Image.open(args.bam_root / "val" / "composite" / row["file"]).convert("RGB")
                           .resize(INPUT_SIZE, Image.Resampling.BILINEAR))
        obj = np.asarray(Image.open(args.bam_root / "val" / "object_mask" / f"{stem}.png").convert("L")
                         .resize(INPUT_SIZE, Image.Resampling.NEAREST)) > 127
        masks = np.stack([m["segmentation"] for m in generator.generate(image)])
        on_object = (masks & obj).sum((1, 2)) / masks.sum((1, 2))
        names = np.where(on_object >= 0.5, "object", "scene")
        subsets = np.random.default_rng(SEED + n).random((N_SAMPLES, len(masks))) < 0.5
        np.savez_compressed(out_dir / "segments" / f"{stem}.npz", masks=masks, names=names, subsets=subsets)
        manifest.append({"file": row["file"], "object": row["object"], "scene": row["scene"], "n_segments": len(masks),
                         "n_object": int((names == "object").sum())})
        if (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(chosen)} images in {time.time() - start:.0f}s", flush=True)
    with open(out_dir / "eac_manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"done: {len(manifest)} images, mean {np.mean([m['n_segments'] for m in manifest]):.1f} segments, "
          f"{np.mean([m['n_object'] == 0 for m in manifest]):.0%} with no object segment", flush=True)


if __name__ == "__main__":
    main()
