"""Explain Any Concept (EAC) on BAM, step 2 of 3: model outputs on segment subsets. Run in the BAM environment.

For every image from run_eac_bam_sam.py and each of its N_SAMPLES segment
subsets, keep the pixels of the selected segments and set every other pixel to
black in raw pixel space (as EAC does), apply BAM's preprocessing, and store
the model's softmax over its 10 classes. Also stores the unmasked image's
probabilities.

The step-1 manifest lists two images per (object, scene) pair, consecutively;
--part first takes one of each pair, --part second the other.

Writes <bam-root>/baselines/eac/tf/<model>/<stem>.npz (probs, original).
Next: fit_eac_bam.py.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

CHANNEL_MEANS = np.array([123.68, 116.78, 103.94], dtype=np.float32)
INPUT_SIZE = (224, 224)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam-root", type=Path, default=Path(os.environ.get("BAM_ROOT", "../Datasets/BAM")))
    parser.add_argument("--model", choices=["obj", "scene"], required=True)
    parser.add_argument("--batch-size", type=int, default=125)
    parser.add_argument("--part", choices=["first", "second", "all"], default="all",
                        help="first: one image per (object, scene) pair; second: the rest")
    args = parser.parse_args()
    eac = args.bam_root / "baselines" / "eac"
    out_dir = eac / "tf" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(eac / "eac_manifest.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = {"first": rows[::2], "second": rows[1::2], "all": rows}[args.part]

    graph = tf.Graph()
    start = time.time()
    with tf.compat.v1.Session(graph=graph) as sess:
        tf.compat.v1.saved_model.loader.load(sess, ["serve"], str(args.bam_root / "models" / args.model))
        input_tensor = graph.get_tensor_by_name("input_tensor:0")
        logits = graph.get_tensor_by_name("resnet_model/final_dense:0")
        for n, row in enumerate(rows):
            stem = Path(row["file"]).stem
            if (out_dir / f"{stem}.npz").exists():
                continue
            seg = np.load(eac / "segments" / f"{stem}.npz")
            masks, subsets = seg["masks"], seg["subsets"]
            image = np.asarray(Image.open(args.bam_root / "val" / "composite" / row["file"]).convert("RGB")
                               .resize(INPUT_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)
            original = softmax(sess.run(logits, {input_tensor: (image - CHANNEL_MEANS)[None]}))[0]
            flat = masks.reshape(len(masks), -1).astype(np.float32)
            probs = []
            for b in range(0, len(subsets), args.batch_size):
                keep = (subsets[b:b + args.batch_size].astype(np.float32) @ flat > 0).reshape(-1, *INPUT_SIZE, 1)
                batch = image[None] * keep - CHANNEL_MEANS
                probs.append(softmax(sess.run(logits, {input_tensor: batch})))
            np.savez(out_dir / f"{stem}.npz", probs=np.concatenate(probs).astype(np.float32), original=original)
            if (n + 1) % 10 == 0:
                print(f"{n + 1}/{len(rows)} images in {time.time() - start:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
