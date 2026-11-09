"""Store BAM model outputs for a list of images. Run in the BAM environment.

BAM's models are TF1 SavedModels and the rest of this repo is PyTorch, so this
step runs in its own environment and everything downstream reads its output.
Preprocessing follows BAM's `attributions.py`.

Always stores logits. Optionally stores the pooled embedding (PCBM's input),
block_layer4 activations (TCAV's layer) and, given a target column, the
gradient of that class's logit with respect to block_layer4 (TCAV's
directional derivatives).
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

CHANNEL_MEANS = np.array([123.68, 116.78, 103.94], dtype=np.float32)
INPUT_SIZE = (224, 224)
N_CLASSES = 10


def load_batch(paths: list) -> np.ndarray:
    arrays = []
    for path in paths:
        with Image.open(path) as handle:
            image = handle.convert("RGB").resize(INPUT_SIZE, Image.Resampling.BILINEAR)
        arrays.append(np.asarray(image, dtype=np.float32) - CHANNEL_MEANS)
    return np.stack(arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="CSV naming images relative to --image-dir.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--file-column", default="file")
    parser.add_argument("--filter", default=None,
                        help="Keep only rows where column=value, e.g. model=obj.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--activations", action="store_true")
    parser.add_argument("--float16", action="store_true", help="Store activations as float16.")
    parser.add_argument("--grad-target-column", default=None)
    args = parser.parse_args()

    with open(args.manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.filter:
        column, value = args.filter.split("=", 1)
        rows = [row for row in rows if row[column] == value]
    names = [row[args.file_column] for row in rows]
    targets = (np.array([int(row[args.grad_target_column]) for row in rows])
               if args.grad_target_column else None)

    keys = ["logits"]
    if args.embeddings:
        keys.append("embeddings")
    if args.activations:
        keys.append("activations")
    if targets is not None:
        keys.append("gradients")
    outputs = {key: [] for key in keys}

    graph = tf.Graph()
    start = time.time()
    with tf.compat.v1.Session(graph=graph) as sess:
        tf.compat.v1.saved_model.loader.load(sess, ["serve"], str(args.model_dir))
        input_tensor = graph.get_tensor_by_name("input_tensor:0")
        block4 = graph.get_tensor_by_name("resnet_model/block_layer4:0")
        tensors = {"logits": graph.get_tensor_by_name("resnet_model/final_dense:0")}
        if args.embeddings:
            tensors["embeddings"] = graph.get_tensor_by_name("resnet_model/Squeeze:0")
        if args.activations:
            tensors["activations"] = block4
        if targets is not None:
            with graph.as_default():
                onehot = tf.compat.v1.placeholder(tf.float32, [None, N_CLASSES])
                # Each image's gradient depends only on its own logit, because the
                # serving graph uses fixed batch-norm statistics.
                selected = tf.reduce_sum(tensors["logits"] * onehot)
                tensors["gradients"] = tf.gradients(selected, block4)[0]

        for begin in range(0, len(names), args.batch_size):
            stop = min(begin + args.batch_size, len(names))
            feed = {input_tensor: load_batch([args.image_dir / n for n in names[begin:stop]])}
            if targets is not None:
                feed[onehot] = np.eye(N_CLASSES, dtype=np.float32)[targets[begin:stop]]
            values = sess.run(tensors, feed)
            for key in keys:
                outputs[key].append(values[key])
            if (stop // args.batch_size) % 50 == 0 or stop == len(names):
                print(f"{stop}/{len(names)} in {time.time() - start:.0f}s", flush=True)

    arrays = {key: np.concatenate(value) for key, value in outputs.items()}
    for key in ("activations", "gradients"):
        if key in arrays:
            arrays[key] = arrays[key].reshape(len(names), -1)
    if args.float16 and "activations" in arrays:
        arrays["activations"] = arrays["activations"].astype(np.float16)
    if targets is not None:
        arrays["targets"] = targets
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, files=np.array(names), **arrays)


if __name__ == "__main__":
    main()
