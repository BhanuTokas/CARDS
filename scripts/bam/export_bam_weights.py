"""Export BAM's obj and scene SavedModel weights to npz, for the PyTorch port in bam_torch_model.py.

Run in the BAM environment. Writes <bam-root>/models/<model>_weights.npz (every
resnet_model/ variable, keyed by its TF name).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam-root", type=Path, default=Path(os.environ.get("BAM_ROOT", "../Datasets/BAM")))
    args = parser.parse_args()
    for model in ("obj", "scene"):
        graph = tf.Graph()
        with tf.compat.v1.Session(graph=graph) as sess:
            tf.compat.v1.saved_model.loader.load(sess, ["serve"], str(args.bam_root / "models" / model))
            variables = [v for v in tf.compat.v1.global_variables() if v.name.startswith("resnet_model/")]
            values = sess.run(variables)
        path = args.bam_root / "models" / f"{model}_weights.npz"
        np.savez(path, **{v.name[:-2]: x for v, x in zip(variables, values)})
        print(f"{model}: {len(variables)} variables to {path}")


if __name__ == "__main__":
    main()
