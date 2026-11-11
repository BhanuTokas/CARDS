"""Explain Any Concept (EAC) on BAM, step 2 of 3, on the GPU.

Produces the same outputs as run_eac_bam_tf.py, using the PyTorch port of the
models in bam_torch_model.py; class probabilities match the TF model to about
1e-5. Run in the repo environment after export_bam_weights.py. Writes
<bam-root>/baselines/eac/tf/<model>/<stem>.npz (probs, original) and skips
existing files, so it can finish a run that run_eac_bam_tf.py started.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from bam_torch_model import CHANNEL_MEANS, load

INPUT_SIZE = (224, 224)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam-root", type=Path, default=Path(os.environ.get("BAM_ROOT", "../Datasets/BAM")))
    parser.add_argument("--model", choices=["obj", "scene"], required=True)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--part", choices=["first", "second", "all"], default="all",
                        help="first: one image per (object, scene) pair; second: the rest")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eac = args.bam_root / "baselines" / "eac"
    out_dir = eac / "tf" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(eac / "eac_manifest.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = {"first": rows[::2], "second": rows[1::2], "all": rows}[args.part]

    net = load(args.bam_root, args.model).to(device)
    mean = torch.tensor(CHANNEL_MEANS, device=device).view(1, 3, 1, 1)
    start = time.time()
    with torch.no_grad():
        for n, row in enumerate(rows):
            stem = Path(row["file"]).stem
            if (out_dir / f"{stem}.npz").exists():
                continue
            seg = np.load(eac / "segments" / f"{stem}.npz")
            image = np.asarray(Image.open(args.bam_root / "val" / "composite" / row["file"]).convert("RGB")
                               .resize(INPUT_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)
            image = torch.from_numpy(image).permute(2, 0, 1)[None].to(device)
            original = torch.softmax(net(image - mean), 1)[0].cpu().numpy()
            flat = torch.from_numpy(seg["masks"].reshape(len(seg["masks"]), -1).astype(np.float32)).to(device)
            subsets = torch.from_numpy(seg["subsets"].astype(np.float32)).to(device)
            probs = []
            for b in range(0, len(subsets), args.batch_size):
                keep = (subsets[b:b + args.batch_size] @ flat > 0).float().view(-1, 1, *INPUT_SIZE)
                probs.append(torch.softmax(net(image * keep - mean), 1).cpu().numpy())
            np.savez(out_dir / f"{stem}.npz", probs=np.concatenate(probs).astype(np.float32), original=original)
            if (n + 1) % 10 == 0:
                print(f"{n + 1}/{len(rows)} images in {time.time() - start:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
