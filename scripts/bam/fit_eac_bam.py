"""Explain Any Concept (EAC) on BAM, step 3 of 3: per-image surrogate and segment values.

Follows EAC's code (github.com/Jerry00917/samshap, sam_explainer.py):
  - PIE surrogate: a single linear layer from the binary segment-presence vector
    to the model's pooled feature (2048-d), followed by the model's own final
    layer, kept frozen. Trained with BCE between the surrogate's and the
    model's probability of the explained class, on the N_SAMPLES subsets from
    step 1 (SGD, lr 0.008, momentum 0.9, batch 100, 50 epochs, torch seed 0).
  - Segment value: draw MC Bernoulli(0.5) coalitions; the value of segment i is
    the mean of P(class | coalition with i) - P(class | coalition without i)
    under the surrogate. (With uniform coalitions this is a Banzhaf value, as
    in EAC's code, not a size-weighted Shapley value.)
Two changes, both needed for this benchmark: EAC explains only the predicted
class, while the benchmark needs every class, so one surrogate is fitted per
class (they share the model calls); and EAC's training loop skips the first
n_segments samples, which looks like a bug and is not copied.

Segments are named softly: a segment adds its value to the image's object
concept in proportion to the share of its pixels on BAM's object mask, and the
rest to the image's scene concept (pixels in no segment are always black, as
in EAC, and carry no value). S[model, class, concept] is the mean of that
concept value over the images containing the concept. The model's final layer is recovered by least squares from
stored embeddings and logits (tf/<model>_train.npz).
Writes results/bam_eac_scores.csv (model, class, concept, n_images, score).
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from fit_concept_sae_bam import linear_head  # noqa: E402
from PIL import Image  # noqa: E402
from sample_bam_concept_sets import OBJECTS, SCENES  # noqa: E402

CLASSES = {"obj": OBJECTS, "scene": SCENES}
EPOCHS, BATCH, LR, MOMENTUM = 50, 100, 0.008, 0.9
MC = 50_000


def fit_values(subsets: np.ndarray, probs: np.ndarray, w: torch.Tensor, b: torch.Tensor, device: str) -> np.ndarray:
    """Segment values (n_classes, n_segments): one PIE surrogate per class, trained jointly but independently."""
    n_seg, n_cls, d = subsets.shape[1], w.shape[0], w.shape[1]
    torch.manual_seed(0)
    layer = torch.nn.Linear(n_seg, d)  # EAC's init, copied for each class
    weight = layer.weight.detach().T[None].repeat(n_cls, 1, 1).to(device).requires_grad_()  # (C, n, d)
    bias = layer.bias.detach()[None].repeat(n_cls, 1).to(device).requires_grad_()  # (C, d)
    opt = torch.optim.SGD([weight, bias], lr=LR, momentum=MOMENTUM)
    x = torch.from_numpy(subsets.astype(np.float32)).to(device)
    y = torch.from_numpy(probs).to(device)  # (N, C)
    eye = torch.arange(n_cls, device=device)

    def surrogate(batch: torch.Tensor) -> torch.Tensor:
        """P(class c) under class c's surrogate, for every c: (B, C)."""
        h = torch.einsum("bn,cnd->cbd", batch, weight) + bias[:, None]  # (C, B, d)
        p = torch.softmax(h @ w.T + b, dim=-1)  # (C, B, C)
        return p[eye, :, eye].T

    gen = torch.Generator().manual_seed(0)
    for _ in range(EPOCHS):
        order = torch.randperm(len(x), generator=gen)
        for s in range(0, len(x), BATCH):
            idx = order[s:s + BATCH].to(device)
            pred = surrogate(x[idx]).clamp(1e-6, 1 - 1e-6)
            loss = torch.nn.functional.binary_cross_entropy(pred, y[idx], reduction="sum") / len(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()

    values = torch.zeros(n_cls, n_seg, device=device)
    rng = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for i in range(n_seg):
            total = torch.zeros(n_cls, device=device)
            for s in range(0, MC, 2_000):
                coal = (torch.rand(2_000, n_seg, generator=rng, device=device) < 0.5).float()
                coal[:, i] = 1
                with_i = surrogate(coal)
                coal[:, i] = 0
                total += (with_i - surrogate(coal)).sum(0)
            values[:, i] = total / MC
    return values.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam-root", type=Path, default=Path(os.environ.get("BAM_ROOT", "../Datasets/BAM")))
    parser.add_argument("--results-dir", type=Path, default=Path(os.environ.get("CARDS_RESULTS_DIR", "results")))
    parser.add_argument("--part", choices=["first", "all"], default="all",
                        help="first: only one image per (object, scene) pair (half the images)")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eac = args.bam_root / "baselines" / "eac"
    with open(eac / "eac_manifest.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.part == "first":
        rows = rows[::2]

    out = []
    for model, classes in CLASSES.items():
        tr = np.load(args.bam_root / "baselines" / "tf" / f"{model}_train.npz")
        w, b, err = linear_head(tr["embeddings"], tr["logits"])
        print(f"{model}: final layer recovered, max logit error {err:.1e}", flush=True)
        w, b = torch.from_numpy(w).to(device), torch.from_numpy(b).to(device)
        per = defaultdict(list)
        fidelity, start = [], time.time()
        for n, row in enumerate(rows):
            stem = Path(row["file"]).stem
            seg = np.load(eac / "segments" / f"{stem}.npz")
            outs = np.load(eac / "tf" / model / f"{stem}.npz")
            values = fit_values(seg["subsets"], outs["probs"], w, b, device)  # (C, n)
            masks = seg["masks"]
            obj = np.asarray(Image.open(args.bam_root / "val" / "object_mask" / f"{stem}.png").convert("L")
                             .resize(masks.shape[1:][::-1], Image.Resampling.NEAREST)) > 127
            frac = (masks & obj).sum((1, 2)) / masks.sum((1, 2))
            per[row["object"]].append(values @ frac)
            per[row["scene"]].append(values @ (1 - frac))
            fidelity.append(values.sum(axis=1)[int(outs["original"].argmax())])
            if (n + 1) % 25 == 0:
                print(f"  {model} {n + 1}/{len(rows)} images in {time.time() - start:.0f}s", flush=True)
        for concept, vals in per.items():
            v = np.mean(vals, axis=0)
            for k, cls in enumerate(classes):
                out.append({"model": model, "class": cls, "concept": concept, "n_images": len(vals), "score": float(v[k])})
        print(f"{model}: mean summed value for the predicted class {np.mean(fidelity):+.3f}", flush=True)

    args.results_dir.mkdir(exist_ok=True)
    path = args.results_dir / "bam_eac_scores.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"saved {len(out)} rows to {path}")


if __name__ == "__main__":
    main()
