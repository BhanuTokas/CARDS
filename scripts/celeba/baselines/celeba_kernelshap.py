# ruff: noqa: SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""KernelSHAP over SAM segments on CelebA: a fully black-box counterpart of EAC. Copy of
the CelebA EAC script (celeba_eac.py; same procedure as the BAM EAC run) with ONLY the value estimator changed: EAC's PIE surrogate (which uses the
model's frozen fc) + MC Shapley is replaced by KernelSHAP on the model's outputs (fit_values / kernel_subsets below).
Same images, SAM settings, black fill, 2,500 model queries per image, probability targets and GT-region crediting.

Original EAC docstring follows.
Explain Any Concept (EAC) on CelebA (official-train ResNet18, Attractive and Male). Run from a CARDS checkout's root
with Meta's segment-anything on PYTHONPATH. Same EAC procedure as scripts/bam/fit_eac_bam.py (SAM ViT-H defaults at the
model's 224 px input, black fill in raw pixel space, PIE surrogate = Linear(n_segments, 512) + the model's frozen fc,
BCE on the task probability, SGD lr .008 / momentum .9 / batch 100 / 50 epochs, 2500 Bernoulli(.5) subsets, 50k MC
coalitions). Targets: P(Attractive) and P(Male), each a softmax over that task's two logits in the 4-way head, with
one surrogate per task.

Images: N_PER_CONCEPT attribute-positive images per concept from the clean CelebA-HQ pool (official val/test), which
has face-part masks. A segment adds its value to a concept in proportion to the share of its pixels inside that
concept's CelebAMask-HQ regions (ATTRIBUTE_TO_REGIONS). Score (task, concept) = mean of that value over the chosen
images where the attribute is present.
argv[1] = SAM checkpoint, argv[2] = output CSV (task, concept_name, n_images, score). env EAC_LIMIT=n images (smoke).
"""
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "src"); sys.path.insert(0, "scripts/celeba/run")
from run_cards_celeba_masking_hybrid_official_train import CELEBA_ROOT
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from cards.data.celeba import load_celebamask_hq_image_paths, load_celebamask_hq_mask
from cards.data.celeba_attributes import ATTRIBUTE_TO_REGIONS, GROUNDABLE_CONCEPTS
from cards.models.backbones import BACKBONES

sys.path.insert(0, "scripts/celeba/ground_truth")
from run_celeba_official_train_faithfulness import CELEBA_HQ_ROOT as HQ

N_PER_CONCEPT, N_SAMPLES, MC, SEED = 20, 2500, 50_000, 0
EPOCHS, BATCH, LR, MOMENTUM = 50, 100, 0.008, 0.9
PAIRS = {"Attractive": (0, 1), "Male": (2, 3)}  # (negative, positive) logits per task
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DEV = "cuda"


def labels_for_hq():
    names = (CELEBA_ROOT / "list_attr_celeba.txt").read_text().splitlines()[1].split()
    attrs = {}
    for line in (CELEBA_ROOT / "list_attr_celeba.txt").read_text().splitlines()[2:]:
        p = line.split()
        if p:
            attrs[p[0]] = {n: v == "1" for n, v in zip(names, p[1:])}
    part = dict(line.split() for line in open(CELEBA_ROOT / "list_eval_partition.txt"))
    out = {}
    with open(HQ / "CelebA-HQ-to-CelebA-mapping.txt") as f:
        next(f)
        for line in f:
            p = line.split()
            if part.get(p[2]) in ("1", "2"):
                out[int(p[0])] = attrs[p[2]]
    return out


def task_probs(model, fc_in_hook, batch):
    """(P(Attractive), P(Male)) per image, from model logits."""
    logits = model(batch)
    return torch.stack([torch.softmax(logits[:, list(PAIRS[t])], 1)[:, 1] for t in PAIRS], 1)


def kernel_subsets(n_seg, n_samples, rng):
    """KernelSHAP sampling (Lundberg & Lee 2017, as in the shap library): coalition size s ~ (M-1)/(s(M-s)) on 1..M-1,
    then a uniform subset of that size; each sample is paired with its complement. Row 0 = empty, row 1 = full."""
    M = n_seg
    rows = [np.zeros(M, bool), np.ones(M, bool)]
    if M > 1:
        s = np.arange(1, M); p = (M - 1) / (s * (M - s)); p /= p.sum()
        for k in rng.choice(s, (n_samples - 2) // 2, p=p):
            z = np.zeros(M, bool); z[rng.choice(M, k, replace=False)] = True
            rows += [z, ~z]
    return np.stack(rows)


def fit_values(subsets, probs):
    """KernelSHAP values (2 tasks, n_segments): least squares on kernel-sampled coalitions (equal weights, since they
    are drawn from the Shapley kernel), with the efficiency constraint sum(phi) = f(full) - f(empty) imposed exactly
    by eliminating the last segment. Uses model outputs only."""
    z = subsets[2:].astype(np.float64); f0, f1 = probs[0].numpy(), probs[1].numpy(); y = probs[2:].numpy() - f0
    M = z.shape[1]; out = np.zeros((2, M))
    for t in range(2):
        tot = f1[t] - f0[t]
        if M == 1: out[t, 0] = tot; continue
        X = z[:, :-1] - z[:, -1:]; yy = y[:, t] - z[:, -1] * tot
        A = X.T @ X + 1e-6 * np.eye(M - 1)  # tiny ridge for numerical stability only
        phi = np.linalg.solve(A, X.T @ yy)
        out[t, :-1] = phi; out[t, -1] = tot - phi.sum()
    return out


@torch.no_grad()
def model_probs(model, raw, masks, subsets):
    img = torch.from_numpy(raw).permute(2, 0, 1)[None].float().to(DEV) / 255
    flat = torch.from_numpy(masks.reshape(len(masks), -1)).float().to(DEV)
    out = []
    for s in range(0, len(subsets), 250):
        keep = (torch.from_numpy(subsets[s:s + 250].astype(np.float32)).to(DEV) @ flat > 0).float().view(-1, 1, *raw.shape[:2])
        x = (img * keep - MEAN.to(DEV)) / STD.to(DEV)
        out.append(task_probs(model, None, x))
    return torch.cat(out).cpu()


def main():
    ckpt, out_path = sys.argv[1], sys.argv[2]
    limit = int(os.environ.get("EAC_LIMIT", "0"))
    labels = labels_for_hq(); paths = load_celebamask_hq_image_paths(HQ)
    rng = np.random.default_rng(SEED)
    chosen = set()
    for c in GROUNDABLE_CONCEPTS:
        pos = sorted(i for i, a in labels.items() if a[c] and i in paths)
        chosen.update(rng.choice(pos, min(N_PER_CONCEPT, len(pos)), replace=False).tolist())
    chosen = sorted(chosen)[:limit] if limit else sorted(chosen)
    print(f"{len(chosen)} images", flush=True)
    spec = BACKBONES["celeba_official_train_attractive_male"]
    model = spec.load_native().to(DEV).eval()
    gen = SamAutomaticMaskGenerator(sam_model_registry["default"](checkpoint=ckpt).to(DEV))
    per = defaultdict(list); start = time.time()
    img_path = Path(out_path + ".images.csv")  # per-image values, so the run can stop and resume
    done = set()
    if img_path.exists():
        for r in csv.DictReader(open(img_path)):
            done.add(int(r["hq"])); per[r["task"], r["concept_name"]].append(float(r["value"]))
    new = not img_path.exists()
    img_f = open(img_path, "a", newline=""); img_w = csv.writer(img_f)
    if new:
        img_w.writerow(["hq", "task", "concept_name", "value"])
    print(f"resuming with {len(done)} images done", flush=True)
    for n, hq in enumerate(chosen):
        if hq in done:
            continue
        raw = np.asarray(Image.open(paths[hq]).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
        masks = np.stack([m["segmentation"] for m in gen.generate(raw)])
        subsets = kernel_subsets(len(masks), N_SAMPLES, np.random.default_rng(SEED + hq))
        values = fit_values(subsets, model_probs(model, raw, masks, subsets))  # (2, n); outputs only
        area = masks.sum((1, 2))
        for c in GROUNDABLE_CONCEPTS:
            if not labels[hq][c]:
                continue
            region = load_celebamask_hq_mask(HQ, hq, ATTRIBUTE_TO_REGIONS[c], target_hw=(224, 224))
            frac = (masks & region).sum((1, 2)) / area
            for i, t in enumerate(PAIRS):
                per[t, c].append(float(values[i] @ frac))
                img_w.writerow([hq, t, c, float(values[i] @ frac)])
        img_f.flush()
        if (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(chosen)} images in {time.time() - start:.0f}s", flush=True)
    with open(out_path, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["task", "concept_name", "n_images", "score"])
        for (t, c), v in sorted(per.items()):
            wr.writerow([t, c, len(v), np.mean(v)])
    print("done", flush=True)


if __name__ == "__main__":
    main()
