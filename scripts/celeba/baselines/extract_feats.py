# ruff: noqa: C408, SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""Black-box features for the concept-discovery baselines (official-train Attractive/Male classifier; env CARDS_ARCH
selects resnet18 / vit / convnext, see black_boxes.py). Run from the repo root.
Saves to $CARDS_FEAT_CACHE (default results/baselines/feat_cache[_vit|_convnext]):
  pooled_{train,val}.pt : (N,d) fp32 input to the final Linear, plus logits, attribute labels, file names
  map_{train40k,val}.pt : (N,d,7,7) fp16 last-stage maps (train40k = fixed random 40k of official train)
val = the clean official val pool used by CARDS (official val minus every CelebAMask-HQ image)."""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "src"); sys.path.insert(0, "scripts/celeba/ground_truth"); sys.path.insert(0, "scripts/celeba/run")
from run_cards_celeba_masking_hybrid_official_val_zscore import build_clean_official_val_paths
from run_celeba_official_train_faithfulness import (
    CELEBA_ROOT,
    DEVICE,
    official_train_preprocess,
)

sys.path.insert(0, str(Path(__file__).parent))
from black_boxes import ARCH, FEAT_CACHE, feats, head, load

OUT = FEAT_CACHE
IMG = CELEBA_ROOT / "img_align_celeba"


class DS(Dataset):
    def __init__(self, files): self.files, self.t = files, official_train_preprocess()
    def __len__(self): return len(self.files)
    def __getitem__(self, i): return self.t(Image.open(IMG / self.files[i]).convert("RGB"))


def labels():
    lines = open(CELEBA_ROOT / "list_attr_celeba.txt").read().splitlines()
    names = lines[1].split()
    lab = {l.split()[0]: np.array([int(v) > 0 for v in l.split()[1:]]) for l in lines[2:]}
    return names, lab


@torch.no_grad()
def run(model, files, want_map):
    P, L, Mp = [], [], []
    for n, x in enumerate(DataLoader(DS(files), batch_size=256, num_workers=8)):
        x = x.to(DEVICE); pooled, m = feats(model, x); logit = head(model)(pooled)
        if n == 0: assert (logit - model(x)).abs().max() < 1e-3, "feats/head do not reproduce the model"
        P.append(pooled.cpu()); L.append(logit.cpu())
        if want_map: Mp.append(m.half().cpu())
        if n % 100 == 0: print(f"   batch {n}", flush=True)
    return torch.cat(P), torch.cat(L), (torch.cat(Mp) if want_map else None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names, lab = labels()
    part = dict(l.split() for l in open(CELEBA_ROOT / "list_eval_partition.txt"))
    train = sorted(f for f, p in part.items() if p == "0")
    val = sorted(p.name for p in build_clean_official_val_paths())
    print("train", len(train), "clean val", len(val), flush=True)
    model = load().to(DEVICE).eval()
    print("black box:", ARCH, flush=True)
    W, b = head(model).weight.detach().cpu(), head(model).bias.detach().cpu()
    for tag, files, want_map in (("val", val, True), ("train", train, False)):
        P, L, Mp = run(model, files, want_map)
        err = (P @ W.T + b - L).abs().max().item()
        print(f"{tag}: pooled->fc reproduces logits, max err {err:.2e}", flush=True)
        assert err < 1e-3
        Y = torch.from_numpy(np.stack([lab[f] for f in files]))
        torch.save(dict(files=files, pooled=P, logits=L, labels=Y, attr_names=names, fc_w=W, fc_b=b), OUT / f"pooled_{tag}.pt")
        if want_map: torch.save(dict(files=files, maps=Mp), OUT / f"map_{tag}.pt")
    sub = sorted(np.random.default_rng(0).choice(len(train), 40_000, replace=False).tolist())
    files = [train[i] for i in sub]
    P, L, Mp = run(model, files, True)
    torch.save(dict(files=files, maps=Mp, logits=L, labels=torch.from_numpy(np.stack([lab[f] for f in files]))), OUT / "map_train40k.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
