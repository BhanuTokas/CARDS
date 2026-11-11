# ruff: noqa: C408, SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""ResNet18 (official-train Attractive/Male) features for the concept-discovery baselines. Run from the repo root.
Saves to $CARDS_FEAT_CACHE (default results/inpainting/feat_cache):
  pooled_{train,val}.pt : (N,512) fp32 avgpool features (input to fc), plus logits, attribute labels, file names
  map_{train40k,val}.pt : (N,512,7,7) fp16 layer4 maps (train40k = fixed random 40k of official train)
val = the clean official val pool used by CARDS (official val minus every CelebAMask-HQ image)."""
import os
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
    load_official_train_native,
    official_train_preprocess,
)

OUT = Path(os.environ.get("CARDS_FEAT_CACHE", "results/inpainting/feat_cache"))
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
    fmap = {}
    h = model.layer4.register_forward_hook(lambda m, i, o: fmap.__setitem__("x", o))
    P, L, Mp = [], [], []
    for n, x in enumerate(DataLoader(DS(files), batch_size=256, num_workers=8)):
        logit = model(x.to(DEVICE))
        m = fmap["x"]; pooled = m.mean((2, 3))
        P.append(pooled.cpu()); L.append(logit.cpu())
        if want_map: Mp.append(m.half().cpu())
        if n % 100 == 0: print(f"   batch {n}", flush=True)
    h.remove()
    return torch.cat(P), torch.cat(L), (torch.cat(Mp) if want_map else None)


def main():
    OUT.mkdir(exist_ok=True)
    names, lab = labels()
    part = dict(l.split() for l in open(CELEBA_ROOT / "list_eval_partition.txt"))
    train = sorted(f for f, p in part.items() if p == "0")
    val = sorted(p.name for p in build_clean_official_val_paths())
    print("train", len(train), "clean val", len(val), flush=True)
    model = load_official_train_native().to(DEVICE).eval()
    W, b = model.fc.weight.detach().cpu(), model.fc.bias.detach().cpu()
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
