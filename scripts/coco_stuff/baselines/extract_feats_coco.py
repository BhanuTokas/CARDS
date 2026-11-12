"""Black-box features for the concept-discovery baselines (ConceptSAE, ConceptSHAP) on COCO-Stuff ResNet18.
Saves to $CARDS_FEAT_CACHE (default results/coco_stuff/baselines/feat_cache):
  pooled.pt     : pooled (N,512) fp32 = input of the 20-way fc, logits (N,20), labels (N,n_concepts) bool, paths, concepts,
                  fc_w, fc_b
  map_train40k.pt : (N,512,7,7) fp16 layer4 maps for a fixed random 40k of the pool (+ matching indices)"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from coco_common import DEVICE, FEAT_CACHE, concept_labels, concept_list, load_black_box, pool_paths, resnet_feats


class DS(Dataset):
    def __init__(self, paths, tf): self.paths, self.tf = paths, tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def main():
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    model, tf = load_black_box()
    paths, concepts = pool_paths(), concept_list()
    print(f"{len(paths)} pool images, {len(concepts)} concepts", flush=True)
    labels = concept_labels(paths, concepts)
    print("concept positives per concept: min/median/max", labels.sum(0).min(), int(np.median(labels.sum(0))), labels.sum(0).max(), flush=True)
    keep40k = set(np.random.default_rng(0).choice(len(paths), min(40_000, len(paths)), replace=False).tolist())
    P, L, M, Mi = [], [], [], []
    i0 = 0
    for n, x in enumerate(DataLoader(DS(paths, tf), batch_size=256, num_workers=8)):
        x = x.to(DEVICE); pooled, maps = resnet_feats(model, x); logit = model.fc(pooled)
        if n == 0: assert (logit - model(x)).abs().max() < 1e-3, "feats/head do not reproduce the model"
        P.append(pooled.cpu()); L.append(logit.cpu())
        sel = [j for j in range(len(x)) if i0 + j in keep40k]
        if sel: M.append(maps[sel].half().cpu()); Mi += [i0 + j for j in sel]
        i0 += len(x)
        if n % 50 == 0: print(f"  batch {n}", flush=True)
    torch.save({"pooled": torch.cat(P), "logits": torch.cat(L), "labels": torch.from_numpy(labels), "paths": paths,
                "concepts": concepts, "fc_w": model.fc.weight.detach().cpu(), "fc_b": model.fc.bias.detach().cpu()},
               FEAT_CACHE / "pooled.pt")
    torch.save({"maps": torch.cat(M), "idx": torch.tensor(Mi)}, FEAT_CACHE / "map_train40k.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
