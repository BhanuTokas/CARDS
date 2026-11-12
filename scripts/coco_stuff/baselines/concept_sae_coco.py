"""ConceptSAE baseline on COCO-Stuff ResNet18 (port of scripts/celeba/baselines/concept_sae.py). argv[1] = out CSV.
Design copied from CelebA, fixed before seeing any ground-truth correlation:
  - TopK SAE (Gao et al. 2024) on the 512-d avgpool features of every pool image; 4096 latents, k=32, unit-norm decoder
    columns, Adam lr 4e-4, batch 4096, 60 epochs, seed 0.
  - Naming: each concept gets the latent with the highest AUROC against its image-level label (supervised naming,
    favours the baseline; labels = Grounding DINO detection, the same images the ground truth is measured on).
  - Score(concept, class) = mean over concept-positive images of z_j(x) * (w_class . d_j): the exact contribution of that
    latent to the class logit through the linear head. Variant column weight_score = w_class . d_j alone.
Needs extract_feats_coco.py first. Output columns: concept_name, class_name, latent, train_auroc, n_pos, score, weight_score."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
from coco_common import CLASSES, DEVICE, FEAT_CACHE

N_LAT, K, LR, BS, EPOCHS = 4096, 32, 4e-4, 4096, 60
torch.manual_seed(0)


class TopKSAE(torch.nn.Module):
    def __init__(self, d, n, k, init_bias):
        super().__init__()
        dec = torch.randn(d, n); dec /= dec.norm(dim=0, keepdim=True)
        self.W_dec = torch.nn.Parameter(dec); self.W_enc = torch.nn.Parameter(dec.T.clone())
        self.b_enc = torch.nn.Parameter(torch.zeros(n)); self.b_dec = torch.nn.Parameter(init_bias.clone()); self.k = k

    def encode(self, x):
        pre = torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)
        v, i = pre.topk(self.k, dim=1)
        return torch.zeros_like(pre).scatter_(1, i, v)

    def forward(self, x):
        z = self.encode(x)
        return z @ self.W_dec.T + self.b_dec, z


def main():
    out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)
    d = torch.load(FEAT_CACHE / "pooled.pt", weights_only=False)
    X, W = d["pooled"].to(DEVICE), d["fc_w"].to(DEVICE)
    Y = d["labels"].numpy().astype(float); concepts = d["concepts"]
    sae = TopKSAE(X.shape[1], N_LAT, K, X.median(0).values).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(X), device=DEVICE); tot = 0
        for s in range(0, len(X), BS):
            x = X[perm[s:s + BS]]; xh, _ = sae(x)
            loss = ((xh - x) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): sae.W_dec /= sae.W_dec.norm(dim=0, keepdim=True)
            tot += loss.item() * len(x)
        if ep % 10 == 0 or ep == EPOCHS - 1: print(f"epoch {ep} mse {tot / len(X):.4f}", flush=True)
    with torch.no_grad():
        xh, _ = sae(X[:20000])
        fvu = (((xh - X[:20000]) ** 2).sum() / ((X[:20000] - X[:20000].mean(0)) ** 2).sum()).item()
        agree = ((X[:20000] @ W.T + d["fc_b"].to(DEVICE) > 0) == (xh @ W.T + d["fc_b"].to(DEVICE) > 0)).float().mean().item()
        print(f"FVU {fvu:.3f}; per-class prediction agreement through SAE {agree:.4f}", flush=True)
        Z = torch.cat([sae.encode(X[s:s + 8192]).cpu() for s in range(0, len(X), 8192)]).numpy()
    npos = Y.sum(0); nneg = len(Y) - npos
    auc = np.zeros((N_LAT, Y.shape[1]))
    for s in range(0, N_LAT, 256):
        R = rankdata(Z[:, s:s + 256], axis=0)  # average ranks: handles the many exact zeros
        auc[s:s + 256] = (R.T @ Y - npos * (npos + 1) / 2) / (npos * nneg)
    pos = Y.astype(bool)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["concept_name", "class_name", "latent", "train_auroc", "n_pos", "score", "weight_score"])
        for n, c in enumerate(concepts):
            j = int(auc[:, n].argmax()); dj = sae.W_dec[:, j].detach(); zbar = float(Z[pos[:, n], j].mean())
            for k, cls in enumerate(CLASSES):
                wd = float(W[k] @ dj)
                w.writerow([c, cls, j, f"{auc[j, n]:.4f}", int(pos[:, n].sum()), zbar * wd, wd])
            print(f"  {c:22s} latent {j:4d} AUROC {auc[j, n]:.3f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
