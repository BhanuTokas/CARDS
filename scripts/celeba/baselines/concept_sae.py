
"""Concept SAE baseline (Discover-then-Name style, adapted to the ResNet18 black box). argv[1]=out CSV.
Pre-set design (fixed before seeing any GT correlation):
  - TopK SAE (Gao et al. 2024) on the 512-d avgpool features of all 162,770 official-train images; 4096 latents, k=32,
    unit-norm decoder columns, Adam lr 4e-4, batch 4096, 60 epochs, seed 0.
  - Naming: DN-CBM names latents with CLIP text, impossible for a ResNet18 feature space. We name each of the 26 concepts by the
    latent with the highest AUROC against the CelebA attribute label on the TRAIN set (supervised naming; favours the baseline).
  - Score(task, concept) = mean over clean-val images labelled with the concept of z_j(x) * (w_task . d_j): the exact
    contribution of that latent to the positive-class logit through the linear head. Variant: w_task . d_j alone."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, "src")
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS

sys.path.insert(0, str(Path(__file__).parent))
from black_boxes import FEAT_CACHE

F = FEAT_CACHE
OUT = Path(sys.argv[1]); DEV = "cuda"
N_LAT, K, LR, BS, EPOCHS = 4096, 32, 4e-4, 4096, 60
POS = {"Attractive": 1, "Male": 3}
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
    tr, va = torch.load(F / "pooled_train.pt"), torch.load(F / "pooled_val.pt")
    X, Xv = tr["pooled"].to(DEV), va["pooled"].to(DEV)
    W, b = tr["fc_w"].to(DEV), tr["fc_b"].to(DEV)
    names = tr["attr_names"]; ci = [names.index(c) for c in GROUNDABLE_CONCEPTS]
    sae = TopKSAE(X.shape[1], N_LAT, K, X.median(0).values).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(X), device=DEV); tot = 0
        for s in range(0, len(X), BS):
            x = X[perm[s:s + BS]]; xh, _ = sae(x)
            loss = ((xh - x) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): sae.W_dec /= sae.W_dec.norm(dim=0, keepdim=True)
            tot += loss.item() * len(x)
        if ep % 10 == 0 or ep == EPOCHS - 1: print(f"epoch {ep} mse {tot / len(X):.4f}", flush=True)
    with torch.no_grad():
        xh, zv = sae(Xv)
        fvu = (((xh - Xv) ** 2).sum() / ((Xv - Xv.mean(0)) ** 2).sum()).item()
        L, Lh = Xv @ W.T + b, xh @ W.T + b
        agree = {t: (L[:, p - 1:p + 1].argmax(1) == Lh[:, p - 1:p + 1].argmax(1)).float().mean().item() for t, p in POS.items()}
        dead = (zv.sum(0) == 0).float().mean().item()
        print(f"val FVU {fvu:.3f}; prediction agreement through SAE {agree}; latents dead on val {dead:.2f}", flush=True)
        Z = torch.cat([sae.encode(X[s:s + 8192]).cpu() for s in range(0, len(X), 8192)]).numpy()
    Y = tr["labels"][:, ci].numpy().astype(float); npos = Y.sum(0); nneg = len(Y) - npos
    auc = np.zeros((N_LAT, len(ci)))
    for s in range(0, N_LAT, 256):
        R = rankdata(Z[:, s:s + 256], axis=0)  # average ranks: handles the many exact zeros
        auc[s:s + 256] = (R.T @ Y - npos * (npos + 1) / 2) / (npos * nneg)
    del Z
    Yv = va["labels"][:, ci].numpy().astype(bool); zv = zv.cpu().numpy()
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "concept_name", "latent", "train_auroc", "n_val_pos", "score", "weight_score"])
        for n, c in enumerate(GROUNDABLE_CONCEPTS):
            j = int(auc[:, n].argmax()); d = sae.W_dec[:, j].detach()
            for t, p in POS.items():
                wd = float(W[p] @ d)
                w.writerow([t, c, j, f"{auc[j, n]:.4f}", int(Yv[:, n].sum()), float(zv[Yv[:, n], j].mean() * wd), wd])
            print(f"  {c:20s} latent {j:4d} AUROC {auc[j, n]:.3f}", flush=True)
    latents = {c: int(auc[:, n].argmax()) for n, c in enumerate(GROUNDABLE_CONCEPTS)}
    torch.save({"state": sae.state_dict(), "k": K, "latents": latents}, OUT.with_suffix(".pt"))  # for local_sae.py
    print("done", flush=True)


if __name__ == "__main__":
    main()
