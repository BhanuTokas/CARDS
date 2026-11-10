
"""ConceptSHAP baseline (Yeh et al., NeurIPS 2020), on the ResNet18 black box's layer4 maps (7x7x512). argv[1]=out CSV.
Pre-set design (fixed before seeing any GT correlation):
  - m=40 concept vectors on layer4 patch features; concept score per patch = TH(cos(phi_p, c_j), beta=0.2)
    (TH zeroes values <= beta); g = per-patch MLP m->500->512 (ReLU), then the model's own avgpool+fc.
  - Loss = CE against the model's own predicted classes (both tasks) - lambda*(coherence - diversity), lambda=0.1,
    coherence = mean top-K(K=32) patch cosines per concept within the batch, diversity = mean pairwise cos(c_j, c_k).
    Trained on a fixed random 40k of official train; Adam 1e-3, batch 256, 10 epochs, seed 0.
  - ConceptSHAP = Shapley value of each concept for the completeness score
      eta_t(S) = (acc_t(S) - a_r) / (1 - a_r), acc vs the model's own predictions, a_r = majority rate,
    by Monte-Carlo permutation sampling (300 permutations) on the clean val pool.
  - Naming: each of the 26 concepts gets the concept vector whose image-level presence (max patch cosine) has the highest
    train AUROC against the CelebA label (supervised naming, favours the baseline; duplicates allowed and reported).
  - Two scores per (task, concept): 'shap_acc' = the original completeness ConceptSHAP (unsigned in spirit), and
    'shap_logit' = our signed adaptation: same Shapley procedure with value = mean positive-class logit over val images
    labelled with that concept."""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS

F = Path(os.environ.get("CARDS_FEAT_CACHE", "results/inpainting/feat_cache"))
OUT = Path(sys.argv[1]); DEV = "cuda"
M, BETA, LAM, TOPK, LR, BS, EPOCHS, N_PERM = 40, 0.2, 0.1, 32, 1e-3, 256, 10, 300
POS = {"Attractive": 1, "Male": 3}
torch.manual_seed(0); rng = np.random.default_rng(0)


def patches(maps):  # (N,512,7,7) -> (N,49,512)
    return maps.flatten(2).transpose(1, 2)


class CShap(torch.nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.C = torch.nn.Parameter(torch.randn(d, M) * 0.05)
        self.g = torch.nn.Sequential(torch.nn.Linear(M, 500), torch.nn.ReLU(), torch.nn.Linear(500, d), torch.nn.ReLU())

    def cos(self, P):
        return Fn.normalize(P, dim=-1) @ Fn.normalize(self.C, dim=0)

    def scores(self, P):
        s = self.cos(P); return s * (s > BETA)

    def pooled(self, v):  # v: (N,49,M) -> (N,512)
        return self.g(v).mean(1)


def main():
    tr, va, pv = torch.load(F / "map_train40k.pt"), torch.load(F / "map_val.pt"), torch.load(F / "pooled_val.pt")
    W, b = pv["fc_w"].to(DEV), pv["fc_b"].to(DEV)
    names = pv["attr_names"]; ci = [names.index(c) for c in GROUNDABLE_CONCEPTS]
    P = patches(tr["maps"]).to(DEV); Lt = tr["logits"].to(DEV)
    yt = torch.stack([Lt[:, p - 1:p + 1].argmax(1) for p in POS.values()], 1)
    model = CShap().to(DEV); opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(P), device=DEV); tot = 0
        for s in range(0, len(P), BS):
            i = perm[s:s + BS]; x = P[i].float()
            logit = model.pooled(model.scores(x)) @ W.T + b
            ce = sum(Fn.cross_entropy(logit[:, p - 1:p + 1], yt[i, k]) for k, p in enumerate(POS.values()))
            cs = model.cos(x).reshape(-1, M)
            coh = cs.topk(TOPK, dim=0).values.mean()
            Cn = Fn.normalize(model.C, dim=0); G = Cn.T @ Cn
            div = (G.sum() - G.diag().sum()) / (M * (M - 1))
            loss = ce - LAM * (coh - div)
            opt.zero_grad(); loss.backward(); opt.step(); tot += ce.item() * len(i)
        print(f"epoch {ep} CE {tot / len(P):.4f} coherence {coh.item():.3f} diversity {div.item():.3f}", flush=True)

    with torch.no_grad():
        # naming on train (image-level presence = max patch cosine)
        pres = torch.cat([model.cos(P[s:s + 2048].float()).amax(1) for s in range(0, len(P), 2048)]).cpu().numpy()
        Ytr = tr["labels"][:, ci].numpy()
        auc = np.array([[roc_auc_score(Ytr[:, n], pres[:, j]) for n in range(len(ci))] for j in range(M)])
        assign = auc.argmax(0)
        del P
        Pv = patches(va["maps"]).to(DEV); V = torch.cat([model.scores(Pv[s:s + 2048].float()) for s in range(0, len(Pv), 2048)])
        Lv = pv["logits"].to(DEV)
        yv = torch.stack([Lv[:, p - 1:p + 1].argmax(1) for p in POS.values()], 1)
        ar = torch.stack([torch.maximum(y.float().mean(), 1 - y.float().mean()) for y in yv.T])
        Yv = torch.from_numpy(pv["labels"][:, ci].numpy()).to(DEV).float()

        def value(mask):  # mask: (M,) 0/1 concepts kept -> completeness per task (2,), subset-mean pos logits (26,2)
            out = []
            for s in range(0, len(V), 2048):
                out.append(model.pooled(V[s:s + 2048] * mask) @ W.T + b)
            L = torch.cat(out)
            acc = torch.stack([(L[:, p - 1:p + 1].argmax(1) == yv[:, k]).float().mean() for k, p in enumerate(POS.values())])
            eta = (acc - ar) / (1 - ar)
            pos = torch.stack([L[:, p] for p in POS.values()], 1)  # (N,2)
            sub = (Yv.T @ pos) / Yv.sum(0)[:, None]  # (26,2)
            return eta, sub

        full_eta, _ = value(torch.ones(M, device=DEV))
        print(f"completeness with all {M} concepts: {dict(zip(POS, full_eta.tolist()))}", flush=True)
        sh_eta = torch.zeros(M, 2, device=DEV); sh_sub = torch.zeros(M, 26, 2, device=DEV)
        for n in range(N_PERM):
            order = rng.permutation(M); mask = torch.zeros(M, device=DEV)
            pe, ps = value(mask)
            for j in order:
                mask[j] = 1; e, s_ = value(mask)
                sh_eta[j] += e - pe; sh_sub[j] += s_ - ps; pe, ps = e, s_
            if n % 50 == 0: print(f"  permutation {n}", flush=True)
        sh_eta /= N_PERM; sh_sub /= N_PERM
    dup = {int(j): [GROUNDABLE_CONCEPTS[n] for n in np.where(assign == j)[0]] for j in set(assign.tolist())}
    print("concept vectors shared by several names:", {j: v for j, v in dup.items() if len(v) > 1}, flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["task", "concept_name", "concept_vector", "train_auroc", "shap_acc", "shap_logit"])
        for n, c in enumerate(GROUNDABLE_CONCEPTS):
            j = int(assign[n])
            for k, t in enumerate(POS):
                w.writerow([t, c, j, f"{auc[j, n]:.4f}", float(sh_eta[j, k]), float(sh_sub[j, n, k])])
            print(f"  {c:20s} vector {j:2d} AUROC {auc[j, n]:.3f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
