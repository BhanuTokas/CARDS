"""ConceptSHAP baseline (Yeh et al., NeurIPS 2020) on COCO-Stuff last-stage maps (7x7xd; env CARDS_ARCH selects resnet18|vit|convnext); port of
scripts/celeba/baselines/concept_shap.py. argv[1] = out CSV. Design copied from CelebA, fixed before seeing any GT correlation:
  - m=40 concept vectors on layer4 patch features; concept score per patch = TH(cos(phi_p, c_j), beta=0.2);
    g = per-patch MLP m->500->d (ReLU), then the model's own avgpool + 20-way fc.
  - Loss = BCE against the model's own hard predictions (logit>0, all 20 classes) - lambda*(coherence - diversity),
    lambda=0.1, coherence = mean top-K (K=32) patch cosines per concept in the batch, diversity = mean pairwise cos(c_j,c_k).
    Trained on the fixed random 40k pool images; Adam 1e-3, batch 256, 10 epochs, seed 0.
  - ConceptSHAP = Shapley value of each concept for the completeness score
      eta_k(S) = (acc_k(S) - a_r) / (1 - a_r), accuracy vs the model's own predictions, a_r = majority rate,
    by Monte-Carlo permutation sampling (N_PERM, default 300) on an evaluation set.
  - Naming: each concept gets the concept vector whose image-level presence (max patch cosine) has the highest train AUROC
    against its label (supervised naming, favours the baseline; duplicates allowed and reported).
  - Two scores per (concept, class): 'shap_acc' = original completeness ConceptSHAP (class-specific, identical for every
    concept sharing a vector), and 'shap_logit' = signed adaptation: same Shapley procedure with value = mean class logit
    over evaluation images labelled with that concept.
COCO difference vs CelebA: the evaluation set is the union of up to EVAL_PER_CONCEPT positives per concept plus EVAL_RANDOM
random images (a full-pool sweep x 12,000 coalitions is not tractable); concepts are not mutually exclusive. env N_PERM,
EVAL_PER_CONCEPT, EVAL_RANDOM override. Needs extract_feats_coco.py first."""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from coco_common import CLASSES, DEVICE, FEAT_CACHE

M, BETA, LAM, TOPK, LR, BS, EPOCHS = 40, 0.2, 0.1, 32, 1e-3, 256, 10
N_PERM = int(os.environ.get("N_PERM", "300"))
EVAL_PER_CONCEPT = int(os.environ.get("EVAL_PER_CONCEPT", "40")); EVAL_RANDOM = int(os.environ.get("EVAL_RANDOM", "2000"))
NC = len(CLASSES)
torch.manual_seed(0); rng = np.random.default_rng(0)


def patches(maps):  # (N,d,7,7) -> (N,49,d)
    return maps.flatten(2).transpose(1, 2)


class CShap(torch.nn.Module):
    def __init__(self, d=512):  # d = head input width of the chosen black box
        super().__init__()
        self.C = torch.nn.Parameter(torch.randn(d, M) * 0.05)
        self.g = torch.nn.Sequential(torch.nn.Linear(M, 500), torch.nn.ReLU(), torch.nn.Linear(500, d), torch.nn.ReLU())

    def cos(self, P): return Fn.normalize(P, dim=-1) @ Fn.normalize(self.C, dim=0)
    def scores(self, P): s = self.cos(P); return s * (s > BETA)
    def pooled(self, v): return self.g(v).mean(1)


def main():
    out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)
    pv = torch.load(FEAT_CACHE / "pooled.pt", weights_only=False); mp = torch.load(FEAT_CACHE / "map_train40k.pt")
    W, b = pv["fc_w"].to(DEVICE), pv["fc_b"].to(DEVICE)
    concepts = pv["concepts"]; Yall = pv["labels"].numpy(); Lall = pv["logits"]
    idx = mp["idx"]; P = patches(mp["maps"]).to(DEVICE); yt = (Lall[idx] > 0).float().to(DEVICE); Ytr = Yall[idx.numpy()]
    model = CShap(W.shape[1]).to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(P), device=DEVICE); tot = 0
        for s in range(0, len(P), BS):
            i = perm[s:s + BS]; x = P[i].float()
            logit = model.pooled(model.scores(x)) @ W.T + b
            ce = Fn.binary_cross_entropy_with_logits(logit, yt[i])
            cs = model.cos(x).reshape(-1, M); coh = cs.topk(TOPK, dim=0).values.mean()
            Cn = Fn.normalize(model.C, dim=0); G = Cn.T @ Cn; div = (G.sum() - G.diag().sum()) / (M * (M - 1))
            loss = ce - LAM * (coh - div)
            opt.zero_grad(); loss.backward(); opt.step(); tot += ce.item() * len(i)
        print(f"epoch {ep} BCE {tot / len(P):.4f} coherence {coh.item():.3f} diversity {div.item():.3f}", flush=True)

    with torch.no_grad():
        pres = torch.cat([model.cos(P[s:s + 2048].float()).amax(1) for s in range(0, len(P), 2048)]).cpu().numpy()
        auc = np.array([[roc_auc_score(Ytr[:, n], pres[:, j]) if 0 < Ytr[:, n].sum() < len(Ytr) else 0.5
                         for n in range(len(concepts))] for j in range(M)])
        assign = auc.argmax(0)
        # evaluation set: enriched in concept positives so every concept has images (maps come from the 40k pool)
        ev = set(rng.choice(len(idx), min(EVAL_RANDOM, len(idx)), replace=False).tolist())
        for n in range(len(concepts)):
            pos = np.where(Ytr[:, n])[0]
            ev.update(rng.choice(pos, min(EVAL_PER_CONCEPT, len(pos)), replace=False).tolist() if len(pos) else [])
        ev = np.array(sorted(ev)); print(f"evaluation set: {len(ev)} images", flush=True)
        Pv = P[ev]; V = torch.cat([model.scores(Pv[s:s + 2048].float()) for s in range(0, len(Pv), 2048)])
        yv = yt[ev]; Yv = torch.from_numpy(Ytr[ev]).to(DEVICE).float(); Lv = Lall[idx][ev].to(DEVICE)
        rate = yv.mean(0); ar = torch.maximum(rate, 1 - rate)
        npos = Yv.sum(0).clamp(min=1)

        def value(mask):  # -> completeness per class (20,), subset-mean class logits (n_concepts, 20)
            L = torch.cat([model.pooled(V[s:s + 2048] * mask) @ W.T + b for s in range(0, len(V), 2048)])
            acc = ((L > 0).float() == yv).float().mean(0)
            return (acc - ar) / (1 - ar).clamp(min=1e-6), (Yv.T @ L) / npos[:, None]

        full_eta, _ = value(torch.ones(M, device=DEVICE))
        print(f"completeness with all {M} concepts, mean over classes: {full_eta.mean().item():.3f}", flush=True)
        sh_eta = torch.zeros(M, NC, device=DEVICE); sh_sub = torch.zeros(M, len(concepts), NC, device=DEVICE)
        for n in range(N_PERM):
            order = rng.permutation(M); mask = torch.zeros(M, device=DEVICE); pe, ps = value(mask)
            for j in order:
                mask[j] = 1; e, s_ = value(mask)
                sh_eta[j] += e - pe; sh_sub[j] += s_ - ps; pe, ps = e, s_
            if n % 50 == 0: print(f"  permutation {n}", flush=True)
        sh_eta /= N_PERM; sh_sub /= N_PERM
    dup = {int(j): [concepts[n] for n in np.where(assign == j)[0]] for j in set(assign.tolist())}
    print("concept vectors shared by several names:", {j: len(v) for j, v in dup.items() if len(v) > 1}, flush=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["concept_name", "class_name", "concept_vector", "train_auroc", "shap_acc", "shap_logit"])
        for n, c in enumerate(concepts):
            j = int(assign[n])
            for k, cls in enumerate(CLASSES):
                w.writerow([c, cls, j, f"{auc[j, n]:.4f}", float(sh_eta[j, k]), float(sh_sub[j, n, k])])
    print("done", flush=True)


if __name__ == "__main__":
    main()
