"""ConceptSHAP (Yeh et al., NeurIPS 2020) as a FunnyBirds explainer; same design as scripts/bam/fit_concept_shap_bam.py.

  - M=40 concept vectors on the ResNet-50's layer4 patch features (8x8x2048 at 256x256 input); concept score per
    patch = TH(cos(phi_p, c_j), beta=0.2), which zeroes values <= beta; g = per-patch MLP M -> 500 -> 2048 (ReLU),
    then the model's own average pool and linear head fc.
  - Loss = cross-entropy against the model's own predicted class - lambda * (coherence - diversity), lambda=0.1;
    coherence = mean top-32 patch cosines per concept in the batch, diversity = mean pairwise cos(c_j, c_k).
    Adam 1e-3, batch 256, 10 epochs, seed 0, on N_FIT training images (random.Random(SEED)); as on BAM, a subset,
    because the patch maps of all 50,000 would not fit in memory.
  - Naming: each of the 26 part VARIANT concepts (as for conventional PCBM and ConceptSAE) gets the vector whose
    image-level presence (max patch cosine) has the highest AUROC on the fitting images. Duplicates are reported.
  - Completeness (Yeh et al.'s eta, all M vectors kept) on N_HOLD held-out training images is reported as a check.
  - Local score (BAM scored each class over held-out images; FunnyBirds needs one image): PI(part, class) = sum over
    the part's named vectors (deduplicated) of the vector's Shapley value for this image's class logit
    fc(avgpool(g(scores * mask))), by Monte-Carlo permutation sampling (N_PERM permutations, the same for every image).
Important parts and background objects as in hns_explainer.py (PI > t * sum of positive PIs; background PI = 0).
"""
import os
import random
import zlib

import fb_data
import numpy as np
import torch
import torch.nn.functional as Fn
from paths import PARTS
from sklearn.metrics import roc_auc_score

SEED, N_BG = 42, 50
M, BETA, LAM, TOPK, LR, BS, EPOCHS, N_PERM = 40, 0.2, 0.1, 32, 1e-3, 256, 10, 300
N_FIT = int(os.environ.get("FUNNYBIRDS_CSHAP_NFIT", 10_000))  # smaller only for smoke tests
N_HOLD = 2_000
D = 2048


class CShap(torch.nn.Module):  # as in scripts/bam/fit_concept_shap_bam.py
    def __init__(self):
        super().__init__()
        self.C = torch.nn.Parameter(torch.randn(D, M) * 0.05)
        self.g = torch.nn.Sequential(torch.nn.Linear(M, 500), torch.nn.ReLU(), torch.nn.Linear(500, D), torch.nn.ReLU())

    def cos(self, P):
        return Fn.normalize(P, dim=-1) @ Fn.normalize(self.C, dim=0)

    def scores(self, P):
        s = self.cos(P)
        return s * (s > BETA)

    def pooled(self, v):
        return self.g(v).mean(-2)


class CShapExplainer:
    def __init__(self, model, device="cuda"):
        self.device, self.net = device, model.model
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        self.W, self.B = self.net.fc.weight.detach().float(), self.net.fc.bias.detach().float()
        idx = list(range(len(fb_data.train_params()))); random.Random(SEED).shuffle(idx)
        hold_idx, fit_idx = idx[:N_HOLD], idx[N_HOLD:N_HOLD + N_FIT]
        P = self._patch_feats(fit_idx)
        with torch.no_grad():
            L = torch.cat([P[s:s + 512].float().mean(1) @ self.W.T + self.B for s in range(0, len(P), 512)])
            x0 = torch.stack([fb_data.train_image(i) for i in fit_idx[:8]]).to(device)
            check = (L[:8] - self.net(x0).float()).abs().max().item()
        print(f"  pooled layer4 -> fc reproduces the logits, max error {check:.2e}", flush=True)
        yt = L.argmax(1)

        torch.manual_seed(0)
        self.cs = cs = CShap().to(device)
        opt = torch.optim.Adam(cs.parameters(), lr=LR)
        for ep in range(EPOCHS):
            perm = torch.randperm(len(P), device=device); tot = 0.0
            for s in range(0, len(P), BS):
                i = perm[s:s + BS]
                x = P[i].float()
                ce = Fn.cross_entropy(cs.pooled(cs.scores(x)) @ self.W.T + self.B, yt[i])
                coh = cs.cos(x).reshape(-1, M).topk(TOPK, dim=0).values.mean()
                Cn = Fn.normalize(cs.C, dim=0); G = Cn.T @ Cn
                div = (G.sum() - G.diag().sum()) / (M * (M - 1))
                loss = ce - LAM * (coh - div)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += ce.item() * len(i)
            print(f"  ConceptSHAP epoch {ep} CE {tot / len(P):.4f} coherence {coh.item():.3f} "
                  f"diversity {div.item():.3f}", flush=True)
        cs.eval()

        with torch.no_grad():
            pres = torch.cat([cs.cos(P[s:s + 1024].float()).amax(1) for s in range(0, len(P), 1024)]).cpu().numpy()
        del P
        Y = fb_data.variant_matrix()[fit_idx]
        auc = np.array([[roc_auc_score(Y[:, n], pres[:, j]) for n in range(Y.shape[1])] for j in range(M)])
        self.part_vectors = {p: set() for p in PARTS}
        self.naming = []
        for n, (p, v) in enumerate(fb_data.variant_concepts()):
            j = int(auc[:, n].argmax())
            self.part_vectors[p].add(j)
            self.naming.append({"concept": f"{p}{v + 1:02d}", "vector": j, "train_auroc": round(float(auc[j, n]), 4)})
            print(f"  {p}{v + 1:02d}: vector {j:2d} AUROC {auc[j, n]:.3f}", flush=True)
        shared = {j for p in PARTS for q in PARTS if p < q for j in self.part_vectors[p] & self.part_vectors[q]}
        if shared:
            print(f"  note: vectors named for more than one part: {sorted(shared)}", flush=True)
        self.part_vectors = {p: sorted(js) for p, js in self.part_vectors.items()}

        Ph = self._patch_feats(hold_idx)
        with torch.no_grad():
            Lh = torch.cat([Ph[s:s + 512].float().mean(1) @ self.W.T + self.B for s in range(0, len(Ph), 512)])
            Lc = torch.cat([cs.pooled(cs.scores(Ph[s:s + 512].float())) @ self.W.T + self.B
                            for s in range(0, len(Ph), 512)])
            yh = Lh.argmax(1)
            a_r = (torch.bincount(yh, minlength=self.W.shape[0]).float() / len(yh)).max()
            acc = (Lc.argmax(1) == yh).float().mean()
            self.completeness = ((acc - a_r) / (1 - a_r)).item()
        del Ph
        print(f"  ConceptSHAP held-out completeness (eta) {self.completeness:.3f}, agreement {acc.item():.3f}",
              flush=True)
        rng = np.random.default_rng(0)
        perms = np.stack([rng.permutation(M) for _ in range(N_PERM)])
        # masks[n, s] = vectors in the first s positions of permutation n, s = 0..M
        masks = np.zeros((N_PERM, M + 1, M), np.float32)
        for n, order in enumerate(perms):
            for s in range(1, M + 1):
                masks[n, s] = masks[n, s - 1]; masks[n, s, order[s - 1]] = 1
        self.masks = torch.tensor(masks.reshape(-1, M), device=device)
        self.perms = torch.tensor(perms, device=device)
        self.meta = {"cshap_completeness": self.completeness, "cshap_agreement": acc.item(),
                     "cshap_mean_auroc": float(np.mean([r["train_auroc"] for r in self.naming])),
                     "cshap_naming": self.naming}

    @torch.no_grad()
    def _layer4(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x.to(self.device)))))
        a = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        return a.flatten(2).transpose(1, 2)  # (B, 64, 2048)

    def _patch_feats(self, idx):
        dl = torch.utils.data.DataLoader(fb_data.TrainImages(idx), batch_size=128,
                                         num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
        out = []
        for b, x in enumerate(dl):
            out.append(self._layer4(x).half())
            if b % 20 == 0:
                print(f"    patch features {min((b + 1) * 128, len(idx))}/{len(idx)}", flush=True)
        return torch.cat(out)

    @torch.no_grad()
    def _shapley(self, image, cls):
        """(M,) Monte-Carlo Shapley values of the concept vectors for this image's class-cls logit."""
        v = self.cs.scores(self._layer4(image[:1]).float())[0]  # (64, M)
        vals = []
        for s in range(0, len(self.masks), 1024):
            mk = self.masks[s:s + 1024]
            vals.append(self.cs.pooled(v[None] * mk[:, None, :]) @ self.W[cls] + self.B[cls])
        vals = torch.cat(vals).reshape(N_PERM, M + 1)
        gain = vals[:, 1:] - vals[:, :-1]  # gain of adding perms[n, s]
        sh = torch.zeros(M, device=self.device).index_add_(0, self.perms.flatten(), gain.flatten())
        return (sh / N_PERM).cpu().numpy()

    # ----- framework interface -----
    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)

    def get_part_importance(self, image, part_map, target, colors_to_part=None, with_bg=False):
        t = int(target.reshape(-1)[0]) if torch.is_tensor(target) else int(target)
        key = (zlib.crc32((image[0].detach().cpu().numpy() * 255).round().astype(np.uint8).tobytes()), t)
        if key not in self._cache:
            self._cache[key] = self._shapley(image.detach(), t)
        sh = self._cache[key]
        pi = {p: float(sum(sh[j] for j in js)) for p, js in self.part_vectors.items()}
        if with_bg:
            pi.update({f"bg_{i:03d}": 0.0 for i in range(N_BG)})
        return pi

    def get_important_parts(self, image, part_map, target, colors_to_part=None, thresholds=None, with_bg=False):
        pi = self.get_part_importance(image, part_map, target, colors_to_part, with_bg=with_bg)
        total = sum(max(v, 0.0) for v in pi.values())
        thresholds = self.get_p_thresholds() if thresholds is None else thresholds
        return [[k for k, v in pi.items() if total > 0 and v > total * t] for t in thresholds]
