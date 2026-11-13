"""ConceptSAE (Discover-then-Name style) as a FunnyBirds explainer; same design as scripts/bam/fit_concept_sae_bam.py.

  - TopK SAE (Gao et al. 2024) on the ResNet-50's pooled 2048-d features (avgpool, the input of its linear head fc)
    of the training images: 4096 latents, k=32, unit-norm decoder columns, Adam lr 4e-4, batch 4096, 60 epochs,
    seed 0. N_HOLD training images (random.Random(SEED)) are held out to report reconstruction quality (FVU) and
    prediction agreement through the SAE; the rest train it.
  - Naming: each of the 26 part VARIANT concepts (fb_data.variant_concepts(); classes differ by variant, as for
    conventional PCBM) gets the latent with the highest AUROC against that variant's presence on the SAE's training
    images. Supervised naming favours the baseline.
  - Local score: PI(part, class) = sum over the part's named latents j (deduplicated) of z_j(x) * (w_class . d_j),
    the latent's exact contribution to the class logit through fc.
Important parts and background objects as in hns_explainer.py (PI > t * sum of positive PIs; background PI = 0).
"""
import os
import random
import zlib

import fb_data
import numpy as np
import torch
from paths import PARTS
from scipy.stats import rankdata

SEED, N_HOLD, N_BG = 42, 5_000, 50
N_LAT, K, LR, BS, EPOCHS = 4096, 32, 4e-4, 4096, 60
N_TRAIN = int(os.environ.get("FUNNYBIRDS_SAE_NTRAIN", 0))  # 0 = all training images; smaller only for smoke tests


class TopKSAE(torch.nn.Module):  # as in scripts/bam/fit_concept_sae_bam.py
    def __init__(self, d, n, k, init_bias):
        super().__init__()
        dec = torch.randn(d, n)
        dec /= dec.norm(dim=0, keepdim=True)
        self.W_dec = torch.nn.Parameter(dec)
        self.W_enc = torch.nn.Parameter(dec.T.clone())
        self.b_enc = torch.nn.Parameter(torch.zeros(n))
        self.b_dec = torch.nn.Parameter(init_bias.clone())
        self.k = k

    def encode(self, x):
        pre = torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)
        v, i = pre.topk(self.k, dim=1)
        return torch.zeros_like(pre).scatter_(1, i, v)

    def forward(self, x):
        z = self.encode(x)
        return z @ self.W_dec.T + self.b_dec, z


class SAEExplainer:
    def __init__(self, model, device="cuda"):
        self.device, self.net = device, model.model
        self._z_cache: dict[int, torch.Tensor] = {}
        idx = list(range(len(fb_data.train_params()))); random.Random(SEED).shuffle(idx)
        if N_TRAIN:
            idx = idx[:N_TRAIN + N_HOLD]
        hold_idx, fit_idx = idx[:N_HOLD], idx[N_HOLD:]
        X, Xh = self._features(fit_idx), self._features(hold_idx)
        W, B = self.net.fc.weight.detach().float(), self.net.fc.bias.detach().float()

        torch.manual_seed(0)
        sae = TopKSAE(X.shape[1], N_LAT, K, X.median(0).values).to(device)
        opt = torch.optim.Adam(sae.parameters(), lr=LR)
        for ep in range(EPOCHS):
            perm = torch.randperm(len(X), device=device); tot = 0.0
            for s in range(0, len(X), BS):
                x = X[perm[s:s + BS]]
                loss = ((sae(x)[0] - x) ** 2).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    sae.W_dec /= sae.W_dec.norm(dim=0, keepdim=True)
                tot += loss.item() * len(x)
            if ep % 10 == 0 or ep == EPOCHS - 1:
                print(f"  SAE epoch {ep} mse {tot / len(X):.4f}", flush=True)
        self.sae = sae.eval()
        with torch.no_grad():
            xh, zh = sae(Xh)
            self.fvu = (((xh - Xh) ** 2).sum() / ((Xh - Xh.mean(0)) ** 2).sum()).item()
            self.agreement = ((Xh @ W.T + B).argmax(1) == (xh @ W.T + B).argmax(1)).float().mean().item()
            dead = (zh.sum(0) == 0).float().mean().item()
            Z = torch.cat([sae.encode(X[s:s + 8192]).cpu() for s in range(0, len(X), 8192)]).numpy()
            self.contrib = (W @ sae.W_dec).cpu().numpy()  # (classes, latents): w_k . d_j
        print(f"  SAE held-out FVU {self.fvu:.3f}; prediction agreement through SAE {self.agreement:.3f}; "
              f"dead latents {dead:.2f}", flush=True)

        Y = fb_data.variant_matrix()[fit_idx].astype(float)
        npos, nneg = Y.sum(0), len(Y) - Y.sum(0)
        auc = np.zeros((N_LAT, Y.shape[1]))
        for s in range(0, N_LAT, 256):
            R = rankdata(Z[:, s:s + 256], axis=0)
            auc[s:s + 256] = (R.T @ Y - npos * (npos + 1) / 2) / (npos * nneg)
        self.part_latents = {p: set() for p in PARTS}
        self.naming = []
        for n, (p, v) in enumerate(fb_data.variant_concepts()):
            j = int(auc[:, n].argmax())
            self.part_latents[p].add(j)
            self.naming.append({"concept": f"{p}{v + 1:02d}", "latent": j, "train_auroc": round(float(auc[j, n]), 4)})
            print(f"  {p}{v + 1:02d}: latent {j:4d} AUROC {auc[j, n]:.3f}", flush=True)
        shared = {j for p in PARTS for q in PARTS if p < q for j in self.part_latents[p] & self.part_latents[q]}
        if shared:
            print(f"  note: latents named for more than one part: {sorted(shared)}", flush=True)
        self.part_latents = {p: sorted(js) for p, js in self.part_latents.items()}
        self.meta = {"sae_fvu": self.fvu, "sae_agreement": self.agreement, "sae_dead": dead,
                     "sae_mean_auroc": float(np.mean([r["train_auroc"] for r in self.naming])), "sae_naming": self.naming}

    @torch.no_grad()
    def _pooled(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x.to(self.device)))))
        x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        return torch.flatten(n.avgpool(x), 1).float()

    def _features(self, idx):
        dl = torch.utils.data.DataLoader(fb_data.TrainImages(idx), batch_size=128,
                                         num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
        out = []
        for b, x in enumerate(dl):
            out.append(self._pooled(x))
            if b % 50 == 0:
                print(f"    features {min((b + 1) * 128, len(idx))}/{len(idx)}", flush=True)
        return torch.cat(out)

    # ----- framework interface -----
    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)

    def get_part_importance(self, image, part_map, target, colors_to_part=None, with_bg=False):
        key = zlib.crc32((image[0].detach().cpu().numpy() * 255).round().astype(np.uint8).tobytes())
        if key not in self._z_cache:
            with torch.no_grad():
                self._z_cache[key] = self.sae.encode(self._pooled(image[:1].detach()))[0].cpu().numpy()
        z = self._z_cache[key]
        t = int(target.reshape(-1)[0]) if torch.is_tensor(target) else int(target)
        pi = {p: float(sum(z[j] * self.contrib[t, j] for j in js)) for p, js in self.part_latents.items()}
        if with_bg:
            pi.update({f"bg_{i:03d}": 0.0 for i in range(N_BG)})
        return pi

    def get_important_parts(self, image, part_map, target, colors_to_part=None, thresholds=None, with_bg=False):
        pi = self.get_part_importance(image, part_map, target, colors_to_part, with_bg=with_bg)
        total = sum(max(v, 0.0) for v in pi.values())
        thresholds = self.get_p_thresholds() if thresholds is None else thresholds
        return [[k for k, v in pi.items() if total > 0 and v > total * t] for t in thresholds]
