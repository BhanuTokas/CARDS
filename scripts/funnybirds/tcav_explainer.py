"""Local TCAV (Kim et al., 2018) as a FunnyBirds explainer, following scripts/celeba/baselines/local_tcav.py.

Layer: layer3 of the ResNet-50 (the paper's TCAV layer; after layer4 only pooling and the linear head remain, so
the gradient there is input-independent). CAVs: for each part, a linear classifier (sklearn SGDClassifier,
alpha=0.01 as captum's default, random_state=0) separating the layer's activations of N_EXEMPLARS training images
WITH the part from each of N_RANDOM sets of N_PER_SET training images WITHOUT it (FunnyBirds renders the same
birds with parts removed, a cleaner contrast than CelebA's random sets, where most images would contain the part).
CAV = unit-norm weight vector, pointing toward the part. Images drawn with random.Random(SEED).

Part importance PI(part, class) = mean over the N_RANDOM CAVs of grad_layer3(logit[class]) . CAV, the continuous
directional derivative ("tcav_dd" in the CelebA local runs; its sign fraction is too coarse for a 5-part ranking).
variant=True (TCAV_variant): one CAV per part VARIANT (positives show that variant; negatives, as above, have the
part removed), and PI(part, class) uses the CAV of the class's own variant of the part. Classes differ by variant,
and with part-presence CAVs the predicted class's logit DEcreases along the CAV for most images (about three quarters
of the positives show another variant), which left most explanations empty (all PIs <= 0); the variant CAVs match
the concepts of PCBM, ConceptSAE and ConceptSHAP.
Important parts and background objects: as in hns_explainer.py (PI > t * sum of positive PIs; background objects
are not scored, PI = 0).
"""
import random
import zlib

import fb_data
import numpy as np
import torch
from paths import PARTS
from sklearn.linear_model import SGDClassifier

N_EXEMPLARS, N_RANDOM, N_PER_SET, SEED = 40, 6, 25, 42
N_BG = 50


def split(net):
    """(pre, post) for a torchvision-style ResNet: pre(x) = layer3 activations, post(a) = logits."""
    def pre(x):
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        return net.layer3(net.layer2(net.layer1(x)))

    def post(a):
        return net.fc(torch.flatten(net.avgpool(net.layer4(a)), 1))
    return pre, post


class TCAVExplainer:
    def __init__(self, model, device="cuda", variant=False):
        self.device, self.variant = device, variant
        self.pre, self.post = split(model.model)
        self.cavs = self._fit_variant_cavs() if variant else self._fit_cavs()
        self._grads: dict[tuple[int, int], torch.Tensor] = {}

    @torch.no_grad()
    def _acts(self, idx):
        out = []
        for s in range(0, len(idx), 32):
            x = torch.stack([fb_data.train_image(i) for i in idx[s:s + 32]]).to(self.device)
            out.append(self.pre(x).flatten(1).cpu())
        return torch.cat(out).numpy()

    def _fit_cavs(self):
        pres = fb_data.presence_matrix()
        rng = random.Random(SEED)
        cavs = {}
        for k, p in enumerate(PARTS):
            pos = [i for i in range(len(pres)) if pres[i, k]]; rng.shuffle(pos)
            neg = [i for i in range(len(pres)) if not pres[i, k]]; rng.shuffle(neg)
            ca = self._acts(pos[:N_EXEMPLARS]); vs = []
            for j in range(N_RANDOM):
                na = self._acts(neg[j * N_PER_SET:(j + 1) * N_PER_SET])
                X = np.concatenate([ca, na]); y = np.r_[np.ones(len(ca)), np.zeros(len(na))]
                w = SGDClassifier(alpha=0.01, max_iter=1000, tol=1e-3, random_state=0).fit(X, y).coef_[0]
                vs.append(w / np.linalg.norm(w))
            cavs[p] = torch.tensor(np.stack(vs), dtype=torch.float32, device=self.device)  # (N_RANDOM, d)
            print(f"  CAVs {p}", flush=True)
        return cavs

    def _fit_variant_cavs(self):
        pres, vm = fb_data.presence_matrix(), fb_data.variant_matrix()
        rng = random.Random(SEED)
        cavs = {}
        for j, (p, v) in enumerate(fb_data.variant_concepts()):
            k = PARTS.index(p)
            pos = [i for i in range(len(vm)) if vm[i, j]]; rng.shuffle(pos)
            neg = [i for i in range(len(pres)) if not pres[i, k]]; rng.shuffle(neg)
            ca = self._acts(pos[:N_EXEMPLARS]); vs = []
            for r in range(N_RANDOM):
                na = self._acts(neg[r * N_PER_SET:(r + 1) * N_PER_SET])
                X = np.concatenate([ca, na]); y = np.r_[np.ones(len(ca)), np.zeros(len(na))]
                w = SGDClassifier(alpha=0.01, max_iter=1000, tol=1e-3, random_state=0).fit(X, y).coef_[0]
                vs.append(w / np.linalg.norm(w))
            cavs[p, v] = torch.tensor(np.stack(vs), dtype=torch.float32, device=self.device)
            print(f"  CAVs {p}{v + 1:02d}", flush=True)
        return cavs

    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)

    def _grad(self, image, cls):
        key = (zlib.crc32((image[0].detach().cpu().numpy() * 255).round().astype(np.uint8).tobytes()), cls)
        if key not in self._grads:
            with torch.enable_grad():
                a = self.pre(image[:1].to(self.device)).detach().requires_grad_(True)
                g = torch.autograd.grad(self.post(a)[0, cls], a)[0].flatten()
            self._grads[key] = g
        return self._grads[key]

    def get_part_importance(self, image, part_map, target, colors_to_part=None, with_bg=False):
        t = int(target.reshape(-1)[0]) if torch.is_tensor(target) else int(target)
        g = self._grad(image, t)
        if self.variant:
            cv = fb_data.classes()[t]["parts"]
            pi = {p: float((self.cavs[p, cv[p]] @ g).mean()) for p in PARTS}
        else:
            pi = {p: float((self.cavs[p] @ g).mean()) for p in PARTS}
        if with_bg:
            pi.update({f"bg_{i:03d}": 0.0 for i in range(N_BG)})
        return pi

    def get_important_parts(self, image, part_map, target, colors_to_part=None, thresholds=None, with_bg=False):
        pi = self.get_part_importance(image, part_map, target, colors_to_part, with_bg=with_bg)
        total = sum(max(v, 0.0) for v in pi.values())
        thresholds = self.get_p_thresholds() if thresholds is None else thresholds
        return [[k for k, v in pi.items() if total > 0 and v > total * t] for t in thresholds]
