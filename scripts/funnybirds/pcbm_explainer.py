"""Post-hoc CBM (Yuksekgonul et al., ICLR 2023) as a FunnyBirds explainer, in the paper's three variants.

Same recipe as the paper's CelebA PCBM runs (scripts/celeba/run/run_pcbm_*_official_train.py, which call the
post_hoc_cbm repo); the few functions involved are reproduced here so the job does not need that repo:
  conventional  concepts = CAVs in the black box's own penultimate features (ResNet-50 avgpool, 2048-d): a linear
                SVC(C=0.1) on N_CAV positive vs N_CAV negative training images (post_hoc_cbm get_cavs); concept
                activation = margin (emb.w + b) / ||w|| (PosthocLinearCBM.compute_dist). Concepts are the 26 part
                VARIANTS (beak01..wing06: classes differ by variant, so 5 presence concepts could not separate them);
                positives show that variant, negatives do not (other variants or the part removed).
  clip_rn50 / siglip
                concepts = text embeddings of the five part queries (the same texts as HnS; the variants are unnamed
                3D models, so a language-defined concept can only name the part); activation = projection
                emb.c / ||c||^2 of the frozen CLIP-RN50 / SigLIP image embedding, as in the paper's script.
Surrogate: SGDClassifier(log_loss, elastic net, l1_ratio=0.99, max_iter=10000, random_state=SEED) on the concept
activations of N_FIT training images, labels = the black box's argmax predictions; lam from the paper's grid
{1e-7, 1e-6, 1e-5, 1e-4, 2e-4, 0.01 / (2 n_concepts)}, chosen by fidelity on N_HOLD held-out TRAINING images (the
paper used its validation split; the FunnyBirds test images are kept for the evaluation).

Local explanation (the per-image reading of a PCBM): PI(part, class) = sum over the part's concepts of
w[class, concept] * activation(image, concept). Important parts and background objects as in hns_explainer.py.
"""
import random
import zlib

import fb_data
import numpy as np
import torch
from omegaconf import OmegaConf
from paths import PARTS, QUERY_TEXT
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

SEED, N_CAV, N_FIT, N_HOLD, N_BG = 42, 50, 20_000, 5_000, 50
CLIP_CFGS = {"siglip": {"model_name": "ViT-B-16-SigLIP", "pretrained": "webli"},
             "clip_rn50": {"model_name": "RN50", "pretrained": "openai"}}


class PCBMExplainer:
    def __init__(self, model, variant, device="cuda"):
        assert variant in ("conventional", *CLIP_CFGS)
        self.model, self.variant, self.device = model, variant, device
        self._feat_cache: dict[int, np.ndarray] = {}
        idx = list(range(len(fb_data.train_params()))); random.Random(SEED).shuffle(idx)
        fit_idx, hold_idx = idx[:N_FIT], idx[N_FIT:N_FIT + N_HOLD]
        if variant == "conventional":
            self.net = model.model
            self._fit_cavs(idx)
        else:
            from cards.pipeline import instantiate_encoder
            cfg = {"name": variant, "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                   "device": device, **CLIP_CFGS[variant]}
            self.enc = instantiate_encoder(OmegaConf.create({"encoder": cfg, "device": device}))
            c = self.enc.encode_text([QUERY_TEXT[p] for p in PARTS]).float().to(device)
            self.cvec, self.cnorm2 = c, (c ** 2).sum(1)
            self.concept_part = list(PARTS)
        x_fit, x_hold = self._acts_train(fit_idx), self._acts_train(hold_idx)
        y_fit, y_hold = self._labels(fit_idx), self._labels(hold_idx)
        nc = x_fit.shape[1]
        best = None
        for lam in sorted({1e-7, 1e-6, 1e-5, 1e-4, 2e-4, round(0.01 / (2 * nc), 10)}):
            clf = SGDClassifier(random_state=SEED, loss="log_loss", alpha=lam, l1_ratio=0.99,
                                penalty="elasticnet", max_iter=10000).fit(x_fit, y_fit)
            fid = float((clf.predict(x_hold) == y_hold).mean())
            print(f"  PCBM {variant} lam={lam:.1e}: fidelity train {(clf.predict(x_fit) == y_fit).mean():.3f} "
                  f"held-out {fid:.3f}, nonzero {(clf.coef_ != 0).mean():.1%}", flush=True)
            if best is None or fid > best[0]:
                best = (fid, lam, clf)
        self.fidelity, self.lam, self.clf = best
        self.w = self.clf.coef_  # (n_classes, n_concepts); classes in clf.classes_ order
        self.class_row = {int(c): i for i, c in enumerate(self.clf.classes_)}
        print(f"  PCBM {variant}: lam={self.lam:.1e}, held-out fidelity {self.fidelity:.3f}", flush=True)

    # ----- concept activations -----
    @torch.no_grad()
    def _penultimate(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        return torch.flatten(n.avgpool(x), 1)

    @torch.no_grad()
    def _emb(self, x):
        """Backbone embedding of a [B,3,256,256] batch in [0,1]."""
        if self.variant == "conventional":
            return self._penultimate(x.to(self.device)).float()
        from PIL import Image
        ims = [Image.fromarray((t.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)) for t in x]
        return self.enc.encode_images(ims).float().to(self.device)

    def _concept_acts(self, emb):
        if self.variant == "conventional":
            return ((emb @ self.cav_w.T + self.cav_b) / self.cav_norm).cpu().numpy()
        return ((emb @ self.cvec.T) / self.cnorm2).cpu().numpy()

    def _acts_train(self, idx):
        out = []
        for s in range(0, len(idx), 64):
            x = torch.stack([fb_data.train_image(i) for i in idx[s:s + 64]])
            out.append(self._concept_acts(self._emb(x)))
            if (s // 64) % 50 == 0:
                print(f"    activations {s + len(x)}/{len(idx)}", flush=True)
        return np.concatenate(out)

    @torch.no_grad()
    def _labels(self, idx):
        out = []
        for s in range(0, len(idx), 64):
            x = torch.stack([fb_data.train_image(i) for i in idx[s:s + 64]]).to(self.device)
            out.append(self.model(x).argmax(1).cpu().numpy())
        return np.concatenate(out)

    def _fit_cavs(self, idx):
        vm = fb_data.variant_matrix()
        concepts = fb_data.variant_concepts()
        rng = random.Random(SEED)
        ws, bs = [], []
        for j, (p, v) in enumerate(concepts):
            pos = [i for i in idx if vm[i, j]]; neg = [i for i in idx if not vm[i, j]]
            rng.shuffle(pos); rng.shuffle(neg)
            sel = pos[:N_CAV] + neg[:N_CAV]
            X = torch.cat([self._emb(torch.stack([fb_data.train_image(i) for i in sel[s:s + 50]]))
                           for s in range(0, len(sel), 50)]).cpu().numpy()
            y = np.r_[np.ones(N_CAV), np.zeros(N_CAV)]
            svm = SVC(C=0.1, kernel="linear").fit(X, y)
            ws.append(svm.coef_[0]); bs.append(svm.intercept_[0])
            print(f"  CAV {p}{v + 1:02d}: train acc {svm.score(X, y):.2f}", flush=True)
        self.cav_w = torch.tensor(np.stack(ws), dtype=torch.float32, device=self.device)
        self.cav_b = torch.tensor(bs, dtype=torch.float32, device=self.device)
        self.cav_norm = self.cav_w.norm(dim=1)
        self.concept_part = [p for p, _ in concepts]

    # ----- framework interface -----
    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)

    def get_part_importance(self, image, part_map, target, colors_to_part=None, with_bg=False):
        key = zlib.crc32((image[0].detach().cpu().numpy() * 255).round().astype(np.uint8).tobytes())
        if key not in self._feat_cache:
            self._feat_cache[key] = self._concept_acts(self._emb(image[:1].detach().cpu()))[0]
        a = self._feat_cache[key]
        t = int(target.reshape(-1)[0]) if torch.is_tensor(target) else int(target)
        row = self.w[self.class_row[t]] if t in self.class_row else np.zeros(len(a))
        contrib = row * a
        pi = {p: float(sum(contrib[j] for j, cp in enumerate(self.concept_part) if cp == p)) for p in PARTS}
        if with_bg:
            pi.update({f"bg_{i:03d}": 0.0 for i in range(N_BG)})
        return pi

    def get_important_parts(self, image, part_map, target, colors_to_part=None, thresholds=None, with_bg=False):
        pi = self.get_part_importance(image, part_map, target, colors_to_part, with_bg=with_bg)
        total = sum(max(v, 0.0) for v in pi.values())
        thresholds = self.get_p_thresholds() if thresholds is None else thresholds
        return [[k for k, v in pi.items() if total > 0 and v > total * t] for t in thresholds]
