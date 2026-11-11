"""ConceptSHAP baseline (Yeh et al., NeurIPS 2020) on BAM's obj and scene models.

Same design as the CelebA version, fixed in advance:
  - m=40 concept vectors on block_layer4 patch features (7x7x2048); concept score per
    patch = TH(cos(phi_p, c_j), beta=0.2), which zeroes values <= beta; g = per-patch MLP
    m -> 500 -> 2048 (ReLU), then the model's own average pool and linear head.
  - Loss = cross-entropy against the model's own predicted class - lambda * (coherence -
    diversity), lambda=0.1; coherence = mean top-32 patch cosines per concept in the batch,
    diversity = mean pairwise cos(c_j, c_k). Adam 1e-3, batch 256, 10 epochs, seed 0.
  - Naming: each of the 20 concepts gets the vector whose image-level presence (max patch
    cosine) has the highest train AUROC against BAM's label. Duplicates are allowed and reported.
  - Score(class k, concept c) = Shapley value of c's vector, by Monte-Carlo permutation
    sampling (300 permutations), for the value "mean class-k logit over held-out images
    containing c". Completeness (Yeh et al.'s eta) is reported as a check.

BAM differences from CelebA: training uses a stratified 10,000 of the 90,000 training
composites (100 per object-scene pair) and scoring a stratified 2,000 held-out ones
(20 per pair), because 7x7x2048 maps for every image would not fit in memory.

  --subsets-only  writes baselines/cshap_{train,val}.csv, for store_bam_outputs.py
                  (--activations --float16) to fill baselines/tf/<model>_cshap_{train,val}.npz.
Writes results/bam_concept_shap_scores.csv.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from fit_concept_sae_bam import linear_head  # noqa: E402
from run_hns_bam_masks import CONCEPTS  # noqa: E402
from sample_bam_concept_sets import OBJECTS, SCENES  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
M, BETA, LAM, TOPK, LR, BS, EPOCHS, N_PERM = 40, 0.2, 0.1, 32, 1e-3, 256, 10, 300
PER_PAIR = {"train": 100, "val": 20}
CLASSES = {"obj": OBJECTS, "scene": SCENES}
D = 2048


def write_subsets(bam_root: Path) -> None:
    rng = np.random.default_rng(0)
    for split, n in PER_PAIR.items():
        with open(bam_root / split / "manifest.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        pairs = defaultdict(list)
        for row in rows:
            pairs[row["object"], row["scene"]].append(row)
        chosen = [r for key in sorted(pairs) for r in rng.choice(pairs[key], size=n, replace=False)]
        with open(bam_root / "baselines" / f"cshap_{split}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(chosen)
        print(f"{split}: {len(chosen)} images", flush=True)


class CShap(torch.nn.Module):
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
        return self.g(v).mean(1)


def patches(acts: np.ndarray) -> torch.Tensor:
    """Flattened NHWC block_layer4 maps -> (N, 49, 2048) on the device."""
    return torch.from_numpy(acts.reshape(len(acts), 49, D)).to(DEV)


def labels(path: Path) -> np.ndarray:
    with open(path, newline="") as handle:
        return np.array([[r["object"] == c or r["scene"] == c for c in CONCEPTS] for r in csv.DictReader(handle)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam-root", type=Path, default=Path(os.environ.get("BAM_ROOT", "../Datasets/BAM")))
    parser.add_argument("--results-dir", type=Path, default=Path(os.environ.get("CARDS_RESULTS_DIR", "results")))
    parser.add_argument("--subsets-only", action="store_true")
    args = parser.parse_args()
    base = args.bam_root / "baselines"
    if args.subsets_only:
        write_subsets(args.bam_root)
        return
    y_train, y_val = labels(base / "cshap_train.csv"), labels(base / "cshap_val.csv")

    out_rows = []
    for model, classes in CLASSES.items():
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        full = np.load(base / "tf" / f"{model}_train.npz")
        w, b, err = linear_head(full["embeddings"], full["logits"])
        print(f"{model}: head recovered, max logit error {err:.2e}", flush=True)
        W, B = torch.tensor(w, device=DEV), torch.tensor(b, device=DEV)
        tr, va = np.load(base / "tf" / f"{model}_cshap_train.npz"), np.load(base / "tf" / f"{model}_cshap_val.npz")
        P, yt = patches(tr["activations"]), torch.tensor(tr["logits"].argmax(1), device=DEV)
        check = (P.float().mean(1) @ W.T + B - torch.tensor(tr["logits"], device=DEV)).abs().max().item()
        print(f"  pooled block_layer4 -> head reproduces logits, max error {check:.2e}", flush=True)

        net = CShap().to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        for ep in range(EPOCHS):
            perm = torch.randperm(len(P), device=DEV)
            tot = 0.0
            for s in range(0, len(P), BS):
                i = perm[s:s + BS]
                x = P[i].float()
                ce = Fn.cross_entropy(net.pooled(net.scores(x)) @ W.T + B, yt[i])
                coh = net.cos(x).reshape(-1, M).topk(TOPK, dim=0).values.mean()
                Cn = Fn.normalize(net.C, dim=0)
                G = Cn.T @ Cn
                div = (G.sum() - G.diag().sum()) / (M * (M - 1))
                loss = ce - LAM * (coh - div)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += ce.item() * len(i)
            print(f"  epoch {ep} CE {tot / len(P):.4f} coherence {coh.item():.3f} diversity {div.item():.3f}", flush=True)

        with torch.no_grad():
            pres = torch.cat([net.cos(P[s:s + 1024].float()).amax(1) for s in range(0, len(P), 1024)]).cpu().numpy()
            auc = np.array([[roc_auc_score(y_train[:, n], pres[:, j]) for n in range(len(CONCEPTS))] for j in range(M)])
            assign = auc.argmax(0)
            del P
            Pv = patches(va["activations"])
            V = torch.cat([net.scores(Pv[s:s + 1024].float()) for s in range(0, len(Pv), 1024)])
            del Pv
            yv = torch.tensor(va["logits"].argmax(1), device=DEV)
            share = torch.bincount(yv, minlength=10).float() / len(yv)
            a_r = share.max()
            Yv = torch.tensor(y_val, device=DEV).float()

            def value(mask):
                L = torch.cat([net.pooled(V[s:s + 1024] * mask) @ W.T + B for s in range(0, len(V), 1024)])
                eta = ((L.argmax(1) == yv).float().mean() - a_r) / (1 - a_r)
                return eta, (Yv.T @ L) / Yv.sum(0)[:, None]  # (concepts, classes)

            full_eta, _ = value(torch.ones(M, device=DEV))
            print(f"  completeness with all {M} vectors: {full_eta.item():.3f}", flush=True)
            sh_eta = torch.zeros(M, device=DEV)
            sh_sub = torch.zeros(M, len(CONCEPTS), 10, device=DEV)
            for n in range(N_PERM):
                order = rng.permutation(M)
                mask = torch.zeros(M, device=DEV)
                pe, ps = value(mask)
                for j in order:
                    mask[j] = 1
                    e, s_ = value(mask)
                    sh_eta[j] += e - pe
                    sh_sub[j] += s_ - ps
                    pe, ps = e, s_
                if n % 100 == 0:
                    print(f"  permutation {n}", flush=True)
            sh_eta /= N_PERM
            sh_sub /= N_PERM
        shared = defaultdict(list)
        for n, j in enumerate(assign):
            shared[int(j)].append(CONCEPTS[n])
        print(f"  vectors named for several concepts: {[v for v in shared.values() if len(v) > 1]}", flush=True)
        for n, concept in enumerate(CONCEPTS):
            j = int(assign[n])
            print(f"  {concept:14s} vector {j:2d} AUROC {auc[j, n]:.3f}", flush=True)
            for k, cls in enumerate(classes):
                out_rows.append({"model": model, "class": cls, "concept": concept, "vector": j,
                                 "train_auroc": round(float(auc[j, n]), 4), "shap_completeness": float(sh_eta[j]),
                                 "score": float(sh_sub[j, n, k])})

    args.results_dir.mkdir(exist_ok=True)
    with open(args.results_dir / "bam_concept_shap_scores.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print("done", flush=True)


if __name__ == "__main__":
    main()
