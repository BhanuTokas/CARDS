"""ConceptSAE baseline on BAM's obj and scene models (Discover-then-Name style).

Same design as the CelebA version, fixed in advance:
  - TopK SAE (Gao et al. 2024) on the model's pooled 2048-d features of all 90,000
    training composites: 4096 latents, k=32, unit-norm decoder columns, Adam lr 4e-4,
    batch 4096, 60 epochs, seed 0.
  - Naming: each of the 20 concepts gets the latent with the highest train AUROC
    against BAM's label (object or scene present). Supervised naming favours the baseline.
  - Score(class k, concept c) = mean over held-out images containing c of
    z_j(x) * (w_k . d_j): that latent's exact contribution to class k's logit
    through the linear head.

The head (w, b) is recovered from stored embeddings and logits by least squares.
Inputs: baselines/tf/<model>_{train,val}.npz from store_bam_outputs.py.
Writes results/bam_concept_sae_scores.csv.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
from run_hns_bam_masks import CONCEPTS  # noqa: E402
from sample_bam_concept_sets import OBJECTS, SCENES  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_LAT, K, LR, BS, EPOCHS = 4096, 32, 4e-4, 4096, 60
CLASSES = {"obj": OBJECTS, "scene": SCENES}


class TopKSAE(torch.nn.Module):
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


def linear_head(emb: np.ndarray, logits: np.ndarray) -> tuple:
    """(W, b) with logits = emb @ W.T + b, by least squares."""
    x = np.hstack([emb, np.ones((len(emb), 1))]).astype(np.float64)
    sol = np.linalg.lstsq(x, logits.astype(np.float64), rcond=None)[0]
    w, b = sol[:-1].T, sol[-1]
    err = np.abs(emb @ w.T + b - logits).max()
    return w.astype(np.float32), b.astype(np.float32), float(err)


def concept_labels(manifest: Path) -> np.ndarray:
    with open(manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return np.array([[r["object"] == c or r["scene"] == c for c in CONCEPTS] for r in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam-root", type=Path, default=Path(os.environ.get("BAM_ROOT", "../Datasets/BAM")))
    parser.add_argument("--results-dir", type=Path, default=Path(os.environ.get("CARDS_RESULTS_DIR", "results")))
    args = parser.parse_args()
    tf_dir = args.bam_root / "baselines" / "tf"
    y_train = concept_labels(args.bam_root / "train" / "manifest.csv")
    y_val = concept_labels(args.bam_root / "val" / "manifest.csv")

    rows = []
    for model, classes in CLASSES.items():
        torch.manual_seed(0)
        tr, va = np.load(tf_dir / f"{model}_train.npz"), np.load(tf_dir / f"{model}_val.npz")
        w, b, err = linear_head(tr["embeddings"], tr["logits"])
        print(f"{model}: head recovered, max logit error {err:.2e}", flush=True)
        X, Xv = torch.tensor(tr["embeddings"], device=DEV), torch.tensor(va["embeddings"], device=DEV)
        W, B = torch.tensor(w, device=DEV), torch.tensor(b, device=DEV)
        sae = TopKSAE(X.shape[1], N_LAT, K, X.median(0).values).to(DEV)
        opt = torch.optim.Adam(sae.parameters(), lr=LR)
        for ep in range(EPOCHS):
            perm = torch.randperm(len(X), device=DEV)
            tot = 0.0
            for s in range(0, len(X), BS):
                x = X[perm[s:s + BS]]
                xh, _ = sae(x)
                loss = ((xh - x) ** 2).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    sae.W_dec /= sae.W_dec.norm(dim=0, keepdim=True)
                tot += loss.item() * len(x)
            if ep % 10 == 0 or ep == EPOCHS - 1:
                print(f"  epoch {ep} mse {tot / len(X):.4f}", flush=True)
        with torch.no_grad():
            xh, zv = sae(Xv)
            fvu = (((xh - Xv) ** 2).sum() / ((Xv - Xv.mean(0)) ** 2).sum()).item()
            agree = ((Xv @ W.T + B).argmax(1) == (xh @ W.T + B).argmax(1)).float().mean().item()
            dead = (zv.sum(0) == 0).float().mean().item()
            print(f"  val FVU {fvu:.3f}; prediction agreement through SAE {agree:.3f}; dead latents {dead:.2f}",
                  flush=True)
            Z = torch.cat([sae.encode(X[s:s + 8192]).cpu() for s in range(0, len(X), 8192)]).numpy()
            zv = zv.cpu().numpy()
            contrib = (W @ sae.W_dec).cpu().numpy()  # (classes, latents): w_k . d_j
        Y = y_train.astype(float)
        npos = Y.sum(0)
        nneg = len(Y) - npos
        auc = np.zeros((N_LAT, len(CONCEPTS)))
        for s in range(0, N_LAT, 256):
            R = rankdata(Z[:, s:s + 256], axis=0)
            auc[s:s + 256] = (R.T @ Y - npos * (npos + 1) / 2) / (npos * nneg)
        for n, concept in enumerate(CONCEPTS):
            j = int(auc[:, n].argmax())
            act = zv[y_val[:, n], j].mean()
            print(f"  {concept:14s} latent {j:4d} AUROC {auc[j, n]:.3f}", flush=True)
            for k, cls in enumerate(classes):
                rows.append({"model": model, "class": cls, "concept": concept, "latent": j,
                             "train_auroc": round(float(auc[j, n]), 4),
                             "score": float(act * contrib[k, j]), "weight_score": float(contrib[k, j])})

    args.results_dir.mkdir(exist_ok=True)
    with open(args.results_dir / "bam_concept_sae_scores.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("done", flush=True)


if __name__ == "__main__":
    main()
