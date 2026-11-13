"""Local (per-image) ConceptSAE on the COCO-Stuff local records (common.py), ResNet18 black box.

The SAE is fitted exactly as concept_sae_coco.py fits it (same module, hyperparameters, seed and order of random
calls, on the features of extract_feats_coco.py in $CARDS_FEAT_CACHE), then saved, because concept_sae_coco.py does
not save it; each concept's latent j = the latent with the highest AUROC against the concept's detection labels.
Local score of (image, concept, class) = z_j(x) * (w_class . d_j): the exact contribution of the concept's latent to
the class logit through the linear head, i.e. the per-image term whose mean over concept-positive images is
concept_sae_coco.py's global score.

argv: none. Writes $COCO_LOCAL_OUT/sae_resnet18.pt (reused if present), sae_global.csv and sae.csv."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, "scripts/coco_stuff/baselines")
import common as C
import concept_sae_coco as S  # seeds torch with 0 on import, as when concept_sae_coco.py runs
from coco_common import ARCH, CLASSES, DEVICE, FEAT_CACHE, feats, head, load_black_box

assert ARCH == "resnet18", "the local COCO experiment uses the ResNet18 black box"


def fit():
    d = torch.load(FEAT_CACHE / "pooled.pt", weights_only=False)
    X, W = d["pooled"].to(DEVICE), d["fc_w"].to(DEVICE)
    Y = d["labels"].numpy().astype(float); concepts = d["concepts"]
    sae = S.TopKSAE(X.shape[1], S.N_LAT, S.K, X.median(0).values).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=S.LR)
    for ep in range(S.EPOCHS):
        perm = torch.randperm(len(X), device=DEVICE); tot = 0
        for s in range(0, len(X), S.BS):
            x = X[perm[s:s + S.BS]]; xh, _ = sae(x)
            loss = ((xh - x) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                sae.W_dec /= sae.W_dec.norm(dim=0, keepdim=True)
            tot += loss.item() * len(x)
        if ep % 10 == 0 or ep == S.EPOCHS - 1:
            print(f"epoch {ep} mse {tot / len(X):.4f}", flush=True)
    with torch.no_grad():
        Z = torch.cat([sae.encode(X[s:s + 8192]).cpu() for s in range(0, len(X), 8192)]).numpy()
    npos = Y.sum(0); nneg = len(Y) - npos
    auc = np.zeros((S.N_LAT, Y.shape[1]))
    for s in range(0, S.N_LAT, 256):
        R = rankdata(Z[:, s:s + 256], axis=0)
        auc[s:s + 256] = (R.T @ Y - npos * (npos + 1) / 2) / (npos * nneg)
    latents = {c: int(auc[:, n].argmax()) for n, c in enumerate(concepts)}
    pos = Y.astype(bool)
    with open(C.OUT / "sae_global.csv", "w", newline="") as f:  # same columns as concept_sae_coco.py, for a check
        w = csv.writer(f); w.writerow(["concept_name", "class_name", "latent", "train_auroc", "n_pos", "score"])
        for n, c in enumerate(concepts):
            j = latents[c]; dj = sae.W_dec[:, j].detach(); zbar = float(Z[pos[:, n], j].mean())
            for k, cls in enumerate(CLASSES):
                w.writerow([c, cls, j, f"{auc[j, n]:.4f}", int(pos[:, n].sum()), zbar * float(W[k] @ dj)])
    ck = {"state": {k: v.cpu() for k, v in sae.state_dict().items()}, "k": S.K, "latents": latents}
    torch.save(ck, C.OUT / "sae_resnet18.pt")
    return ck


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    p = C.OUT / "sae_resnet18.pt"
    ck = torch.load(p) if p.exists() else fit()  # fit() trains, so it must run outside no_grad
    with torch.no_grad():
        score(ck)


def score(ck):
    st = ck["state"]
    sae = S.TopKSAE(st["W_dec"].shape[0], st["W_dec"].shape[1], ck["k"], st["b_dec"]).to(DEVICE)
    sae.load_state_dict(st)
    model, prep = load_black_box(); W = head(model).weight
    by_image = C.groups(C.records(), by=("image",))
    with open(C.OUT / "sae.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow([*C.BASE_COLS, "sae"])
        for s in range(0, len(by_image), 64):
            chunk = by_image[s:s + 64]
            x = torch.stack([prep(Image.open(C.image_path(n)).convert("RGB")) for (n,), _ in chunk]).to(DEVICE)
            z = sae.encode(feats(model, x)[0])
            for i, (_, rs) in enumerate(chunk):
                for r in rs:
                    j = ck["latents"][r["concept_name"]]
                    w.writerow([*C.base(r), float(z[i, j] * (W[CLASSES.index(r["class_name"])] @ sae.W_dec[:, j]))])
    print("done", flush=True)


if __name__ == "__main__":
    main()
