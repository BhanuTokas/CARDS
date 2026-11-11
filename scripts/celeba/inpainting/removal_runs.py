# ruff: noqa: SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""Removal diagnostic for every inpainting run (diagnostic only; never used to select fills).
Same detector as removal_check.py: linear probes on OpenAI CLIP ViT-L/14 image embeddings, fit on 6,000 CelebAMask-HQ
images outside the GT set, Youden threshold on 2,000 more. For every GT pair, scores the original (512 px + JPEG q95
round trip, as the saved fills) and, for each run, the concept fill (m0) and the first random-shape fill (m1).
argv[1] = out CSV (long format: image, concept_name, hq_idx, label, detector_auroc, threshold, run, s_orig, s_m0, s_m1).
Run from the repo root."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, "src"); sys.path.insert(0, "scripts/celeba/ground_truth"); sys.path.insert(0, str(Path(__file__).parent))
from hq_nulls import clean_hq_indices
from open_clip.openai import load_openai_model
from removal_check import N_PROBE, prep, roundtrip
from run_celeba_official_train_faithfulness import CELEBA_HQ_ROOT

from cards.data.celeba import load_celebamask_hq_image_paths
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS

OUT = Path(sys.argv[1]) if __name__ == "__main__" else None; DEV = "cuda"
from paths import BUNDLE as B
from paths import CLIP_L14
from paths import FILLS as F

RUNS = ["lama_s0", "sd2_quality_s0", "sdxl_quality_s0", "flux_none_s0", "sd2_concept_s0", "sdxl_concept_s0"]


def main():
    lines = open(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt").read().splitlines()
    names = lines[1].split(); ci = [names.index(c) for c in GROUNDABLE_CONCEPTS]
    lab = {int(l.split()[0].split(".")[0]): np.array([int(v) > 0 for v in l.split()[1:]])[ci] for l in lines[2:]}
    paths = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    clean_idx, _ = clean_hq_indices(); clean_set = set(clean_idx)
    pool = sorted(i for i in paths if i not in clean_set)
    probe_idx = sorted(np.random.default_rng(0).choice(pool, N_PROBE, replace=False).tolist())  # same probe set as removal_check
    clip = load_openai_model(str(CLIP_L14), precision="fp32", device=DEV).eval()

    @torch.no_grad()
    def emb(ims):
        e = clip.encode_image(torch.stack([prep(i) for i in ims]).to(DEV))
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()

    def embed(loaders, tag):
        out = []
        for s in range(0, len(loaders), 64):
            out.append(emb([f() for f in loaders[s:s + 64]]))
            if s % 4096 == 0: print(f"   {tag}: embedded {s}/{len(loaders)}", flush=True)
        return np.concatenate(out)

    orig = lambda i: (lambda: roundtrip(Image.open(paths[i]).convert("RGB")))
    E = embed([orig(i) for i in probe_idx], "probe"); Y = np.stack([lab[i] for i in probe_idx])
    probes, thr = [], []
    for n in range(len(ci)):
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(E[:6000], Y[:6000, n])
        s = clf.decision_function(E[6000:]); fpr, tpr, t = roc_curve(Y[6000:, n], s)
        probes.append(clf); thr.append(float(t[np.argmax(tpr - fpr)]))
    man = [(int(r["hq_idx"]), r["concept_name"], int(r["n_masks"])) for r in csv.DictReader(open(B / "manifest.csv"))]
    idxs = sorted({i for i, _, _ in man}); Eo = embed([orig(i) for i in idxs], "originals"); row = {i: k for k, i in enumerate(idxs)}
    Yc = np.stack([lab[i] for i in idxs])
    auc = {c: roc_auc_score(Yc[:, n], probes[n].decision_function(Eo)) for n, c in enumerate(GROUNDABLE_CONCEPTS)}
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["hq_idx", "concept_name", "label", "detector_auroc", "threshold", "run", "s_orig", "s_m0", "s_m1"])
        for run in RUNS:
            ld = []
            for i, c, nm in man:
                ld.append(lambda i=i, c=c, run=run: Image.open(F / run / f"{i}_{c}_m0.jpg").convert("RGB"))
                ld.append((lambda i=i, c=c, run=run: Image.open(F / run / f"{i}_{c}_m1.jpg").convert("RGB")) if nm > 1 else None)
            keep = [k for k, l in enumerate(ld) if l is not None]
            Ef = np.full((len(ld), Eo.shape[1]), np.nan, dtype=np.float32); Ef[keep] = embed([ld[k] for k in keep], run)
            for p, (i, c, nm) in enumerate(man):
                n = GROUNDABLE_CONCEPTS.index(c); pr = probes[n]
                s0 = pr.decision_function(Eo[row[i]][None])[0]; sm0 = pr.decision_function(Ef[2 * p][None])[0]
                sm1 = pr.decision_function(Ef[2 * p + 1][None])[0] if nm > 1 else ""
                w.writerow([i, c, int(lab[i][n]), f"{auc[c]:.4f}", thr[n], run, s0, sm0, sm1])
            f.flush(); print(f"  {run} done", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
