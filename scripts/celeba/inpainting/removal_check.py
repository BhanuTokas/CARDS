# ruff: noqa: SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""Does the fill actually remove the concept? Independent detector: linear probes on OpenAI CLIP ViT-L/14 image embeddings
(not SigLIP, not the black box), trained on CelebAMask-HQ images OUTSIDE the GT set with CelebAMask-HQ attribute labels.
For every GT pair whose image is labelled with the concept, score: original, LaMa on the concept region (saved by lama_gt.py),
blur on the concept region (the blur GT's fill), LaMa on one equal-area random shape (first draw of lama_gt.py's control).
All variants go through the same 512-px + JPEG q95 round trip as the saved LaMa images. Run from the repo root. argv[1]=out CSV."""
import csv
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, "src"); sys.path.insert(0, "scripts/celeba/ground_truth"); sys.path.insert(0, str(Path(__file__).parent))
from blob_control import random_blob
from lama_gt import DILATE, LAMA, RES, SAVE, make_inpaint
from open_clip.openai import load_openai_model
from paths import CLIP_L14
from run_celeba_official_train_faithfulness import CELEBA_HQ_ROOT, CONCEPT_TO_IDX, RESULTS_DIR, SEED
from run_celeba_official_train_strict_control import hq_index

from cards.data.celeba import load_celebamask_hq_image_paths, load_celebamask_hq_mask
from cards.data.celeba_attributes import ATTRIBUTE_TO_REGIONS, GROUNDABLE_CONCEPTS
from cards.validation.broden_faithfulness import mask_region

OUT = Path(sys.argv[1]) if __name__ == "__main__" else None; DEV = "cuda"; N_PROBE = 8000
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1); STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def roundtrip(im):  # same path as the saved LaMa images: 512 px LANCZOS, JPEG q95
    b = io.BytesIO(); im.resize((RES, RES), Image.LANCZOS).save(b, "JPEG", quality=95); b.seek(0)
    return Image.open(b).convert("RGB")


def prep(im):
    x = torch.from_numpy(np.asarray(im.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.).permute(2, 0, 1)
    return (x - MEAN) / STD


def main():
    lines = open(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt").read().splitlines()
    names = lines[1].split(); ci = [names.index(c) for c in GROUNDABLE_CONCEPTS]
    lab = {int(l.split()[0].split(".")[0]): np.array([int(v) > 0 for v in l.split()[1:]])[ci] for l in lines[2:]}
    gt_rows = list(csv.DictReader(open(RESULTS_DIR / "celeba_official_train_faithfulness_non_overlapping.csv", newline="")))
    pairs = sorted({(r["image"], r["concept_name"]) for r in gt_rows})
    import os
    if os.environ.get("RC_LIMIT"): pairs = pairs[:: len(pairs) // int(os.environ["RC_LIMIT"])][: int(os.environ["RC_LIMIT"])]
    paths = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    from hq_nulls import clean_hq_indices
    clean_idx, _ = clean_hq_indices(); clean_set = set(clean_idx)
    pool = sorted(i for i in paths if i not in clean_set)
    probe_idx = sorted(np.random.default_rng(0).choice(pool, N_PROBE, replace=False).tolist())
    clip = load_openai_model(str(CLIP_L14), precision="fp32", device=DEV).eval()

    @torch.no_grad()
    def emb(ims):
        e = clip.encode_image(torch.stack([prep(i) for i in ims]).to(DEV))
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()

    def embed_idx(idxs):
        out = []
        for s in range(0, len(idxs), 64):
            out.append(emb([roundtrip(Image.open(paths[i]).convert("RGB")) for i in idxs[s:s + 64]]))
            if s % 2048 == 0: print(f"   embedded {s}/{len(idxs)}", flush=True)
        return np.concatenate(out)

    E = embed_idx(probe_idx); Y = np.stack([lab[i] for i in probe_idx])
    fit, cal = slice(0, 6000), slice(6000, N_PROBE)
    probes, thr = [], []
    for n in range(len(ci)):
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(E[fit], Y[fit, n])
        s = clf.decision_function(E[cal]); fpr, tpr, t = roc_curve(Y[cal, n], s)
        probes.append(clf); thr.append(float(t[np.argmax(tpr - fpr)]))  # Youden threshold on held-out calibration split
    Ec = embed_idx(clean_idx); Yc = np.stack([lab[i] for i in clean_idx])
    det_auc = {c: roc_auc_score(Yc[:, n], probes[n].decision_function(Ec)) for n, c in enumerate(GROUNDABLE_CONCEPTS)}
    print("detector AUROC on the clean GT originals:", {c: round(a, 3) for c, a in det_auc.items()}, flush=True)

    inpaint = make_inpaint(torch.jit.load(str(LAMA), map_location=DEV).eval())
    f = open(OUT, "w", newline=""); w = csv.writer(f)
    w.writerow(["image", "concept_name", "hq_idx", "label", "detector_auroc", "threshold", "s_orig", "s_lama", "s_blur", "s_rand_lama"])
    for k, (img, c) in enumerate(pairs):
        idx = hq_index(img); n = GROUNDABLE_CONCEPTS.index(c)
        image = Image.open(paths[idx]).convert("RGB"); hw = (image.height, image.width)
        mask = load_celebamask_hq_mask(CELEBA_HQ_ROOT, idx, ATTRIBUTE_TO_REGIONS[c], target_hw=hw)
        dmask = binary_dilation(mask, iterations=DILATE) if mask.any() else mask
        rng = np.random.default_rng(SEED + idx * 100 + CONCEPT_TO_IDX[c] + 7)
        blob = random_blob(dmask, rng)  # first draw of lama_gt.py's control
        blur = mask_region(image, mask, strategy="blur", rng=np.random.default_rng(0))
        ims = [roundtrip(image), Image.open(SAVE / f"{idx}_{c}.jpg").convert("RGB"), roundtrip(blur)]
        if blob is not None: ims.append(roundtrip(inpaint(image, blob)))
        s = probes[n].decision_function(emb(ims))
        w.writerow([img, c, idx, int(lab[idx][n]), f"{det_auc[c]:.4f}", thr[n], s[0], s[1], s[2], s[3] if blob is not None else ""])
        if (k + 1) % 250 == 0: f.flush(); print(f"  {k + 1}/{len(pairs)}", flush=True)
    f.close(); print("done", flush=True)


if __name__ == "__main__":
    main()
