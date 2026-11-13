"""Choose the (concept, class) pairs of the COCO-Stuff local experiment and sample their present / absent records
(see common.py). CPU only; streams the 2.6 GB coco_stuff_faithfulness.csv once.

Pairs: concepts = the 103 scored concepts (coco_common.concept_list()); a pair is eligible when >= MIN_PRESENT
class-present images have the concept detected. Per class, three pairs (deduplicated): the largest mean ground truth
(mean delta_p over those images), the smallest, and the most detected images; plus the three cases of the existing
COCO local figures (greenness / blueness x sports ball, bottle x wine glass) when they have >= 20 images. The rule is
fixed before any method is scored on these records. env LOCAL_PAIRS replaces the rule with an explicit pair list.
Records per pair: N_PER_SET present (random, seeded) and N_PER_SET absent (class-present pool images where the concept
was not detected).

Writes $COCO_LOCAL_OUT/records.csv and pairs.csv."""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, "scripts/coco_stuff/baselines")
from coco_common import CLASSES, FAITH_CSV, concept_list
from common import COCO_ROOT, OUT, RECORDS, SEED

MIN_PRESENT = int(os.environ.get("LOCAL_MIN_PRESENT", "100"))  # env overrides only for synthetic smoke tests
N_PER_SET = int(os.environ.get("LOCAL_N_PER_SET", "100"))
MIN_CASE = 20
# Ground-truth variant: the ResNet18 file has delta_p_/delta_logit_ columns per masking strategy (black_mask = the
# concept's union mask filled black, black_bbox = its boxes); black_mask is the default of the COCO scorers.
GT_STRATEGY = os.environ.get("LOCAL_GT_STRATEGY", "black_mask")
CASES = [("greenness", "sports ball"), ("blueness", "sports ball"), ("bottle", "wine glass")]
# env LOCAL_PAIRS="class:concept;class:concept;..." replaces the selection rule by exactly these pairs (any number of
# present images), e.g. to recreate the paper's example figures.
PAIRS = [tuple(p.split(":")) for p in os.environ.get("LOCAL_PAIRS", "").split(";") if p.strip()]
for _k, _c in PAIRS:
    assert _k in CLASSES, f"unknown class {_k!r} in LOCAL_PAIRS"


def class_images() -> dict[str, set[str]]:
    """file name -> the target classes annotated in it (train2017, multi-label)."""
    with open(COCO_ROOT / "annotations" / "instances_train2017.json") as f:
        data = json.load(f)
    cid = {c["id"]: c["name"] for c in data["categories"] if c["name"] in CLASSES}
    names = {im["id"]: im["file_name"] for im in data["images"]}
    out: dict[str, set[str]] = defaultdict(set)
    for a in data["annotations"]:
        if a["category_id"] in cid:
            out[names[a["image_id"]]].add(cid[a["category_id"]])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    concepts = set(concept_list())
    cls_of = class_images()
    print(f"{len(cls_of)} train2017 images with a target class; {len(concepts)} concepts", flush=True)
    detected: dict[str, set[str]] = defaultdict(set)
    gt: dict[tuple[str, str], dict[str, tuple[float, float]]] = defaultdict(dict)
    header = pd.read_csv(FAITH_CSV, nrows=0).columns
    dp_col, dl_col = ((f"delta_p_{GT_STRATEGY}", f"delta_logit_{GT_STRATEGY}") if f"delta_p_{GT_STRATEGY}" in header
                      else ("delta_p", "delta_logit"))  # the single-strategy format the build script documents
    print(f"ground truth columns: {dp_col}, {dl_col}", flush=True)
    cols = ["concept_name", "image_path", "class_name", dp_col, dl_col]
    present_keys = {f"{im}|{k}" for im, ks in cls_of.items() for k in ks}
    for n, ch in enumerate(pd.read_csv(FAITH_CSV, usecols=cols, chunksize=5_000_000)):
        ch = ch.rename(columns={dp_col: "delta_p", dl_col: "delta_logit"})
        ch = ch[ch["concept_name"].isin(concepts)]
        ch = ch.assign(image=ch["image_path"].str.split(r"[\\/]").str[-1])
        for c, ims in ch.groupby("concept_name")["image"].unique().items():
            detected[c].update(ims)
        ch = ch[(ch["image"] + "|" + ch["class_name"]).isin(present_keys)]
        for c, im, k, dp, dl in zip(ch["concept_name"], ch["image"], ch["class_name"], ch["delta_p"], ch["delta_logit"]):
            gt[c, k].setdefault(im, (dp, dl))  # first row wins if the file has duplicates
        print(f"  chunk {n}: {len(gt)} (concept, class) pairs so far", flush=True)
    stats = {p: (len(v), float(np.mean([x[0] for x in v.values()]))) for p, v in gt.items()}
    chosen: dict[tuple[str, str], list[str]] = defaultdict(list)
    if PAIRS:  # explicit pair list instead of the selection rule
        for k, c in PAIRS:
            if (c, k) in stats:
                chosen[c, k].append("requested")
            else:
                print(f"  {c} x {k}: no class-present image with the concept detected; skipped", flush=True)
    for k in CLASSES if not PAIRS else []:
        el = [(p, s) for p, s in stats.items() if p[1] == k and s[0] >= MIN_PRESENT]
        if not el:
            print(f"  {k}: no eligible pair"); continue
        chosen[max(el, key=lambda x: x[1][1])[0]].append("max_gt")
        chosen[min(el, key=lambda x: x[1][1])[0]].append("min_gt")
        chosen[max(el, key=lambda x: (x[1][0], x[0]))[0]].append("most_images")
    for p in CASES if not PAIRS else []:
        if stats.get(p, (0,))[0] >= MIN_CASE:
            chosen[p].append("paper_case")
    pairs = sorted(chosen)
    print(f"{len(pairs)} pairs", flush=True)
    with open(OUT / "pairs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["concept_name", "class_name", "n_present_available", "mean_gt_delta_p",
                                       "n_absent_available", "reason"])
        with open(RECORDS, "w", newline="") as g:
            wr = csv.writer(g)
            wr.writerow(["set", "image", "concept_name", "class_name", "gt_delta_p", "gt_delta_logit"])
            for i, (c, k) in enumerate(pairs):
                rng = np.random.default_rng(SEED + i)
                pres = sorted(gt[c, k]); absent = sorted(im for im, ks in cls_of.items() if k in ks and im not in detected[c])
                for im in sorted(rng.choice(pres, min(N_PER_SET, len(pres)), replace=False)):
                    wr.writerow(["present", im, c, k, *gt[c, k][im]])
                for im in sorted(rng.choice(absent, min(N_PER_SET, len(absent)), replace=False)):
                    wr.writerow(["absent", im, c, k, 0.0, 0.0])
                w.writerow([c, k, len(pres), f"{stats[c, k][1]:.5f}", len(absent), "+".join(chosen[c, k])])
                print(f"  {c:22s} x {k:15s} present {len(pres):5d}  mean gt {stats[c, k][1]:+.4f}  "
                      f"absent {len(absent):6d}  {'+'.join(chosen[c, k])}", flush=True)
    print(f"wrote {RECORDS} and {OUT / 'pairs.csv'}", flush=True)


if __name__ == "__main__":
    main()
