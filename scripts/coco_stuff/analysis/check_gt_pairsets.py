"""Sanity check: the ViT / ConvNeXt ground-truth files must cover exactly the same (concept, image) pairs as the ResNet18
file (detection is model-independent), with 20 class rows each. Merges shards first if the merged file is missing.
Run from the repo root."""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

R = Path("results/coco_stuff")


def pairs(path: Path) -> pd.DataFrame:
    df = pd.concat(pd.read_csv(path, usecols=["concept_name", "image_path", "class_name"], chunksize=3_000_000,
                               dtype={"concept_name": "category", "class_name": "category"}))
    return df


def key_set(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["concept_name", "image_path"], observed=True).size()


def merged(bb: str) -> Path:
    m = R / f"coco_stuff_faithfulness_{bb}.csv"
    shards = sorted(glob.glob(str(R / f"coco_stuff_faithfulness_{bb}_shard*of*.csv")))
    if shards:
        with open(m, "w") as out:
            for i, s in enumerate(shards):
                lines = open(s).readlines()
                out.writelines(lines if i == 0 else lines[1:])
    return m


ref = key_set(pairs(R / "coco_stuff_faithfulness.csv"))
print(f"resnet18: {len(ref)} (concept, image) pairs, rows/pair min {ref.min()} max {ref.max()}")
for bb in ["coco_stuff_vit_b_16", "coco_stuff_convnext_tiny"]:
    df = pairs(merged(bb))
    k = key_set(df)
    only_ref = len(ref.index.difference(k.index)); only_bb = len(k.index.difference(ref.index))
    print(f"{bb}: {len(k)} pairs, rows/pair min {k.min()} max {k.max()}, missing vs resnet18 {only_ref}, "
          f"extra vs resnet18 {only_bb}, duplicate rows {int(df.duplicated(['concept_name', 'image_path', 'class_name']).sum())}")
