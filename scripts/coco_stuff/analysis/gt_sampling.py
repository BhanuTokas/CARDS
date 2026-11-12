"""Shared random ground-truth sampling for the COCO-Stuff scorers.

Replaces `compare_all_methods_faithfulness.load_faith`'s "first 100 images by sorted path" rule with a seeded random
draw of N_IMAGES images per concept, shared by every black box so the models are evaluated on identical images:

  1. For each black box, read its ground-truth CSV (all detected concept-image pairs, 20 class rows each).
  2. A concept is kept only if it has >= N_IMAGES images in EVERY black box's file (concept intersection).
  3. For each kept concept, draw N_IMAGES images at random (seeded) from the images present in all files, and use that
     same draw for every black box.

Returns, per strategy and backbone, (faith, image_counts) in the format the existing evaluators take:
  faith[(concept, class)] -> list of delta_p ;  image_counts[concept] -> N_IMAGES
Memory: reads only 4-5 columns, concept / class as categoricals."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

N_IMAGES = int(os.environ.get("GT_N_IMAGES", "100"))
SEED = int(os.environ.get("GT_SEED", "0"))


def _read(path: Path, strategies: tuple[str, ...]) -> pd.DataFrame:
    cols = ["concept_name", "image_path", "class_name"] + [f"delta_p_{s}" for s in strategies]
    parts = []
    for chunk in pd.read_csv(path, usecols=cols, chunksize=3_000_000,
                             dtype={"concept_name": "category", "class_name": "category"}):
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    df["concept_name"] = df["concept_name"].astype(str)
    df["class_name"] = df["class_name"].astype(str)
    return df


def load_gt_random(
    paths: dict[str, Path], strategies: tuple[str, ...], n_images: int = N_IMAGES, seed: int = SEED,
) -> tuple[dict[str, dict[str, tuple[dict, dict]]], dict]:
    """-> ({strategy: {backbone: (faith, image_counts)}}, info). `info` reports the concept and image bookkeeping."""
    dfs = {bb: _read(p, strategies) for bb, p in paths.items()}
    per_bb_images = {bb: df.groupby("concept_name")["image_path"].agg(lambda s: set(s)) for bb, df in dfs.items()}

    concepts = set.intersection(*[set(s.index) for s in per_bb_images.values()])
    shared: dict[str, list[str]] = {}
    for c in sorted(concepts):
        imgs = set.intersection(*[per_bb_images[bb][c] for bb in dfs])
        if len(imgs) >= n_images:
            shared[c] = sorted(imgs)
    rng = np.random.default_rng(seed)
    chosen_rows = []
    for c in sorted(shared):  # sorted -> the draw is reproducible
        pick = rng.choice(shared[c], size=n_images, replace=False)
        chosen_rows += [(c, str(p)) for p in pick]
    chosen = pd.DataFrame(chosen_rows, columns=["concept_name", "image_path"])

    out: dict[str, dict[str, tuple[dict, dict]]] = {s: {} for s in strategies}
    for bb, df in dfs.items():
        sub = df.merge(chosen, on=["concept_name", "image_path"], how="inner")
        for s in strategies:
            faith = {k: v.tolist() for k, v in sub.groupby(["concept_name", "class_name"])[f"delta_p_{s}"]}
            out[s][bb] = (faith, {c: n_images for c in shared})
    info = {
        "n_concepts": len(shared), "n_images": n_images, "seed": seed,
        "dropped_concepts": {bb: sorted(set(per_bb_images[bb].index) - set(shared)) for bb in dfs},
    }
    return out, info
