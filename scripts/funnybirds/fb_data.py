"""FunnyBirds training-set labels for the concept-based baselines (TCAV, PCBM).

Each training image's parameters (dataset_train.json) name the 3D model of every part, or "placeholder" when the
part was removed (about a quarter of the images per part). A present part's variant is its class's variant
(classes.json); variants are unnamed 3D models (beak01.glb, ...), so only conventional (CAV) concepts can use them.
"""
import json
from functools import lru_cache

import numpy as np
import torch
from paths import PARTS, ROOT
from PIL import Image


@lru_cache(maxsize=1)
def train_params() -> list[dict]:
    return json.loads((ROOT / "dataset_train.json").read_text())


@lru_cache(maxsize=1)
def classes() -> list[dict]:
    return json.loads((ROOT / "classes.json").read_text())


@lru_cache(maxsize=1)
def n_variants() -> dict[str, int]:
    return {p: len(v) for p, v in json.loads((ROOT / "parts.json").read_text()).items()}


def present(row: dict, part: str) -> bool:
    return row[f"{part}_model"] != "placeholder"


def presence_matrix() -> np.ndarray:
    """(n_train, 5) bool, part present, columns in PARTS order."""
    return np.array([[present(r, p) for p in PARTS] for r in train_params()])


def variant_concepts() -> list[tuple[str, int]]:
    """All (part, variant) concepts, e.g. ('beak', 0) .. ('wing', 5): 26 in total."""
    nv = n_variants()
    return [(p, v) for p in PARTS for v in range(nv[p])]


def variant_matrix() -> np.ndarray:
    """(n_train, 26) bool: image shows that part in that variant (present and its class's variant)."""
    cl = classes()
    concepts = variant_concepts()
    out = np.zeros((len(train_params()), len(concepts)), bool)
    for i, r in enumerate(train_params()):
        cv = cl[r["class_idx"]]["parts"]
        for j, (p, v) in enumerate(concepts):
            out[i, j] = present(r, p) and cv[p] == v
    return out


def train_image(i: int) -> torch.Tensor:
    """[3, 256, 256] float in [0, 1], as the framework loads images (alpha dropped)."""
    r = train_params()[i]
    a = np.asarray(Image.open(ROOT / "train" / str(r["class_idx"]) / f"{i:06d}.png"), dtype=np.float32)[..., :3]
    return torch.from_numpy(a / 255.0).permute(2, 0, 1)
