"""CAV fit for the official-train classifier, prompted directly ("Do we
also have results of PCBM and TCAV on this version of ground truth?" --
they didn't exist yet, building them now).

**Switched from region crops to WHOLE IMAGES** (attribute-filtered from
official CelebA train), confirmed directly ("Switch PCBM conventional's
concept source to whole images for the new architectures?" -> "Yes,
whole images for all 3 (redo ResNet18 too)") -- matches `run_pcbm_
shortcut_experiment_official_train.py`'s own whole-image design
(MAX_PER_CLASS=150 candidates, N_CAV_SAMPLES=50 used) rather than the
static region-crop bank (`Datasets/celeba_full_concepts/`) this script
used before. Two real consequences: (1) this classifier's "conventional
PCBM" concept source is now consistent across ALL THREE architectures
(previously only true going forward, with ResNet18's own already-
reported result left on crops -- redone here instead, replacing that
result, not leaving the inconsistency); (2) the region-crop bank is no
longer a dependency anywhere in the official-train pipeline, and
CelebAMask-HQ is no longer needed for THIS script specifically (still
needed elsewhere in the pipeline -- ground truth generation and the
masking hybrid's leakage check both still require real segmentation
masks, which only CelebAMask-HQ has; that's unrelated to PCBM's own
concept source and doesn't go away).

**Generalized to any official-train backbone** (`CARDS_BACKBONE_NAME`
env var, default `celeba_official_train_attractive_male`) -- already
used `spec.feature_extractor(native_model)` generically (the exact-
reconstruction ViT/ConvNeXt extractors verified earlier when those
backbones were registered), so this only needed the backbone-name
lookup parameterized, plus OUT_DIR/filenames keyed by BACKBONE_NAME so
each architecture's CAVs land in their own directory.
"""

from __future__ import annotations

import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from concepts.concept_utils import ListDataset, learn_concept_bank
from torch.utils.data import DataLoader

from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.models.backbones import BACKBONES

CELEBA_ROOT = Path(os.environ.get("CELEBA_ROOT", r"C:\Users\btokas\Projects\Datasets\CelebA\celeba"))
BACKBONE_NAME = os.environ.get("CARDS_BACKBONE_NAME", "celeba_official_train_attractive_male")
OUT_DIR = Path(os.environ.get("CARDS_TRAINED_CONCEPTS_DIR", "trained_concepts_new/celeba_full")) / BACKBONE_NAME
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 50  # needs 2*50=100 pos/neg whole-image exemplars per concept
MAX_PER_CLASS = 150  # matches run_pcbm_shortcut_experiment_official_train.py's own MAX_PER_CLASS
C_VALUES = [0.01, 0.1]
BATCH_SIZE = 25
SEED = 42


class FlattenedFeatureExtractor(nn.Module):
    def __init__(self, feature_extractor: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, x):
        x = self.feature_extractor(x)
        return torch.flatten(x, 1)


def load_attr_names(path: Path) -> list[str]:
    return path.read_text().splitlines()[1].split()


def load_attr_labels(path: Path) -> dict[str, np.ndarray]:
    lines = path.read_text().splitlines()
    result: dict[str, np.ndarray] = {}
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split()
        result[parts[0]] = np.array([v == "1" for v in parts[1:]], dtype=bool)
    return result


def main():
    torch.manual_seed(SEED)
    rng_py = random.Random(SEED)
    print(f"BACKBONE_NAME={BACKBONE_NAME}  OUT_DIR={OUT_DIR}", flush=True)
    spec = BACKBONES[BACKBONE_NAME]
    native_model = spec.load_native().to(DEVICE)
    backbone = FlattenedFeatureExtractor(spec.feature_extractor(native_model)).to(DEVICE).eval()

    print("Loading official CelebA metadata...", flush=True)
    attr_names = load_attr_names(CELEBA_ROOT / "list_attr_celeba.txt")
    attr_labels = load_attr_labels(CELEBA_ROOT / "list_attr_celeba.txt")
    partition = {}
    with open(CELEBA_ROOT / "list_eval_partition.txt") as f:
        for line in f:
            fname, part = line.split()
            partition[fname] = part
    train_files = sorted(f for f, p in partition.items() if p == "0")
    all_train_paths = [CELEBA_ROOT / "img_align_celeba" / f for f in train_files]
    print(f"{len(all_train_paths)} official-train images.", flush=True)

    concept_libs = {C: {} for C in C_VALUES}
    for concept_name in GROUNDABLE_CONCEPTS:
        concept_attr_idx = attr_names.index(concept_name)
        pos_all = [p for p in all_train_paths if attr_labels[p.name][concept_attr_idx]]
        neg_all = [p for p in all_train_paths if not attr_labels[p.name][concept_attr_idx]]
        rng_py.shuffle(pos_all)
        rng_py.shuffle(neg_all)
        pos_paths = pos_all[:MAX_PER_CLASS]
        neg_paths = neg_all[:MAX_PER_CLASS]
        print(f"{concept_name}: {len(pos_paths)} positive / {len(neg_paths)} negative whole-image candidates", flush=True)
        if min(len(pos_paths), len(neg_paths)) < 2 * N_SAMPLES:
            raise ValueError(f"{concept_name} has too few whole images for n_samples={N_SAMPLES}")

        pos_loader = DataLoader(ListDataset(pos_paths, spec.preprocess), batch_size=BATCH_SIZE, shuffle=False)
        neg_loader = DataLoader(ListDataset(neg_paths, spec.preprocess), batch_size=BATCH_SIZE, shuffle=False)

        cav_info = learn_concept_bank(pos_loader, neg_loader, backbone, N_SAMPLES, C_VALUES, device=DEVICE)
        for C in C_VALUES:
            concept_libs[C][concept_name] = cav_info[C]
            print(f"  C={C}: train_acc={cav_info[C][1]:.3f}, test_acc={cav_info[C][2]:.3f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for C in C_VALUES:
        out_path = OUT_DIR / f"celeba_full_{BACKBONE_NAME}_{C}_{2 * N_SAMPLES}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(concept_libs[C], f)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
