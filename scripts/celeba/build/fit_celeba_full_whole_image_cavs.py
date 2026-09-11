"""Whole-image PCBM variant -- fits CAVs for ALL 26 groundable concepts
directly from CelebAMask-HQ's own whole train images, filtered by the
concept's own real attribute label (True/False), instead of
fit_celeba_full_cavs.py's region-crop bank. Prompted directly ("What
about PCBM with whole images?") to resolve a known apples-to-oranges
asymmetry: TCAV in this track has always been whole-image (`run_tcav_
celeba_full.py`'s own docstring: "whole-image real-attribute-label
concept definition"), while PCBM has always been region-crop -- this
gives PCBM a matching whole-image variant for a fair comparison.

No crop-file materialization step needed (unlike the region-crop bank,
whole images already exist as normal files) -- positives/negatives are
just TRAIN-split image paths split by the concept's own attribute label,
fed straight into the same `learn_concept_bank` CAV-fitting machinery.
Negative definition: the SAME concept's attribute being False (the
natural whole-image complement), not "other concepts' crops" (that
contrastive-negative design in the region-crop bank was specific to
needing real, different visual content in a crop -- doesn't apply here).

Logic otherwise identical to fit_celeba_full_cavs.py (same N_SAMPLES=50,
C_VALUES, backbone, seed) so a whole-image-vs-region-crop comparison
isn't confounded by unrelated hyperparameter differences.
"""

from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from concepts.concept_utils import ListDataset, learn_concept_bank
from torch.utils.data import DataLoader

from cards.data.celeba import load_celebamask_hq_image_paths, split_celebamask_hq
from cards.data.celeba_attributes import (
    GROUNDABLE_CONCEPTS,
    TARGET_CLASSES,
    load_attribute_labels,
    load_attribute_names,
)
from cards.models.backbones import BACKBONES

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
OUT_DIR = Path("trained_concepts_new/celeba_full_whole_image/celeba_attractive_young")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 50  # needs 2*50=100 pos/neg per concept -- matches fit_celeba_full_cavs.py exactly
MAX_PER_CLASS = 150  # matches the region-crop bank's own MAX_POSITIVES/NEGATIVES_PER_CONCEPT
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


def main():
    torch.manual_seed(SEED)
    rng_py = random.Random(SEED)
    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE)
    backbone = FlattenedFeatureExtractor(spec.feature_extractor(native_model)).to(DEVICE).eval()

    print("Loading CelebAMask-HQ metadata...", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]
    train_hq, _ = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices)
    print(f"{len(train_hq)} train images available.", flush=True)
    print(f"{len(GROUNDABLE_CONCEPTS)} groundable concepts to build.", flush=True)

    concept_libs = {C: {} for C in C_VALUES}
    for concept_name in GROUNDABLE_CONCEPTS:
        concept_attr_idx = attr_names.index(concept_name)
        pos_hq = [i for i in train_hq if attr_labels_by_file[f"{i}.jpg"][concept_attr_idx]]
        neg_hq = [i for i in train_hq if not attr_labels_by_file[f"{i}.jpg"][concept_attr_idx]]
        rng_py.shuffle(pos_hq)
        rng_py.shuffle(neg_hq)
        pos_paths = [image_paths_by_idx[i] for i in pos_hq[:MAX_PER_CLASS]]
        neg_paths = [image_paths_by_idx[i] for i in neg_hq[:MAX_PER_CLASS]]
        print(f"{concept_name:<20s}: {len(pos_paths)} positives, {len(neg_paths)} negatives "
              f"(from {len(pos_hq)}/{len(neg_hq)} candidates)", flush=True)
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
        out_path = OUT_DIR / f"celeba_full_whole_image_celeba_attractive_young_{C}_{2 * N_SAMPLES}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(concept_libs[C], f)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
