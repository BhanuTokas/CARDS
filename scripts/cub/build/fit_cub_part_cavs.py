"""Fits CAVs for the 8 CUB part concepts materialized by
build_cub_part_concept_bank.py (real CUB70 mask crops, contrastive
negatives from other parts), against resnet18_cub's embedding space --
the CUB-track analogue of the ImageNet track's Broden CAV fitting.

Calls post_hoc_cbm's own learn_concept_bank/get_cavs directly (not the
learn_concepts_dataset.py CLI), since that CLI's data loading is hardwired
to the official 112-attribute CUB concept bank (data/concept_loaders.py's
cub_concept_loaders) -- bypassing it avoids touching post_hoc_cbm's own
constants for what is a separate, new concept source.

Output pickle format matches post_hoc_cbm's own convention exactly
({concept_name: (cav_vector, train_acc, test_acc, intercept, margin_info)}
per C), so it's a drop-in ConceptBank/PosthocLinearCBM input.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from concepts.concept_utils import ListDataset, learn_concept_bank
from torch.utils.data import DataLoader

from cards.models.backbones import BACKBONES

CONCEPT_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\cub_part_concepts")
OUT_DIR = Path("trained_concepts_new/cub_parts/resnet18_cub")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 50  # matches post_hoc_cbm's own CUB default (learn_concepts_dataset.py) -- needs 2*50=100 pos/neg per concept; right_eye's 112 positives is the tightest fit
C_VALUES = [0.01, 0.1]
BATCH_SIZE = 25  # chosen so no crop-count remainder is 1 (avoids get_embeddings' .squeeze() collapsing a size-1 batch to 1-D and breaking np.concatenate)
SEED = 42


class FlattenedFeatureExtractor(nn.Module):
    """Matches post_hoc_cbm's own ResNetBottom exactly (features stack,
    last `output` child dropped, explicit flatten) -- so embeddings here
    are directly comparable to the official 112-concept bank's own."""

    def __init__(self, feature_extractor: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, x):
        x = self.feature_extractor(x)
        return torch.flatten(x, 1)


def main():
    torch.manual_seed(SEED)
    spec = BACKBONES["resnet18_cub"]
    native_model = spec.load_native().to(DEVICE)
    backbone = FlattenedFeatureExtractor(spec.feature_extractor(native_model)).to(DEVICE).eval()

    concept_libs = {C: {} for C in C_VALUES}
    for concept_dir in sorted(CONCEPT_ROOT.iterdir()):
        part_name = concept_dir.name
        pos_paths = sorted((concept_dir / "positives").glob("*.jpg"))
        neg_paths = sorted((concept_dir / "negatives").glob("*.jpg"))
        print(f"{part_name}: {len(pos_paths)} positives, {len(neg_paths)} negatives", flush=True)
        if min(len(pos_paths), len(neg_paths)) < 2 * N_SAMPLES:
            raise ValueError(f"{part_name} has too few crops for n_samples={N_SAMPLES}")

        pos_loader = DataLoader(ListDataset(pos_paths, spec.preprocess), batch_size=BATCH_SIZE, shuffle=False)
        neg_loader = DataLoader(ListDataset(neg_paths, spec.preprocess), batch_size=BATCH_SIZE, shuffle=False)

        cav_info = learn_concept_bank(pos_loader, neg_loader, backbone, N_SAMPLES, C_VALUES, device=DEVICE)
        for C in C_VALUES:
            concept_libs[C][part_name] = cav_info[C]
            print(f"  C={C}: train_acc={cav_info[C][1]:.3f}, test_acc={cav_info[C][2]:.3f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for C in C_VALUES:
        out_path = OUT_DIR / f"cub_parts_resnet18_cub_{C}_{2 * N_SAMPLES}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(concept_libs[C], f)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
