"""Phase 7 scale-up of the CelebA plan: fits CAVs for ALL 26 groundable
concepts from build_celeba_full_concept_bank.py's real region-crop bank,
against celeba_attractive_young's own embedding space -- extends
fit_celeba_pilot_cavs.py's own 8-concept pilot fit (v71) the same way;
logic is otherwise identical, see that script's own docstring.
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

from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.models.backbones import BACKBONES

CONCEPT_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\celeba_full_concepts")
OUT_DIR = Path("trained_concepts_new/celeba_full/celeba_attractive_young")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 50  # needs 2*50=100 pos/neg per concept; every groundable concept has 150 of each
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
    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE)
    backbone = FlattenedFeatureExtractor(spec.feature_extractor(native_model)).to(DEVICE).eval()

    concept_libs = {C: {} for C in C_VALUES}
    for concept_name in GROUNDABLE_CONCEPTS:
        concept_dir = CONCEPT_ROOT / concept_name
        pos_paths = sorted((concept_dir / "positives").glob("*.jpg"))
        neg_paths = sorted((concept_dir / "negatives").glob("*.jpg"))
        print(f"{concept_name}: {len(pos_paths)} positives, {len(neg_paths)} negatives", flush=True)
        if min(len(pos_paths), len(neg_paths)) < 2 * N_SAMPLES:
            raise ValueError(f"{concept_name} has too few crops for n_samples={N_SAMPLES}")

        pos_loader = DataLoader(ListDataset(pos_paths, spec.preprocess), batch_size=BATCH_SIZE, shuffle=False)
        neg_loader = DataLoader(ListDataset(neg_paths, spec.preprocess), batch_size=BATCH_SIZE, shuffle=False)

        cav_info = learn_concept_bank(pos_loader, neg_loader, backbone, N_SAMPLES, C_VALUES, device=DEVICE)
        for C in C_VALUES:
            concept_libs[C][concept_name] = cav_info[C]
            print(f"  C={C}: train_acc={cav_info[C][1]:.3f}, test_acc={cav_info[C][2]:.3f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for C in C_VALUES:
        out_path = OUT_DIR / f"celeba_full_celeba_attractive_young_{C}_{2 * N_SAMPLES}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(concept_libs[C], f)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
