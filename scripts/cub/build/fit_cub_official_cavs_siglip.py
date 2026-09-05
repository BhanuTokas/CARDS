"""Refits PCBM's OFFICIAL 112-attribute CUB concept bank's CAVs in
SigLIP's own embedding space, instead of resnet18_cub's -- prompted
directly ("Can we also test PCBM with a SigLIP backbone?"), after
building the CLIP-concepts-with-SigLIP variant (train_pcbm_siglip_
concepts_cub.py) showed a large fidelity jump over CLIP-RN50 (41.1% vs
15.4% test fidelity). This tests whether SigLIP helps the OTHER PCBM
variant too -- the CAV-based one, using the same real per-image positive/
negative splits (post_hoc_cbm's own `cub_concept_loaders`, whole train
images, per-instance attribute labels) the official resnet18_cub CAV
bank was fit against, just embedded through SigLIP instead.

Same pattern as fit_cub_part_cavs.py (calls post_hoc_cbm's own
learn_concept_bank/get_cavs directly, backbone = any callable
image-batch -> embedding function -- SigLIP's own encode_image, wrapped
to accept a preprocessed batch tensor exactly like get_embeddings expects,
matching `OpenClipEncoder.encode_images`'s own L2-normalization
convention). Uses post_hoc_cbm's `get_concept_loaders("cub", ...)`
directly (NOT bypassed here, unlike the part-crop bank) since this is
explicitly re-fitting the SAME official concept definition/positive-
negative split on a new backbone, not a new concept source.

n_samples=50 (2*50=100 pos/neg per concept) matches post_hoc_cbm's own
CUB default and the existing resnet18_cub official bank
(`../post_hoc_cbm/trained_concepts_new/cub/resnet18_cub/
cub_resnet18_cub_{C}_100.pkl`), for a like-for-like comparison. C in
{0.01, 0.1}, matching fit_cub_part_cavs.py's own choice.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from concepts.concept_utils import learn_concept_bank
from data import get_concept_loaders
from omegaconf import OmegaConf

from cards.pipeline import instantiate_encoder

OUT_DIR = Path("trained_concepts_new/cub_official/siglip")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 50  # matches post_hoc_cbm's own CUB default / the existing resnet18_cub official bank
C_VALUES = [0.01, 0.1]
BATCH_SIZE = 25  # 2*N_SAMPLES=100 divides evenly -- avoids get_embeddings' .squeeze() collapsing a size-1 batch
SEED = 42


class SiglipImageBackbone:
    """Callable batch-tensor -> embedding wrapper around SigLIP's own
    encode_image, matching what get_embeddings expects (a plain callable
    taking a preprocessed image batch). Mirrors OpenClipEncoder.
    encode_images' own L2-normalization convention."""

    def __init__(self, siglip_encoder):
        self.model = siglip_encoder.model

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model.encode_image(batch), dim=-1)


def main():
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SigLIP (ViT-B-16-SigLIP/webli)...", flush=True)
    siglip_cfg = OmegaConf.create({"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE})
    siglip_encoder = instantiate_encoder(OmegaConf.create({"encoder": siglip_cfg, "device": DEVICE}))
    backbone = SiglipImageBackbone(siglip_encoder)

    print("Loading official 112-attribute concept loaders (post_hoc_cbm's own cub_concept_loaders)...", flush=True)
    concept_loaders = get_concept_loaders(
        "cub", siglip_encoder.preprocess, n_samples=N_SAMPLES, batch_size=BATCH_SIZE, num_workers=0, seed=SEED
    )
    print(f"{len(concept_loaders)} concepts loaded.", flush=True)

    concept_libs = {C: {} for C in C_VALUES}
    for i, (concept_idx, loaders) in enumerate(concept_loaders.items()):
        cav_info = learn_concept_bank(loaders["pos"], loaders["neg"], backbone, N_SAMPLES, C_VALUES, device=DEVICE)
        for C in C_VALUES:
            concept_libs[C][concept_idx] = cav_info[C]
        if i % 10 == 0 or i == len(concept_loaders) - 1:
            accs = {C: f"{cav_info[C][2]:.3f}" for C in C_VALUES}
            print(f"[{i + 1}/{len(concept_loaders)}] concept {concept_idx}: test_acc={accs}", flush=True)

    for C in C_VALUES:
        out_path = OUT_DIR / f"cub_official_siglip_{C}_{2 * N_SAMPLES}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(concept_libs[C], f)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
