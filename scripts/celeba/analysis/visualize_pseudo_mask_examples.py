"""Visual sanity check requested mid-run ("can I see some examples of the
concept masked images?") -- shows v78/v79's SigLIP patch-similarity
pseudo-mask pipeline end to end for a handful of concepts: original image,
mask overlay, and the blur/zero_fill masked results. Picks a mix of
concepts v78's negative control found genuinely well-localized
(Eyeglasses, Smiling, Wearing_Hat) plus one it flagged as confounded by
face-centering (Big_Nose), so the good and bad cases are both visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from localize_concept_patches_celeba import patch_similarity_grid, upsample_to_mask
from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.celeba_attributes import ATTRIBUTE_TO_REGIONS
from cards.data.datasets import load_celeba
from cards.pipeline import instantiate_encoder
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import mask_region

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
TOP_PCT = 15
CONCEPTS_TO_SHOW = ["Eyeglasses", "Smiling", "Wearing_Hat", "Big_Nose"]
N_EXAMPLES_PER_CONCEPT = 2


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
    })
    encoder = instantiate_encoder(cfg)
    siglip_model, siglip_preprocess = encoder.model, encoder.preprocess
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    cfg.dataset = {"name": "celeba", "root": str(CELEBA_HQ_ROOT)}
    cfg.pool_source = "val"
    pairs = load_celeba(CELEBA_HQ_ROOT, split="val")
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)

    n_rows = len(CONCEPTS_TO_SHOW) * N_EXAMPLES_PER_CONCEPT
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3 * n_rows))

    row = 0
    for concept_name in CONCEPTS_TO_SHOW:
        query_text = CONCEPT_QUERY_TEXT[concept_name]
        t_c = build_concept_query(query_text, encoder)
        t_c = demean_query(t_c, text_center)
        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)

        for i in range(N_EXAMPLES_PER_CONCEPT):
            idx = present_indices[i * 7]  # spread picks out a bit, not just the top-2
            image = Image.open(pool.paths[idx]).convert("RGB")
            sim_grid = patch_similarity_grid(siglip_model, siglip_preprocess, image, t_c)
            sim_map = upsample_to_mask(sim_grid, (image.height, image.width))
            thresh = np.percentile(sim_map, 100 - TOP_PCT)
            mask = sim_map >= thresh

            overlay = np.array(image).copy()
            overlay[mask] = (overlay[mask] * 0.3 + np.array([255, 40, 40]) * 0.7).astype(np.uint8)

            blurred = mask_region(image, mask, strategy="blur")
            zero_filled = mask_region(image, mask, strategy="zero_fill")

            axes[row, 0].imshow(image)
            axes[row, 0].set_ylabel(f"{concept_name}\n(#{int(idx)})", fontsize=9)
            axes[row, 1].imshow(overlay)
            axes[row, 2].imshow(blurred)
            axes[row, 3].imshow(zero_filled)
            for c in range(4):
                axes[row, c].set_xticks([])
                axes[row, c].set_yticks([])
            if row == 0:
                for c, title in enumerate(["original", "pseudo-mask (top-15%)", "blurred", "zero_fill"]):
                    axes[row, c].set_title(title, fontsize=10)
            row += 1

    plt.tight_layout()
    out_path = RESULTS_DIR / "celeba_pseudo_mask_examples.png"
    plt.savefig(out_path, dpi=110)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
