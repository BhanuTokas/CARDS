"""CUB analogue of scripts/celeba/analysis/visualize_pseudo_mask_examples.py
-- shows v67's ported SigLIP patch-similarity pseudo-mask pipeline end to
end for a handful of CUB attributes: original image, mask overlay, and
the blur/zero_fill masked results. Picks 4 attributes spanning different
body regions (bill, wing, crown, belly) for visual variety.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "celeba" / "analysis"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from localize_concept_patches_celeba import patch_similarity_grid, upsample_to_mask
from run_cards_cub_attributes import PREFIX_TEMPLATES

from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.cub_attributes import load_attribute_names
from cards.data.cub_parts import load_images_txt
from cards.pipeline import instantiate_encoder
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import mask_region

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
ATTRIBUTE_NAMES_PATH = CUB_ROOT / "attributes" / "new_attributes.txt"
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
TOP_PCT = 15
ATTRIBUTES_TO_SHOW = ["has_bill_shape::dagger", "has_wing_color::brown", "has_crown_color::black", "has_belly_color::yellow"]
N_EXAMPLES_PER_ATTR = 2


def readable(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


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

    attribute_names = load_attribute_names(ATTRIBUTE_NAMES_PATH)
    name_to_idx = {name: i for i, name in enumerate(attribute_names)}

    image_paths = load_images_txt(CUB_ROOT)
    class_labels = {}
    for line in (CUB_ROOT / "image_class_labels.txt").read_text().splitlines():
        image_id, class_id = line.split()
        class_labels[image_id] = int(class_id) - 1
    test_ids = [
        line.split()[0]
        for line in (CUB_ROOT / "train_test_split.txt").read_text().splitlines()
        if line.split()[1] == "0"
    ]
    pairs = [(image_paths[i], class_labels[i]) for i in test_ids]

    cfg.dataset = {"name": "cub", "root": str(CUB_ROOT)}
    cfg.pool_source = "test"
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)

    n_rows = len(ATTRIBUTES_TO_SHOW) * N_EXAMPLES_PER_ATTR
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3 * n_rows))

    row = 0
    for attr_name in ATTRIBUTES_TO_SHOW:
        prefix, value = attr_name.split("::", 1)
        query_text = PREFIX_TEMPLATES[prefix].format(value=readable(value))
        t_c = build_concept_query(query_text, encoder)
        t_c = demean_query(t_c, text_center)
        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)

        for i in range(N_EXAMPLES_PER_ATTR):
            idx = present_indices[i * 7]
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
            axes[row, 0].set_ylabel(f"{attr_name}\n(#{int(idx)})", fontsize=9)
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
    out_path = RESULTS_DIR / "cub_pseudo_mask_examples.png"
    plt.savefig(out_path, dpi=110)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
