"""Rebuilds the "Validation Dataset" flowchart box as a simpler "Image
Set" box, prompted directly ("can we have the same images as we have in
ranking... in random order, and rename it to simple image set"). Reuses
the exact same 6 CelebA-HQ images shown in build_ranking_along_vector_
figure.py's own ranking figure (ranks 1, 2, 3, 250, 2500, 4500 by
cosine similarity to the "smiling" concept vector), just shuffled and
relabeled, so the two flowchart panels visually reference the same
underlying pool.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "run"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.celeba import load_celebamask_hq_image_paths, split_celebamask_hq
from cards.data.celeba_attributes import TARGET_CLASSES, load_attribute_labels, load_attribute_names
from cards.pipeline import instantiate_encoder

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
OUT_PATH = Path("results/image_set_box.png")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONCEPT = "Smiling"
SHOWN_RANKS = [1, 2, 3, 250, 2500, None]  # same ranks as the ranking figure
SEED = 7


def main():
    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)
    query = demean_query(build_concept_query(CONCEPT_QUERY_TEXT[CONCEPT], encoder), text_center).to(DEVICE)

    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]
    _train_hq, val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices)
    val_paths = [image_paths_by_idx[i] for i in val_hq]

    cache_path = Path("results/.ranking_figure_embed_cache.pt")
    assert cache_path.exists(), "run build_ranking_along_vector_figure.py first to build this cache"
    embeds = torch.load(cache_path).to(DEVICE)

    sims = (embeds @ query).detach().cpu().numpy()
    order = sims.argsort()[::-1]

    n = len(order)
    shown_paths = []
    for r in SHOWN_RANKS:
        idx_in_order = (n - 1) if r is None else (r - 1)
        pool_idx = order[idx_in_order]
        shown_paths.append(val_paths[pool_idx])

    rng = random.Random(SEED)
    rng.shuffle(shown_paths)

    n_cols, n_rows = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 6.6))
    for ax, path in zip(axes.flat, shown_paths):
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#999999")
            spine.set_linewidth(1.5)

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.85])
    fig.suptitle("Image Set", fontsize=40, y=0.97)

    # title-bar box: outer border + separator line under the title, matching the old box style
    fig.canvas.draw()
    outer = plt.Rectangle((0.015, 0.015), 0.97, 0.965, transform=fig.transFigure,
                           fill=False, edgecolor="black", linewidth=4, zorder=10)
    fig.add_artist(outer)
    fig.add_artist(plt.Line2D([0.015, 0.985], [0.87, 0.87], transform=fig.transFigure, color="black", linewidth=4))

    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
