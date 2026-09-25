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
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(9, 6.6),
        gridspec_kw={"wspace": 0.05, "hspace": 0.04},
    )
    for ax, path in zip(axes.flat, shown_paths):
        img = Image.open(path).convert("RGB")
        # aspect="auto" fills the whole axes cell -- with the default
        # "equal" aspect, imshow preserves each source photo's own pixel
        # aspect ratio and letterboxes it inside the cell, so the visible
        # gap between rows/columns is real hspace/wspace PLUS a variable
        # per-image letterbox strip, making the vertical gap look larger
        # and inconsistent even at hspace=0 -- prompted directly ("Reduce
        # the vertical space, it looks weird").
        ax.imshow(img, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#999999")
            spine.set_linewidth(1.5)

    # No in-figure title -- added separately in the main figure for font
    # consistency (prompted directly: "can we just get rid of the name on
    # top, I can add it in the main figure for font consistency"). Equal
    # margins on all four sides + equal wspace/hspace above give evenly
    # spaced images filling the whole box, replacing the old tight_layout
    # call whose rect reserved an asymmetric top strip for the title.
    fig.subplots_adjust(left=0.025, right=0.975, top=0.965, bottom=0.035)

    # outer border box, matching the old flowchart-panel style (title-bar
    # separator line removed along with the title -- nothing left to
    # separate it from)
    outer = plt.Rectangle((0.015, 0.015), 0.97, 0.97, transform=fig.transFigure,
                           fill=False, edgecolor="black", linewidth=4, zorder=10)
    fig.add_artist(outer)

    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
