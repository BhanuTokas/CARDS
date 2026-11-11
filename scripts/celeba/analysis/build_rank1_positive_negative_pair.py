"""Shows the real original + perturbed pair for the rank-1 "smiling"
image from the ranking figure, prompted directly ("the image and the
perturbed image for the first person in top-k"). Runs the actual
pipeline (patch-level localization, z-score cutoff over a real K=50
present set, best-of-7 perturbation selection by cosine alignment to
the concept vector) rather than a placeholder blur -- same methodology
as build_ranking_along_vector_figure.py / the paper's own Positive Set
/ Negative Set box figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "run"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.celeba import load_celebamask_hq_image_paths, split_celebamask_hq
from cards.data.celeba_attributes import TARGET_CLASSES, load_attribute_labels, load_attribute_names
from cards.pipeline import instantiate_encoder
from cards.validation.broden_faithfulness import mask_region

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
TARGET_RANK = 3
OUT_PATH = Path(f"results/rank{TARGET_RANK}_positive_negative_pair.png")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONCEPT = "Smiling"
K = 50
ALPHA = 2.0
SEED = 42
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]


def main():
    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)
    t_c = demean_query(build_concept_query(CONCEPT_QUERY_TEXT[CONCEPT], encoder), text_center).to(DEVICE)

    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]
    _train_hq, val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices)
    val_paths = [image_paths_by_idx[i] for i in val_hq]

    cache_path = Path("results/.ranking_figure_embed_cache.pt")
    assert cache_path.exists(), "run build_ranking_along_vector_figure.py first to build this cache"
    embeds = torch.load(cache_path).to(DEVICE)
    sims = (embeds @ t_c).detach().cpu().numpy()
    order = sims.argsort()[::-1]

    print(f"Localizing '{CONCEPT}' in the top-{K} present set (for a real z-score cutoff)...", flush=True)
    sim_maps = []
    images_topk = []
    for pool_idx in order[:K]:
        image = Image.open(val_paths[pool_idx]).convert("RGB")
        sim_map = localize_concept(encoder, image, t_c, (image.height, image.width))
        sim_maps.append(sim_map)
        images_topk.append(image)
    cutoff = concept_zscore_cutoff(sim_maps, ALPHA)
    print(f"z-score cutoff (alpha={ALPHA}): {cutoff:.4f}", flush=True)

    target_image = images_topk[TARGET_RANK - 1]
    target_sim_map = sim_maps[TARGET_RANK - 1]
    mask = threshold_mask(target_sim_map, method="fixed", cutoff=cutoff)
    print(f"rank-{TARGET_RANK} mask covers {mask.mean():.1%} of the image", flush=True)
    assert mask.any() and not mask.all(), f"rank-{TARGET_RANK} mask is degenerate -- pick a different example"

    rng = np.random.default_rng(SEED)
    candidates = [mask_region(target_image, mask, strategy=s, rng=rng) for s in FILL_STRATEGIES]
    with torch.no_grad():
        cand_embeds = encoder.encode_images([target_image] + candidates).to(DEVICE)
    embed_orig = cand_embeds[0]
    best_angle, best_i = None, None
    for i, strategy in enumerate(FILL_STRATEGIES):
        diff = embed_orig - cand_embeds[1 + i]
        diff_unit = diff / diff.norm()
        cos_sim = float(torch.clamp(diff_unit @ t_c, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_sim)))
        print(f"  {strategy:<16s} angle={angle_deg:6.2f} deg", flush=True)
        if best_angle is None or angle_deg < best_angle:
            best_angle, best_i = angle_deg, i
    best_strategy = FILL_STRATEGIES[best_i]
    perturbed_image = candidates[best_i]
    print(f"selected strategy: {best_strategy} (angle={best_angle:.2f} deg)", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    axes[0].imshow(target_image)
    axes[0].set_title(f"original (rank {TARGET_RANK})", fontsize=22)
    axes[1].imshow(perturbed_image)
    axes[1].set_title(f"perturbed ({best_strategy})", fontsize=22)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#333333")
            spine.set_linewidth(2)
    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_PATH}", flush=True)

    original_path = Path(f"results/rank{TARGET_RANK}_original.png")
    perturbed_path = Path(f"results/rank{TARGET_RANK}_perturbed_{best_strategy}.png")
    target_image.save(original_path)
    perturbed_image.save(perturbed_path)
    print(f"Saved {original_path}", flush=True)
    print(f"Saved {perturbed_path}", flush=True)


if __name__ == "__main__":
    main()
