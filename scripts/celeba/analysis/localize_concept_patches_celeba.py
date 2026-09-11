"""Feasibility check for mask-free concept localization (option (1) from
the "how do we determine the mask for the masking hybrid" discussion):
reuse CARDS' own SigLIP encoder as a zero-shot localizer, instead of
CelebAMask-HQ's real per-pixel masks or a heavier open-vocab segmenter
(SAM/Grounded-SAM). No new model, no manual annotation -- just the same
encoder + the same demeaned text query CARDS' production run already
computes for each concept.

Mechanism: SigLIP's ViT (open_clip's `ViT-B-16-SigLIP`/webli) pools its
196 patch tokens through a learned `AttentionPoolLatent` (MAP head), not
a CLS token + linear projection the way plain CLIP works -- so a patch's
raw pre-pool hidden state does NOT live in the same space as the final
image/text embedding, and dotting it against a text query directly isn't
meaningful. Fix used here (verified empirically first, see conversation):
route each patch through that SAME learned attn_pool as if it were the
ONLY token in the sequence (a length-1 K/V sequence) -- softmax attention
over one token is trivially identity, so this deterministically produces
a per-patch vector in the exact final embedding space (confirmed:
averaging the whole-image call reproduces `encode_image` exactly,
cos=1.0; per-patch outputs are non-degenerate, pairwise cosine std=0.106,
not collapsed to a single point).

For each of the 8 pilot concepts (cheap first pass, not the full 26):
sample N=25 val images with a non-empty real region mask, compute the
14x14 per-patch similarity grid to the concept's own demeaned text query,
bilinear-upsample to the mask's native 512x512, and score it against the
REAL CelebAMask-HQ mask via ROC-AUC (threshold-free localization
quality -- 0.5 = chance ranking, 1.0 = perfect). This only checks whether
the localization idea is viable at all; it does NOT yet feed into any
faithfulness/hybrid scoring.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba import (
    load_celebamask_hq_image_paths,
    load_celebamask_hq_mask,
    split_celebamask_hq,
)
from cards.data.celeba_attributes import (
    ATTRIBUTE_TO_REGIONS,
    PILOT_CONCEPTS,
    TARGET_CLASSES,
    load_attribute_labels,
    load_attribute_names,
)
from cards.pipeline import instantiate_encoder

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PER_CONCEPT = 25


@torch.no_grad()
def patch_similarity_grid(model, preprocess, image: Image.Image, query_vec: torch.Tensor) -> np.ndarray:
    pixels = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    trunk = model.visual.trunk
    feats = trunk.forward_features(pixels)  # (1, N, C)
    n_patches = feats.shape[1]
    grid_side = round(n_patches**0.5)
    assert grid_side * grid_side == n_patches, f"non-square patch grid: {n_patches}"

    per_patch_in = feats[0].unsqueeze(1)  # (N, 1, C) -- each patch its own length-1 sequence
    per_patch_out = trunk.attn_pool(per_patch_in)  # (N, C), same space as encode_image's output
    per_patch_out = F.normalize(per_patch_out, dim=-1)

    sims = (per_patch_out @ query_vec.to(DEVICE)).cpu().numpy()
    return sims.reshape(grid_side, grid_side)


@torch.no_grad()
def patch_similarity_grid_linear_proj(model, preprocess, image: Image.Image, query_vec: torch.Tensor) -> np.ndarray:
    """For plain open_clip ViTs (CLIP, open_clip_h) -- unlike SigLIP's
    TimmModel wrapper (no standalone final projection, only the learned
    nonlinear AttentionPoolLatent -- patch_similarity_grid's own trick),
    these expose a genuinely separate LINEAR `visual.proj` applied to the
    pooled token after `visual.ln_post` (confirmed by reconstruction:
    `ln_post(x)[:,0] @ proj`, normalized, exactly reproduces
    `encode_image(..., normalize=True)`, cos=1.0). Since `ln_post` is
    applied elementwise to every token (not just the pooled one) BEFORE
    pooling, applying the SAME `proj` matrix directly to each patch token
    is architecturally exact, not an approximation -- no attn_pool-style
    trick needed here at all.
    """
    pixels = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    v = model.visual
    feats = v._embeds(pixels)
    feats = v.transformer(feats)
    ln = v.ln_post(feats)  # (1, N+1, C) -- token 0 is CLS, applied uniformly
    patch_tokens = ln[0, 1:]  # (N, C)
    n_patches = patch_tokens.shape[0]
    grid_side = round(n_patches**0.5)
    assert grid_side * grid_side == n_patches, f"non-square patch grid: {n_patches}"

    per_patch_out = F.normalize(patch_tokens @ v.proj, dim=-1)
    sims = (per_patch_out @ query_vec.to(DEVICE)).cpu().numpy()
    return sims.reshape(grid_side, grid_side)


@torch.no_grad()
def patch_similarity_grid_perception(model, preprocess, image: Image.Image, query_vec: torch.Tensor) -> np.ndarray:
    """For Meta's PE-Core (Perception encoder) -- same linear-projection
    logic as patch_similarity_grid_linear_proj (their own `visual.proj`
    is also a plain linear map applied after pooling, and
    `forward_features(norm=True)` applies `ln_post` elementwise to every
    token before pooling), using PE's own public `forward_features(norm=
    True, strip_cls_token=True)` helper instead of manually replicating
    the internal embed/transformer/ln_post chain.
    """
    pixels = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    v = model.visual
    patch_tokens = v.forward_features(pixels, norm=True, strip_cls_token=True)[0]  # (N, C)
    n_patches = patch_tokens.shape[0]
    grid_side = round(n_patches**0.5)
    assert grid_side * grid_side == n_patches, f"non-square patch grid: {n_patches}"

    per_patch_out = F.normalize(patch_tokens @ v.proj, dim=-1)
    sims = (per_patch_out @ query_vec.to(DEVICE)).cpu().numpy()
    return sims.reshape(grid_side, grid_side)


PATCH_SIMILARITY_FN = {
    "siglip": patch_similarity_grid,
    "clip": patch_similarity_grid_linear_proj,
    "open_clip_h": patch_similarity_grid_linear_proj,
    "perception_encoder": patch_similarity_grid_perception,
}


def upsample_to_mask(sim_grid: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    sim_t = torch.from_numpy(sim_grid).float()[None, None]
    upsampled = F.interpolate(sim_t, size=target_hw, mode="bilinear", align_corners=False)
    return upsampled[0, 0].numpy()


def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Per-image adaptive threshold (no external calibration, no fixed
    area budget) -- picks the cutoff maximizing between-class variance
    of the histogram, i.e. the split that best separates the score
    distribution into two clusters. Proposed as a fix for the fixed
    top-15% threshold's own scale-mismatch problem (v69: wildly
    oversized for CUB's small parts, calibrated for CelebA's much
    larger facial features instead). No skimage dependency in this env
    -- standard vectorized histogram-based Otsu, verified against a
    synthetic bimodal distribution before use.
    """
    hist, bin_edges = np.histogram(values, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_centers) / np.where(weight1 == 0, 1, weight1)
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.where(weight2[::-1] == 0, 1, weight2[::-1]))[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = int(np.argmax(variance12))
    return float(bin_centers[idx])


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rng_py = random.Random(SEED)

    print("Loading CelebAMask-HQ metadata...", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]
    _, val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices)
    print(f"{len(val_hq)} held-out val images.", flush=True)

    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
    })
    encoder = instantiate_encoder(cfg)
    model, preprocess = encoder.model, encoder.preprocess
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    rows = []  # (concept_name, hq_idx, auroc)
    for concept_name in PILOT_CONCEPTS:
        region_names = ATTRIBUTE_TO_REGIONS[concept_name]
        query_text = CONCEPT_QUERY_TEXT[concept_name]
        t_c = build_concept_query(query_text, encoder)
        t_c = demean_query(t_c, text_center)

        candidates = list(val_hq)
        rng_py.shuffle(candidates)

        aurocs = []
        for hq_idx in candidates:
            if len(aurocs) >= N_PER_CONCEPT:
                break
            mask = load_celebamask_hq_mask(CELEBA_HQ_ROOT, hq_idx, region_names)  # native 512x512
            if not mask.any() or mask.all():
                continue
            image = Image.open(image_paths_by_idx[hq_idx]).convert("RGB")
            sim_grid = patch_similarity_grid(model, preprocess, image, t_c)
            sim_map = upsample_to_mask(sim_grid, mask.shape)
            auroc = roc_auc_score(mask.flatten(), sim_map.flatten())
            aurocs.append(auroc)
            rows.append((concept_name, hq_idx, auroc))

        arr = np.array(aurocs)
        t_stat, p_val = stats.ttest_1samp(arr, 0.5)
        print(
            f"{concept_name:<20s} n={len(arr):>3d}  AUROC mean={arr.mean():.3f} std={arr.std():.3f}  "
            f"t-test vs 0.5: p={p_val:.4g}", flush=True,
        )

    all_aurocs = np.array([r[2] for r in rows])
    t_stat, p_val = stats.ttest_1samp(all_aurocs, 0.5)
    print(
        f"\nOVERALL  n={len(all_aurocs)}  AUROC mean={all_aurocs.mean():.3f} "
        f"std={all_aurocs.std():.3f}  t-test vs 0.5 chance: t={t_stat:.3f} p={p_val:.4g}", flush=True,
    )

    with open(RESULTS_DIR / "celeba_patch_localization_auroc.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "hq_idx", "auroc"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} (concept, image) AUROC rows to results/celeba_patch_localization_auroc.csv")


if __name__ == "__main__":
    main()
