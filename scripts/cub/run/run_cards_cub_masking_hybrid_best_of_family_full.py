"""Ports the CelebA masking hybrid (v78-v84 in notes/celeba_correlation_
investigation.md) to CUB, prompted directly ("Can we run experiments on
CUB as well?"), scope confirmed directly (full 87-attribute bank,
SigLIP only). Same mechanism as `scripts/celeba/run/run_cards_celeba_
masking_hybrid_best_of_family_full.py`: instead of CARDS' usual
cross-image contrast (mean(b(P_c)) - mean(b(N_c)) over two DIFFERENT
retrieved image sets), for each retrieved present image compute a
pseudo-mask from SigLIP's own patch-token similarity to the attribute's
text query (v78's mechanism, reused verbatim from
`scripts/celeba/analysis/localize_concept_patches_celeba.py` -- fully
dataset-agnostic, only needs an encoder/image/query), mask that region
OUT of the SAME image via whichever of 7 fill strategies gives the
smallest angle to +t_c (v82's best-of-N selection), and measure the
CLASSIFIER's own per-class logit change on that single image.

Unlike CelebA (a 2-task binary classifier), CUB's resnet18_cub is a
200-way species classifier, so unlike the CelebA script this gets ALL
200 classes' delta_logit from a SINGLE forward pass per image (no
per-task loop needed) -- mirrors `run_cards_cub_attributes.py`'s own
`raw_score_all_classes` pattern exactly. Retrieval is per-attribute only
(no per-species retrieval), matching CARDS' existing CUB convention.

Reuses the CURRENT, true-max ground truth (`cub_attribute_faithfulness.
csv`, 17,619 records / 3,526 pairs, v62) completely unchanged -- no new
ground truth needed, since the hybrid's own pseudo-mask never touches
ground-truth generation, only its own same-image scoring.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "celeba" / "analysis"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from localize_concept_patches_celeba import PATCH_SIMILARITY_FN, upsample_to_mask
from run_cards_cub_attributes import PREFIX_TEMPLATES

from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.cub_attributes import groundable_attributes, load_attribute_names
from cards.data.cub_parts import load_images_txt
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    mask_region,
    score_method_agreement,
    score_sign_agreement,
)

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
ATTRIBUTE_NAMES_PATH = CUB_ROOT / "attributes" / "new_attributes.txt"
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
TOP_PCT = 15
SEED = 42
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]


def readable(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


def load_faithfulness_records() -> list[FaithfulnessResult]:
    records = []
    with open(RESULTS_DIR / "cub_attribute_faithfulness.csv", newline="") as f:
        for row in csv.DictReader(f):
            records.append(FaithfulnessResult(
                image=row["image"], concept_number=int(row["concept_number"]), category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return records


def load_siglip_baseline() -> dict[tuple[int, int], float]:
    scores = {}
    with open(RESULTS_DIR / "cards_cub_attribute_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            scores[(int(row["attribute_index"]), int(row["native_class_idx"]))] = float(row["raw_score"])
    return scores


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    faithfulness_records = load_faithfulness_records()
    print(f"{len(faithfulness_records)} faithfulness records loaded.", flush=True)

    attribute_names = load_attribute_names(ATTRIBUTE_NAMES_PATH)
    groundable = groundable_attributes(attribute_names)
    print(f"{len(groundable)}/{len(attribute_names)} attributes have a faithfulness ground truth.", flush=True)

    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
    })
    encoder = instantiate_encoder(cfg)
    siglip_model, siglip_preprocess = encoder.model, encoder.preprocess
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

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
    print(f"pool: {len(pool.paths)} images", flush=True)

    spec = BACKBONES["resnet18_cub"]
    native_model = spec.load_native().to(DEVICE).eval()

    hybrid_scores: dict[tuple[int, int], float] = {}
    raw_rows = []  # (attribute_index, attribute_name, attribute_prefix, native_class_idx, hybrid_raw_score)

    for i, (attr_idx, (prefix, _part_names)) in enumerate(groundable.items()):
        attr_name = attribute_names[attr_idx]
        value = attr_name.split("::", 1)[1]
        query_text = PREFIX_TEMPLATES[prefix].format(value=readable(value))
        t_c = build_concept_query(query_text, encoder)
        t_c = demean_query(t_c, text_center)
        t_c_dev = t_c.to(DEVICE)

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
        delta_logits = []  # list of (200,) tensors, one per successfully-masked image

        for idx in present_indices:
            image = Image.open(pool.paths[idx]).convert("RGB")
            sim_grid = PATCH_SIMILARITY_FN["siglip"](siglip_model, siglip_preprocess, image, t_c)
            sim_map = upsample_to_mask(sim_grid, (image.height, image.width))
            thresh = np.percentile(sim_map, 100 - TOP_PCT)
            mask = sim_map >= thresh
            if not mask.any() or mask.all():
                continue

            rng = np.random.default_rng(SEED + attr_idx * 10_000 + int(idx))
            candidates = [mask_region(image, mask, strategy=s, rng=rng) for s in FILL_STRATEGIES]
            with torch.no_grad():
                embeds = encoder.encode_images([image] + candidates).to(DEVICE)
            embed_orig = embeds[0]
            best_angle, best_i = None, None
            for j in range(len(FILL_STRATEGIES)):
                diff = embed_orig - embeds[1 + j]
                diff_unit = diff / diff.norm()
                cos_sim = float(torch.clamp(diff_unit @ t_c_dev, -1.0, 1.0))
                angle_deg = float(np.degrees(np.arccos(cos_sim)))
                if best_angle is None or angle_deg < best_angle:
                    best_angle, best_i = angle_deg, j

            masked_image = candidates[best_i]
            pixels_orig = spec.preprocess(image).unsqueeze(0)
            pixels_masked = spec.preprocess(masked_image).unsqueeze(0)
            batch = torch.cat([pixels_orig, pixels_masked], dim=0).to(DEVICE)
            with torch.no_grad():
                logits = native_model(batch)  # (2, 200)
            delta_logits.append((logits[0] - logits[1]).cpu())

        if delta_logits:
            mean_delta = torch.stack(delta_logits).mean(dim=0)  # (200,)
        else:
            mean_delta = torch.zeros(200)
        for native_idx in range(mean_delta.shape[0]):
            score = float(mean_delta[native_idx])
            hybrid_scores[(attr_idx, native_idx)] = score
            raw_rows.append((attr_idx, attr_name, prefix, native_idx, score))

        print(f"[{i + 1:>3d}/{len(groundable)}] {attr_name:<40s} n_images={len(delta_logits)}/{K}", flush=True)

    print("\n=== hybrid CARDS (same-image masking, best-of-7-strategy, SigLIP) vs. real ground truth ===", flush=True)
    rho_result = score_method_agreement(faithfulness_records, hybrid_scores, min_samples_per_pair=3)
    sign_result = score_sign_agreement(faithfulness_records, hybrid_scores, min_samples_per_pair=3)
    if rho_result is not None:
        print(f"hybrid: n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} p={rho_result.spearman_p:.4g} "
              f"| sign={sign_result.agreement_frac:.1%} ({sign_result.n_agree}/{sign_result.n_pairs}) "
              f"binom_p={sign_result.binom_p:.4g}", flush=True)

    siglip_baseline = load_siglip_baseline()
    rho_b = score_method_agreement(faithfulness_records, siglip_baseline, min_samples_per_pair=3)
    sign_b = score_sign_agreement(faithfulness_records, siglip_baseline, min_samples_per_pair=3)
    print(f"plain CARDS (v62 baseline, for reference): n={rho_b.n_pairs} rho={rho_b.spearman_rho:+.4f} "
          f"p={rho_b.spearman_p:.4g} | sign={sign_b.agreement_frac:.1%} ({sign_b.n_agree}/{sign_b.n_pairs}) "
          f"binom_p={sign_b.binom_p:.4g}", flush=True)

    with open(RESULTS_DIR / "cards_cub_masking_hybrid_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attribute_index", "attribute_name", "attribute_prefix", "native_class_idx", "hybrid_raw_score"])
        writer.writerows(raw_rows)
    print(f"\nSaved {len(raw_rows)} (attribute, class) hybrid scores to results/cards_cub_masking_hybrid_scores.csv")


if __name__ == "__main__":
    main()
