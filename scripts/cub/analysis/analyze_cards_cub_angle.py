"""Two follow-ups to v44's ablation sweep, both prompted directly:

1. "Can we test lower k? Maybe higher k leads to more irrelevant
   features getting mixed?" -- sweeps k in [5, 10, 15, 20, 30] (30 is
   the value used everywhere else, included as the reference point),
   SigLIP/no-demean/baseline-phrasing fixed (isolating k as the one
   variable), scored against the 87-attribute faithfulness ground truth.

2. "Can we look at the angle between p_c - n_c and text vector to see
   if that shows any information. Does it have any correlation with
   rho?" -- for each concept, the retrieved direction
   (mean(P_c) - mean(N_c), i.e. what Step 4 actually estimates) and the
   text query t_c used to RETRIEVE P_c/N_c in the first place are two
   different vectors in the same embedding space. If retrieval worked
   cleanly, they should point in roughly the same direction (small
   angle) -- P_c was chosen for high cosine similarity to t_c, N_c for
   low, so the group that actually separates them ought to align with
   what was asked for. A LARGE angle is a diagnostic: it means the
   retrieved P_c/N_c split's own dominant axis of difference is NOT the
   intended concept, i.e. something else (pose, lighting, a correlated
   attribute) dominates the split instead. Tests whether this per-concept
   angle predicts per-pair sign agreement with the faithfulness ground
   truth -- a real, checkable link between "how clean was retrieval for
   this concept" and "how much can we trust its score."
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_cards_cub_attributes import PREFIX_TEMPLATES  # noqa: E402

from cards.data.cub_attributes import groundable_attributes, load_attribute_names  # noqa: E402
from cards.data.cub_parts import load_images_txt  # noqa: E402
from cards.directions.estimate import estimate_direction  # noqa: E402
from cards.models.backbones import BACKBONES  # noqa: E402
from cards.pipeline import instantiate_encoder  # noqa: E402
from cards.retrieval.confound import matched_retrieval  # noqa: E402
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool  # noqa: E402
from cards.retrieval.retrieve import retrieve_top_bottom_k  # noqa: E402
from cards.concepts.prompts import build_concept_query  # noqa: E402
from cards.validation.broden_faithfulness import (  # noqa: E402
    FaithfulnessResult,
    _aggregate_faithfulness_pairs,
    score_method_agreement,
    score_sign_agreement,
)

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
ATTRIBUTE_NAMES_PATH = CUB_ROOT / "attributes" / "new_attributes.txt"
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANGLE_K = 15  # k used for the angle-vs-agreement analysis -- v44's best-performing k on this bank


def readable(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


def load_faithfulness_records() -> list[FaithfulnessResult]:
    records = []
    with open(RESULTS_DIR / "cub_attribute_faithfulness.csv", newline="") as f:
        for row in csv.DictReader(f):
            records.append(
                FaithfulnessResult(
                    image=row["image"], concept_number=int(row["concept_number"]), category=row["category"],
                    predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                    delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                    random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                    z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
                )
            )
    return records


def score_from_indices(spec, native_model, pool, present_indices, absent_indices) -> list[float]:
    present_paths = [pool.paths[i] for i in present_indices]
    absent_paths = [pool.paths[i] for i in absent_indices]
    present_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in present_paths]).to(DEVICE)
    absent_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in absent_paths]).to(DEVICE)
    with torch.no_grad():
        present_logits = native_model(present_batch)
        absent_logits = native_model(absent_batch)
    return (present_logits.mean(dim=0) - absent_logits.mean(dim=0)).tolist()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    faithfulness_records = load_faithfulness_records()
    print(f"{len(faithfulness_records)} attribute-level faithfulness records loaded (unchanged ground truth).", flush=True)

    attribute_names = load_attribute_names(ATTRIBUTE_NAMES_PATH)
    groundable = groundable_attributes(attribute_names)

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

    print("\nLoading SigLIP + pool...", flush=True)
    encoder_cfg = OmegaConf.create({"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                                     "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE})
    encoder = instantiate_encoder(OmegaConf.create({"encoder": encoder_cfg, "device": DEVICE}))
    pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": encoder_cfg, "cache_dir": "embedding_cache"})
    pool_cfg.dataset = {"name": "cub", "root": str(CUB_ROOT)}
    pool_cfg.pool_source = "test"
    pairs = [(image_paths[i], class_labels[i]) for i in test_ids]
    pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)

    spec = BACKBONES["resnet18_cub"]
    native_model = spec.load_native().to(DEVICE).eval()

    # ---- Part 1: lower-k sweep ----
    print("\n=== Part 1: k sweep (SigLIP, no demean, baseline phrasing) ===", flush=True)
    k_results = []
    for k in [5, 10, 15, 20, 30]:
        scores: dict[tuple[int, int], float] = {}
        for attr_idx, (prefix, _part_names) in groundable.items():
            value = attribute_names[attr_idx].split("::", 1)[1]
            text = PREFIX_TEMPLATES[prefix].format(value=readable(value))
            t_c = build_concept_query(text, encoder)
            present_indices, _ = retrieve_top_bottom_k(pool, t_c, k)
            absent_indices = matched_retrieval(pool, present_indices, t_c)
            raw = score_from_indices(spec, native_model, pool, present_indices, absent_indices)
            for native_idx, s in enumerate(raw):
                scores[(attr_idx, native_idx)] = s

        rho_result = score_method_agreement(faithfulness_records, scores, min_samples_per_pair=3)
        sign_result = score_sign_agreement(faithfulness_records, scores, min_samples_per_pair=3)
        k_results.append((k, rho_result, sign_result))
        if rho_result is None:
            print(f"[k={k}] too few pairs", flush=True)
        else:
            print(f"[k={k}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} p={rho_result.spearman_p:.4g} "
                  f"| sign={sign_result.agreement_frac:.1%} ({sign_result.n_agree}/{sign_result.n_pairs}) "
                  f"binom_p={sign_result.binom_p:.4g}", flush=True)

    with open(RESULTS_DIR / "cards_cub_k_sweep.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for k, rho_result, sign_result in k_results:
            if rho_result is None:
                writer.writerow([k, "", "", "", "", "", ""])
            else:
                writer.writerow([k, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])

    # ---- Part 2: angle(direction, t_c) vs. pair-level sign agreement, at ANGLE_K ----
    print(f"\n=== Part 2: angle(retrieved direction, text query) at k={ANGLE_K} ===", flush=True)
    angle_by_attr: dict[int, float] = {}
    scores_at_angle_k: dict[tuple[int, int], float] = {}
    for attr_idx, (prefix, _part_names) in groundable.items():
        value = attribute_names[attr_idx].split("::", 1)[1]
        text = PREFIX_TEMPLATES[prefix].format(value=readable(value))
        t_c = build_concept_query(text, encoder)
        present_indices, _ = retrieve_top_bottom_k(pool, t_c, ANGLE_K)
        absent_indices = matched_retrieval(pool, present_indices, t_c)

        direction = estimate_direction(
            str(attr_idx), pool.embeddings[present_indices], pool.embeddings[absent_indices]
        )
        cos_sim = float(torch.clamp(t_c @ direction.unit_vector, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_sim)))
        angle_by_attr[attr_idx] = angle_deg

        raw = score_from_indices(spec, native_model, pool, present_indices, absent_indices)
        for native_idx, s in enumerate(raw):
            scores_at_angle_k[(attr_idx, native_idx)] = s

    angles = np.array(list(angle_by_attr.values()))
    print(f"angle stats across {len(angle_by_attr)} concepts: "
          f"mean={angles.mean():.2f} deg, min={angles.min():.2f}, max={angles.max():.2f}, std={angles.std():.2f}",
          flush=True)

    aggregated = _aggregate_faithfulness_pairs(faithfulness_records, scores_at_angle_k, min_samples_per_pair=3)
    pair_angles = []
    pair_sign_match = []
    pair_rows = []
    for (attr_idx, class_idx), gt_delta in aggregated.items():
        angle = angle_by_attr[attr_idx]
        method_score = scores_at_angle_k[(attr_idx, class_idx)]
        match = int((gt_delta > 0) == (method_score > 0))
        pair_angles.append(angle)
        pair_sign_match.append(match)
        pair_rows.append((attr_idx, class_idx, angle, gt_delta, method_score, match))

    print(f"\n{len(pair_rows)} (concept, class) pairs with a faithfulness aggregate + score (k={ANGLE_K}).", flush=True)
    if len(pair_rows) >= 3:
        rho, p = spearmanr(pair_angles, pair_sign_match)
        print(f"Spearman(angle, sign_match) = {rho:+.4f} (p={p:.4g}) "
              f"-- negative means larger angle predicts LOWER sign agreement, as hypothesized", flush=True)

        agree_angles = [a for a, m in zip(pair_angles, pair_sign_match) if m == 1]
        disagree_angles = [a for a, m in zip(pair_angles, pair_sign_match) if m == 0]
        print(f"mean angle where signs AGREE: {np.mean(agree_angles):.2f} deg (n={len(agree_angles)})", flush=True)
        print(f"mean angle where signs DISAGREE: {np.mean(disagree_angles):.2f} deg (n={len(disagree_angles)})", flush=True)

    with open(RESULTS_DIR / "cards_cub_angle_analysis.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attribute_index", "class_idx", "angle_deg", "faithfulness_delta_p", "raw_score", "sign_match"])
        writer.writerows(pair_rows)

    print("\nSaved results/cards_cub_k_sweep.csv and results/cards_cub_angle_analysis.csv")


if __name__ == "__main__":
    main()
