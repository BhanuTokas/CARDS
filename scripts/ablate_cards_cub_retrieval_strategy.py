"""Ablates Step 3's retrieval/matching strategy, prompted directly by
"Have we tried ablating the matching algorithm?" -- every CUB script so
far has used `matched_retrieval` (nearest-neighbor-after-projecting-out-
t_c) without ever testing the alternatives already built into the repo:

- naive: plain top-k/bottom-k by raw cosine similarity to t_c, no
  confound control at all (the "baseline ablation arm" per its own
  config comment).
- matched: the default used everywhere in this investigation so far.
- aligned: directly optimizes cos(mean(P_c) - mean(N_c), t_c) via greedy
  N_c search (cards.retrieval.aligned) -- literally the objective v45's
  angle diagnostic measured as bad (83 degrees average). Its own module
  docstring flags it was only ever validated against the OLD, abandoned
  CARDS-vs-PCBM-weight correlation (CIFAR-100) and never against a real
  ground truth -- faithfulness is exactly the validation signal it was
  waiting for.
- stratified: retrieves P_c/N_c independently within each of CUB's 200
  species (cards.retrieval.confound.stratified_retrieval) -- tests
  whether species-identity variance (CUB's dominant source of embedding
  structure, given how fine-grained the classification is) is part of
  what's swamping the intended concept direction. Needs k small enough
  to fit every stratum (`2*k <= min class test-split size`); checked
  directly: min=11 images/class on the test split, so k=5 is the largest
  feasible value across all 200 strata.

For naive/matched/aligned (same P_c, only N_c selection differs): k=15,
SigLIP, no demean, baseline phrasing -- matching v45's ANGLE_K so the
angle numbers are directly comparable. Reports both the per-concept
angle(direction, t_c) AND the faithfulness correlation for every
strategy, so it's clear whether shrinking the angle (if it happens)
actually moves the correlation, or whether they're decoupled.
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
from cards.retrieval.aligned import aligned_retrieval  # noqa: E402
from cards.retrieval.confound import matched_retrieval, stratified_retrieval  # noqa: E402
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool  # noqa: E402
from cards.retrieval.pool import CandidatePool  # noqa: E402
from cards.retrieval.retrieve import retrieve_top_bottom_k  # noqa: E402
from cards.concepts.prompts import build_concept_query  # noqa: E402
from cards.validation.broden_faithfulness import (  # noqa: E402
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
ATTRIBUTE_NAMES_PATH = CUB_ROOT / "attributes" / "new_attributes.txt"
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 15
STRATIFIED_K = 5


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


def evaluate(label, records, scores, angles, results):
    rho_result = score_method_agreement(records, scores, min_samples_per_pair=3)
    sign_result = score_sign_agreement(records, scores, min_samples_per_pair=3)
    results.append((label, rho_result, sign_result))
    angle_arr = np.array(list(angles.values()))
    angle_str = f"angle: mean={angle_arr.mean():.2f} std={angle_arr.std():.2f}"
    if rho_result is None:
        print(f"[{label}] too few pairs | {angle_str}", flush=True)
    else:
        print(f"[{label}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} p={rho_result.spearman_p:.4g} "
              f"| sign={sign_result.agreement_frac:.1%} ({sign_result.n_agree}/{sign_result.n_pairs}) "
              f"binom_p={sign_result.binom_p:.4g} | {angle_str}", flush=True)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    faithfulness_records = load_faithfulness_records()
    print(f"{len(faithfulness_records)} faithfulness records loaded.", flush=True)

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
    pool_with_labels = CandidatePool(paths=pool.paths, embeddings=pool.embeddings, labels=[label for _, label in pairs])

    spec = BACKBONES["resnet18_cub"]
    native_model = spec.load_native().to(DEVICE).eval()

    results = []

    for strategy in ["naive", "matched", "aligned"]:
        print(f"\n=== {strategy} (k={K}) ===", flush=True)
        scores: dict[tuple[int, int], float] = {}
        angles: dict[int, float] = {}
        for attr_idx, (prefix, _part_names) in groundable.items():
            value = attribute_names[attr_idx].split("::", 1)[1]
            text = PREFIX_TEMPLATES[prefix].format(value=readable(value))
            t_c = build_concept_query(text, encoder)

            present_indices, naive_absent = retrieve_top_bottom_k(pool, t_c, K)
            if strategy == "naive":
                absent_indices = naive_absent
            elif strategy == "matched":
                absent_indices = matched_retrieval(pool, present_indices, t_c)
            else:  # aligned
                absent_indices = aligned_retrieval(pool, present_indices, t_c, K)

            direction = estimate_direction(str(attr_idx), pool.embeddings[present_indices], pool.embeddings[absent_indices])
            cos_sim = float(torch.clamp(t_c @ direction.unit_vector, -1.0, 1.0))
            angles[attr_idx] = float(np.degrees(np.arccos(cos_sim)))

            raw = score_from_indices(spec, native_model, pool, present_indices, absent_indices)
            for native_idx, s in enumerate(raw):
                scores[(attr_idx, native_idx)] = s

        evaluate(strategy, faithfulness_records, scores, angles, results)

    print(f"\n=== stratified (k={STRATIFIED_K}, within-species retrieval) ===", flush=True)
    scores = {}
    angles = {}
    for attr_idx, (prefix, _part_names) in groundable.items():
        value = attribute_names[attr_idx].split("::", 1)[1]
        text = PREFIX_TEMPLATES[prefix].format(value=readable(value))
        t_c = build_concept_query(text, encoder)

        present_indices, absent_indices = stratified_retrieval(pool_with_labels, t_c, STRATIFIED_K)
        direction = estimate_direction(str(attr_idx), pool.embeddings[present_indices], pool.embeddings[absent_indices])
        cos_sim = float(torch.clamp(t_c @ direction.unit_vector, -1.0, 1.0))
        angles[attr_idx] = float(np.degrees(np.arccos(cos_sim)))

        raw = score_from_indices(spec, native_model, pool, present_indices, absent_indices)
        for native_idx, s in enumerate(raw):
            scores[(attr_idx, native_idx)] = s

    evaluate("stratified", faithfulness_records, scores, angles, results)

    with open(RESULTS_DIR / "cards_cub_retrieval_strategy_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for label, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([label, "", "", "", "", "", ""])
            else:
                writer.writerow([label, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])

    print(f"\nSaved {len(results)} strategies to results/cards_cub_retrieval_strategy_ablation.csv")


if __name__ == "__main__":
    main()
