"""Encoder ablation for CARDS on CUB, at the CURRENT full 3,526-pair
ground truth -- prompted directly ("Can we do an ablation on the encoder
being used?" / "For both CUB and CelebA?"). Earlier CUB encoder checks
(`ablate_cards_cub_open_clip_h_aligned.py`, `ablate_cards_cub_perception_
aligned.py`, both still on disk) were run against the OLD 46-pair ground
truth (results/cards_cub_open_clip_h_aligned_ablation.csv/cards_cub_
perception_aligned_ablation.csv both show n_pairs=46) -- this
investigation's own winner's-curse lesson (v60) means those numbers
should not be trusted at face value now that the ground truth has grown
20x+ (v53/v56/v62) to its true maximum. This script re-tests all 4
encoders this codebase supports at IDENTICAL settings (K=50, demean=True,
aligned_retrieval -- the current production default) against the LIVE,
current-scale ground truth, for a clean, directly-comparable 4-way table
that wasn't previously possible (the old per-encoder scripts used
slightly different K/demean subsets each).

SigLIP's own score is NOT recomputed -- `results/cards_cub_attribute_
scores.csv` (from `run_cards_cub_attributes.py`, the official production
script) already holds exactly this config's scores, reused directly.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

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
from cards.retrieval.aligned import aligned_retrieval
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
ATTRIBUTE_NAMES_PATH = CUB_ROOT / "attributes" / "new_attributes.txt"
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
USE_DEMEAN = True

ENCODERS_TO_RUN = {
    "clip": {"name": "clip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
             "model_name": "ViT-B-32", "pretrained": "openai", "device": DEVICE},
    "open_clip_h": {"name": "open_clip_h", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                     "model_name": "ViT-H-14", "pretrained": "laion2b_s32b_b79k", "device": DEVICE},
    "perception": {"name": "perception_encoder", "_target_": "cards.encoders.perception_encoder.PerceptionEncoder",
                    "model_name": "PE-Core-B16-224", "perception_models_path": "../perception_models", "device": DEVICE},
}


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


def load_siglip_scores() -> dict[tuple[int, int], float]:
    scores = {}
    with open(RESULTS_DIR / "cards_cub_attribute_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            scores[(int(row["attribute_index"]), int(row["native_class_idx"]))] = float(row["raw_score"])
    return scores


def run_encoder(groundable, attribute_names, encoder, spec, pool, native_model, text_center) -> dict[tuple[int, int], float]:
    scores: dict[tuple[int, int], float] = {}
    for attr_idx, (prefix, _part_names) in groundable.items():
        value = attribute_names[attr_idx].split("::", 1)[1]
        text = PREFIX_TEMPLATES[prefix].format(value=readable(value))
        t_c = build_concept_query(text, encoder)
        t_c = demean_query(t_c, text_center)

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
        absent_indices = aligned_retrieval(pool, present_indices, t_c, K)

        present_paths = [pool.paths[i] for i in present_indices]
        absent_paths = [pool.paths[i] for i in absent_indices]
        present_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in present_paths]).to(DEVICE)
        absent_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in absent_paths]).to(DEVICE)

        with torch.no_grad():
            present_logits = native_model(present_batch)
            absent_logits = native_model(absent_batch)

        raw_score_all_classes = (present_logits.mean(dim=0) - absent_logits.mean(dim=0)).tolist()
        for native_idx, s in enumerate(raw_score_all_classes):
            scores[(attr_idx, native_idx)] = s
    return scores


def evaluate(label, records, scores, results):
    rho_result = score_method_agreement(records, scores, min_samples_per_pair=3)
    sign_result = score_sign_agreement(records, scores, min_samples_per_pair=3)
    results.append((label, rho_result, sign_result))
    if rho_result is None:
        print(f"[{label}] too few pairs", flush=True)
    else:
        print(f"[{label}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} p={rho_result.spearman_p:.4g} "
              f"| sign={sign_result.agreement_frac:.1%} ({sign_result.n_agree}/{sign_result.n_pairs}) "
              f"binom_p={sign_result.binom_p:.4g}", flush=True)


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
    pairs = [(image_paths[i], class_labels[i]) for i in test_ids]

    spec = BACKBONES["resnet18_cub"]
    native_model = spec.load_native().to(DEVICE).eval()

    results = []
    print("\n=== siglip (reused from results/cards_cub_attribute_scores.csv) ===", flush=True)
    evaluate("siglip", faithfulness_records, load_siglip_scores(), results)

    for name, encoder_cfg in ENCODERS_TO_RUN.items():
        print(f"\n=== {name} (K={K}, demean={USE_DEMEAN}, aligned) ===", flush=True)
        encoder = instantiate_encoder(OmegaConf.create({"encoder": encoder_cfg, "device": DEVICE}))
        pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": encoder_cfg, "cache_dir": "embedding_cache"})
        pool_cfg.dataset = {"name": "cub", "root": str(CUB_ROOT)}
        pool_cfg.pool_source = "test"
        pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
        print(f"pool: {len(pool.paths)} images", flush=True)
        text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

        scores = run_encoder(groundable, attribute_names, encoder, spec, pool, native_model, text_center)
        evaluate(name, faithfulness_records, scores, results)

    with open(RESULTS_DIR / "cards_cub_encoder_ablation_full_scale.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["encoder", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for label, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([label, "", "", "", "", "", ""])
            else:
                writer.writerow([label, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])
    print(f"\nSaved {len(results)} encoders to results/cards_cub_encoder_ablation_full_scale.csv")


if __name__ == "__main__":
    main()
