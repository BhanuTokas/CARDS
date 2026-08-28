"""Third encoder ablation point, 87-attribute bank: open_clip_h
(ViT-H-14, laion2b_s32b_b79k -- configs/encoder/open_clip_h.yaml),
mirroring the CLIP ViT-B-32 comparison already run in
ablate_cards_cub_attributes.py (k=30, baseline phrasing, demean on/off).
Standalone script (not folded into the main ablation grid) so it can run
as an independent background process alongside the perception_encoder
ablation without one crash taking down the other, and so a much bigger
model's own pool-embedding cost doesn't block the smaller encoders'
already-running comparisons.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_cards_cub_attributes import PREFIX_TEMPLATES  # noqa: E402

from cards.data.cub_attributes import groundable_attributes, load_attribute_names  # noqa: E402
from cards.data.cub_parts import load_images_txt  # noqa: E402
from cards.models.backbones import BACKBONES  # noqa: E402
from cards.pipeline import instantiate_encoder  # noqa: E402
from cards.retrieval.confound import matched_retrieval  # noqa: E402
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool  # noqa: E402
from cards.retrieval.retrieve import retrieve_top_bottom_k  # noqa: E402
from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query  # noqa: E402
from cards.validation.broden_faithfulness import (  # noqa: E402
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
ATTRIBUTE_NAMES_PATH = CUB_ROOT / "attributes" / "new_attributes.txt"
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 30

ENCODER_CFG = {"name": "open_clip_h", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
               "model_name": "ViT-H-14", "pretrained": "laion2b_s32b_b79k", "device": DEVICE}


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


def run_config(groundable, attribute_names, encoder, spec, pool, native_model, use_demean, text_center):
    scores: dict[tuple[int, int], float] = {}
    for attr_idx, (prefix, _part_names) in groundable.items():
        value = attribute_names[attr_idx].split("::", 1)[1]
        text = PREFIX_TEMPLATES[prefix].format(value=readable(value))
        t_c = build_concept_query(text, encoder)
        if use_demean:
            t_c = demean_query(t_c, text_center)

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
        absent_indices = matched_retrieval(pool, present_indices, t_c)

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

    print("\nLoading open_clip_h (ViT-H-14/laion2b_s32b_b79k) + pool (this will take a while, large model)...", flush=True)
    encoder_cfg = OmegaConf.create({"device": DEVICE, **ENCODER_CFG})
    encoder = instantiate_encoder(OmegaConf.create({"encoder": encoder_cfg, "device": DEVICE}))
    pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": encoder_cfg, "cache_dir": "embedding_cache"})
    pool_cfg.dataset = {"name": "cub", "root": str(CUB_ROOT)}
    pool_cfg.pool_source = "test"
    pairs = [(image_paths[i], class_labels[i]) for i in test_ids]
    pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images", flush=True)

    spec = BACKBONES["resnet18_cub"]
    native_model = spec.load_native().to(DEVICE).eval()

    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    results = []
    for use_demean in [False, True]:
        label = f"open_clip_h_k30_demean{use_demean}_baseline"
        scores = run_config(groundable, attribute_names, encoder, spec, pool, native_model, use_demean, text_center)
        rho_result = score_method_agreement(faithfulness_records, scores, min_samples_per_pair=3)
        sign_result = score_sign_agreement(faithfulness_records, scores, min_samples_per_pair=3)
        results.append((label, rho_result, sign_result))
        if rho_result is None:
            print(f"[{label}] too few pairs", flush=True)
        else:
            print(f"[{label}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} p={rho_result.spearman_p:.4g} "
                  f"| sign={sign_result.agreement_frac:.1%} ({sign_result.n_agree}/{sign_result.n_pairs}) "
                  f"binom_p={sign_result.binom_p:.4g}", flush=True)

    with open(RESULTS_DIR / "cards_cub_open_clip_h_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for label, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([label, "", "", "", "", "", ""])
            else:
                writer.writerow([label, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])

    print(f"\nSaved {len(results)} configs to results/cards_cub_open_clip_h_ablation.csv")


if __name__ == "__main__":
    main()
