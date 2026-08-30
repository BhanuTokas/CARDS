"""Ablates CARDS' own knobs against the 8-part CUB faithfulness ground
truth (results/cub_faithfulness.csv, v41's ground-truth-label version),
prompted directly by "Can we run some ablations on CARDS with scaling,
encoders, etc." -- CARDS has been a flat null (~50% sign agreement,
~0 Spearman rho) in every comparison run so far (v41/v42/v43), and none
of its own configuration knobs have actually been varied yet to check
whether that's inherent to the method or an artifact of one fixed
configuration.

Axes ablated (each scored against the SAME unchanged 8-part ground
truth, via score_method_agreement + score_sign_agreement):
- K (retrieval depth): 15, 30 (the value used everywhere so far), 50.
- demean_query: off (used everywhere so far) vs. on, against
  CARDS' generic reference vocabulary -- flagged in memory
  (feedback_demean_query_per_dataset_encoder) as previously found to
  HURT on CUB with a different encoder/validation signal (the old
  PCBM-weight approach, since abandoned) -- worth re-testing fresh
  against the faithfulness-based ground truth, not assumed to transfer.
- prompt phrasing ensembling: the single fixed phrase used so far
  (e.g. "a bird's beak") vs. averaging several synonym phrasings per
  concept (each itself already template-ensembled by build_concept_query
  -- a second, outer ensembling layer).
- encoder: SigLIP (used everywhere so far) vs. CLIP ViT-B-32/openai
  (configs/encoder/clip.yaml) -- open_clip_h.yaml (ViT-H-14, much larger/
  slower) and perception_encoder.yaml (needs the sibling
  perception_models repo) are also configured and available but not
  included in this first pass, to keep the grid tractable.

Scope: K x demean x phrasing crossed fully under SigLIP (12 configs,
reusing the already-cached SigLIP pool); CLIP tested only at the
K=30/phrasing=baseline settings, with and without demean (2 more
configs) -- not a full cross with CLIP, to bound runtime. 14 configs
total. Only extends to the 87-attribute bank if something here actually
moves the needle.
"""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import torch.nn.functional as F

from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.cub_parts import load_images_txt
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder
from cards.retrieval.confound import matched_retrieval
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PART_NAME_TO_ID = {
    "beak": 2, "left_eye": 7, "left_leg": 8, "left_wing": 9,
    "right_eye": 11, "right_leg": 12, "right_wing": 13, "tail": 14,
}

BASELINE_QUERY_TEXT = {
    "beak": "a bird's beak", "left_eye": "a bird's eye", "right_eye": "a bird's eye",
    "left_wing": "a bird's wing", "right_wing": "a bird's wing",
    "left_leg": "a bird's leg", "right_leg": "a bird's leg", "tail": "a bird's tail",
}

# Synonym phrasings per part -- each phrasing itself already gets
# template-ensembled by build_concept_query; averaging across these is a
# second, outer ensembling layer over word choice, not just carrier
# sentence style.
PHRASING_VARIANTS = {
    "beak": ["a bird's beak", "a bird's bill", "the beak of a bird"],
    "left_eye": ["a bird's eye", "the eye of a bird", "a bird's eyeball"],
    "right_eye": ["a bird's eye", "the eye of a bird", "a bird's eyeball"],
    "left_wing": ["a bird's wing", "the wing of a bird", "a bird's feathered wing"],
    "right_wing": ["a bird's wing", "the wing of a bird", "a bird's feathered wing"],
    "left_leg": ["a bird's leg", "the leg of a bird", "a bird's foot"],
    "right_leg": ["a bird's leg", "the leg of a bird", "a bird's foot"],
    "tail": ["a bird's tail", "the tail feathers of a bird", "a bird's tail plumage"],
}

ENCODER_CONFIGS = {
    "siglip": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
               "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
    "clip": {"name": "clip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
             "model_name": "ViT-B-32", "pretrained": "openai", "device": DEVICE},
}


def build_query(concept: str, encoder, phrasing: str) -> torch.Tensor:
    if phrasing == "baseline":
        return build_concept_query(BASELINE_QUERY_TEXT[concept], encoder)
    variants = PHRASING_VARIANTS[concept]
    embeddings = torch.stack([build_concept_query(v, encoder) for v in variants])
    return F.normalize(embeddings.mean(dim=0), dim=0)


def load_faithfulness_records() -> list[FaithfulnessResult]:
    records = []
    with open(RESULTS_DIR / "cub_faithfulness.csv", newline="") as f:
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


def run_config(encoder_name: str, encoder, spec, pool, native_model, k: int, use_demean: bool, phrasing: str,
                text_center: torch.Tensor | None) -> dict[tuple[int, int], float]:
    scores: dict[tuple[int, int], float] = {}
    for part_name, part_id in PART_NAME_TO_ID.items():
        t_c = build_query(part_name, encoder, phrasing)
        if use_demean:
            t_c = demean_query(t_c, text_center)

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, k)
        absent_indices = matched_retrieval(pool, present_indices, t_c)

        present_paths = [pool.paths[i] for i in present_indices]
        absent_paths = [pool.paths[i] for i in absent_indices]
        present_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in present_paths]).to(DEVICE)
        absent_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in absent_paths]).to(DEVICE)

        with torch.no_grad():
            present_logits = native_model(present_batch)
            absent_logits = native_model(absent_batch)

        raw_score_all_classes = (present_logits.mean(dim=0) - absent_logits.mean(dim=0)).tolist()
        for native_idx, score in enumerate(raw_score_all_classes):
            scores[(part_id, native_idx)] = score

    return scores


def build_pool_for_encoder(encoder_cfg: dict, encoder, image_paths, class_labels, test_ids):
    cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": encoder_cfg, "cache_dir": "embedding_cache"})
    cfg.dataset = {"name": "cub", "root": str(CUB_ROOT)}
    cfg.pool_source = "test"
    pairs = [(image_paths[i], class_labels[i]) for i in test_ids]
    return load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    faithfulness_records = load_faithfulness_records()
    print(f"{len(faithfulness_records)} faithfulness records loaded (unchanged ground truth).", flush=True)

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

    spec = BACKBONES["resnet18_cub"]
    native_model = spec.load_native().to(DEVICE).eval()

    results = []

    print("\nLoading SigLIP + pool...", flush=True)
    siglip_cfg = OmegaConf.create({"device": DEVICE, **ENCODER_CONFIGS["siglip"]})
    siglip_encoder = instantiate_encoder(OmegaConf.create({"encoder": siglip_cfg, "device": DEVICE}))
    siglip_pool = build_pool_for_encoder(ENCODER_CONFIGS["siglip"], siglip_encoder, image_paths, class_labels, test_ids)
    siglip_text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, siglip_encoder)

    configs = list(itertools.product([15, 30, 50], [False, True], ["baseline", "ensemble"]))
    for k, use_demean, phrasing in configs:
        label = f"siglip_k{k}_demean{use_demean}_{phrasing}"
        scores = run_config("siglip", siglip_encoder, spec, siglip_pool, native_model, k, use_demean, phrasing,
                             siglip_text_center)
        rho_result = score_method_agreement(faithfulness_records, scores, min_samples_per_pair=3)
        sign_result = score_sign_agreement(faithfulness_records, scores, min_samples_per_pair=3)
        results.append((label, rho_result, sign_result))
        if rho_result is None:
            print(f"[{label}] too few pairs", flush=True)
        else:
            print(f"[{label}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} p={rho_result.spearman_p:.4g} "
                  f"| sign={sign_result.agreement_frac:.1%} ({sign_result.n_agree}/{sign_result.n_pairs}) binom_p={sign_result.binom_p:.4g}",
                  flush=True)

    print("\nLoading CLIP (ViT-B-32/openai) + pool...", flush=True)
    clip_cfg = OmegaConf.create({"device": DEVICE, **ENCODER_CONFIGS["clip"]})
    clip_encoder = instantiate_encoder(OmegaConf.create({"encoder": clip_cfg, "device": DEVICE}))
    clip_pool = build_pool_for_encoder(ENCODER_CONFIGS["clip"], clip_encoder, image_paths, class_labels, test_ids)
    clip_text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, clip_encoder)

    for use_demean in [False, True]:
        label = f"clip_k30_demean{use_demean}_baseline"
        scores = run_config("clip", clip_encoder, spec, clip_pool, native_model, 30, use_demean, "baseline",
                             clip_text_center)
        rho_result = score_method_agreement(faithfulness_records, scores, min_samples_per_pair=3)
        sign_result = score_sign_agreement(faithfulness_records, scores, min_samples_per_pair=3)
        results.append((label, rho_result, sign_result))
        if rho_result is None:
            print(f"[{label}] too few pairs", flush=True)
        else:
            print(f"[{label}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} p={rho_result.spearman_p:.4g} "
                  f"| sign={sign_result.agreement_frac:.1%} ({sign_result.n_agree}/{sign_result.n_pairs}) binom_p={sign_result.binom_p:.4g}",
                  flush=True)

    with open(RESULTS_DIR / "cards_cub_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for label, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([label, "", "", "", "", "", ""])
            else:
                writer.writerow([label, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])

    print(f"\nSaved {len(results)} configs to results/cards_cub_ablation.csv")
    print("\n=== summary, sorted by |spearman_rho| descending ===")
    scored = [(label, r, s) for label, r, s in results if r is not None]
    scored.sort(key=lambda t: -abs(t[1].spearman_rho))
    for label, r, s in scored:
        print(f"{label:<45s} rho={r.spearman_rho:+.4f} (p={r.spearman_p:.4g})  "
              f"sign={s.agreement_frac:.1%} (p={s.binom_p:.4g})")


if __name__ == "__main__":
    main()
