"""Encoder ablation for CARDS on CelebA -- prompted directly ("Can we do
an ablation on the encoder being used?" / "For both CUB and CelebA?"),
mirroring `scripts/cub/ablate/ablate_cards_cub_encoder_full_scale.py`.
CelebA has never had an encoder axis tested at all (SigLIP was simply
carried over from CUB's own settled default from the start, same as
K=50/aligned/demean=True -- v76 already ablated the retrieval-strategy
axis and found no config rescues CARDS; this ablates the remaining
untested axis).

3 new encoders (CLIP, open_clip_h, Perception) run at IDENTICAL settings
to CARDS' CelebA production config (K=50, demean_query=True,
aligned_retrieval) against the full 26-concept, 2-task blur-based ground
truth. SigLIP's own score is NOT recomputed -- `results/cards_celeba_
full_scores.csv` (from `run_cards_celeba_full.py`, the official
production script) already holds exactly this config's scores, reused
directly.
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

from run_cards_celeba_full import CONCEPT_QUERY_TEXT, TASK_POSITIVE_LOGIT_INDEX

from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS, TARGET_CLASSES
from cards.data.datasets import load_celeba
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

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}

ENCODERS_TO_RUN = {
    "clip": {"name": "clip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
             "model_name": "ViT-B-32", "pretrained": "openai", "device": DEVICE},
    "open_clip_h": {"name": "open_clip_h", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                     "model_name": "ViT-H-14", "pretrained": "laion2b_s32b_b79k", "device": DEVICE},
    "perception": {"name": "perception_encoder", "_target_": "cards.encoders.perception_encoder.PerceptionEncoder",
                    "model_name": "PE-Core-B16-224", "perception_models_path": "../perception_models", "device": DEVICE},
}


def load_records_by_task() -> dict[str, list[FaithfulnessResult]]:
    by_task: dict[str, list[FaithfulnessResult]] = {t: [] for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "celeba_full_faithfulness.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]].append(FaithfulnessResult(
                image=row["image"], concept_number=CONCEPT_TO_IDX[row["concept_name"]], category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return by_task


def load_siglip_scores_by_task() -> dict[str, dict[tuple[int, int], float]]:
    by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "cards_celeba_full_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]][(CONCEPT_TO_IDX[row["concept_name"]], 1)] = float(row["raw_score"])
    return by_task


def score_from_indices(spec, native_model, pool, present_indices, absent_indices, task_idx: int) -> float:
    present_paths = [pool.paths[i] for i in present_indices]
    absent_paths = [pool.paths[i] for i in absent_indices]
    present_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in present_paths]).to(DEVICE)
    absent_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in absent_paths]).to(DEVICE)
    with torch.no_grad():
        present_logits = native_model(present_batch)
        absent_logits = native_model(absent_batch)
    return (present_logits[:, task_idx].mean() - absent_logits[:, task_idx].mean()).item()


def evaluate(label, task_name, records, scores, results):
    rho_result = score_method_agreement(records, scores)
    sign_result = score_sign_agreement(records, scores)
    results.append((label, task_name, rho_result, sign_result))
    if rho_result is not None:
        print(f"  [{task_name}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} "
              f"p={rho_result.spearman_p:.4g} | sign={sign_result.agreement_frac:.1%} "
              f"({sign_result.n_agree}/{sign_result.n_pairs}) binom_p={sign_result.binom_p:.4g}", flush=True)
    else:
        print(f"  [{task_name}] too few pairs", flush=True)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    records_by_task = load_records_by_task()
    for task in TARGET_CLASSES:
        print(f"{task}: {len(records_by_task[task])} faithfulness records loaded.", flush=True)

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()

    results = []
    print("\n=== siglip (reused from results/cards_celeba_full_scores.csv) ===", flush=True)
    siglip_scores = load_siglip_scores_by_task()
    for task_name in TARGET_CLASSES:
        evaluate("siglip", task_name, records_by_task[task_name], siglip_scores[task_name], results)

    pairs = load_celeba(CELEBA_HQ_ROOT, split="val")
    for name, encoder_cfg in ENCODERS_TO_RUN.items():
        print(f"\n=== {name} (K={K}, demean=True, aligned) ===", flush=True)
        encoder = instantiate_encoder(OmegaConf.create({"encoder": encoder_cfg, "device": DEVICE}))
        pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": encoder_cfg, "cache_dir": "embedding_cache"})
        pool_cfg.dataset = {"name": "celeba", "root": str(CELEBA_HQ_ROOT)}
        pool_cfg.pool_source = "val"
        pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
        print(f"pool: {len(pool.paths)} images", flush=True)
        text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

        scores_by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
        for concept_name in GROUNDABLE_CONCEPTS:
            query_text = CONCEPT_QUERY_TEXT[concept_name]
            t_c = build_concept_query(query_text, encoder)
            t_c = demean_query(t_c, text_center)

            present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
            absent_indices = aligned_retrieval(pool, present_indices, t_c, K)

            for task_name in TARGET_CLASSES:
                task_idx = TASK_POSITIVE_LOGIT_INDEX[task_name]
                raw_score = score_from_indices(spec, native_model, pool, present_indices, absent_indices, task_idx)
                scores_by_task[task_name][(CONCEPT_TO_IDX[concept_name], 1)] = raw_score

        for task_name in TARGET_CLASSES:
            evaluate(name, task_name, records_by_task[task_name], scores_by_task[task_name], results)

    with open(RESULTS_DIR / "cards_celeba_encoder_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["encoder", "target_task", "n_pairs", "spearman_rho", "spearman_p",
                          "sign_agreement", "n_agree", "binom_p"])
        for label, task_name, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([label, task_name, "", "", "", "", "", ""])
            else:
                writer.writerow([label, task_name, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])
    print(f"\nSaved {len(results)} (encoder, task) rows to results/cards_celeba_encoder_ablation.csv")


if __name__ == "__main__":
    main()
