"""Retrieval-strategy/demean ablation for CARDS on CelebA -- prompted
directly ("Can you run an ablation on CUB parameters to see if anything
works here?"), closing the "still open" item flagged repeatedly since
v72: CARDS' K=50/aligned_retrieval/demean_query=True settings on CelebA
were simply carried over from CUB's own settled defaults
(notes/cub_correlation_investigation.md v46/v47), never re-tuned or
ablated on CelebA's own concept bank. CARDS' chance-level result (v72,
v73) could mean "CARDS doesn't work here" or just "these particular
settings are wrong for this concept bank" -- this script is the direct
test, mirroring scripts/cub/ablate/ablate_cards_cub_retrieval_strategy.py's
own design (naive/matched/aligned retrieval strategies) plus a
demean_query on/off axis (CUB's v47 grid found demean=True was part of
its own single best config, also never re-tested here).

**`stratified_retrieval` is deliberately excluded, not forgotten**: CUB's
own stratified ablation retrieves P_c/N_c independently within each of
CUB's 200 species, testing whether species-identity variance dominates
the embedding space enough to swamp the intended concept direction.
CelebA has only 2 target classes (Attractive, Young) -- stratifying
within 2 strata isn't a structurally comparable test of the same
hypothesis (CUB's own version needed 200 strata to be a meaningful
check), so it's skipped rather than forced into a weak analog.

6 configs (3 strategies x 2 demean settings) x 26 concepts x 2 tasks,
scored against the SAME blur-based masking ground truth (celeba_full_
faithfulness.csv) v72's own comparison used, so results here are
directly comparable to the v72/v73/v74/v75 table without re-deriving a
ground truth.
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
from cards.retrieval.confound import matched_retrieval
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
K = 50  # matches the production CelebA default (run_cards_celeba_full.py), unchanged across this ablation
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}


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


def score_from_indices(spec, native_model, pool, present_indices, absent_indices, task_idx: int) -> float:
    present_paths = [pool.paths[i] for i in present_indices]
    absent_paths = [pool.paths[i] for i in absent_indices]
    present_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in present_paths]).to(DEVICE)
    absent_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in absent_paths]).to(DEVICE)
    with torch.no_grad():
        present_logits = native_model(present_batch)
        absent_logits = native_model(absent_batch)
    return (present_logits[:, task_idx].mean() - absent_logits[:, task_idx].mean()).item()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    records_by_task = load_records_by_task()
    for task in TARGET_CLASSES:
        print(f"{task}: {len(records_by_task[task])} faithfulness records loaded.", flush=True)

    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
    })
    print("Loading SigLIP encoder + CelebA retrieval pool...", flush=True)
    encoder = instantiate_encoder(cfg)
    cfg.dataset = {"name": "celeba", "root": str(CELEBA_HQ_ROOT)}
    cfg.pool_source = "val"
    pairs = load_celeba(CELEBA_HQ_ROOT, split="val")
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images", flush=True)

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    results = []
    raw_score_rows = []  # (config, concept_name, target_task, raw_score) -- saved so downstream
    # scripts (e.g. scoring each config against a non-delta_p reference like
    # the Integrated Gradients baseline) can reuse these without re-running
    # retrieval, which is the expensive part of this script.
    for strategy in ["naive", "matched", "aligned"]:
        for demean in [False, True]:
            label = f"{strategy}{'+demean' if demean else ''}"
            print(f"\n=== {label} (K={K}) ===", flush=True)
            scores_by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}

            for concept_name in GROUNDABLE_CONCEPTS:
                query_text = CONCEPT_QUERY_TEXT[concept_name]
                t_c = build_concept_query(query_text, encoder)
                if demean:
                    t_c = demean_query(t_c, text_center)

                present_indices, naive_absent = retrieve_top_bottom_k(pool, t_c, K)
                if strategy == "naive":
                    absent_indices = naive_absent
                elif strategy == "matched":
                    absent_indices = matched_retrieval(pool, present_indices, t_c)
                else:  # aligned
                    absent_indices = aligned_retrieval(pool, present_indices, t_c, K)

                for task_name in TARGET_CLASSES:
                    task_idx = TASK_POSITIVE_LOGIT_INDEX[task_name]
                    raw_score = score_from_indices(spec, native_model, pool, present_indices, absent_indices, task_idx)
                    scores_by_task[task_name][(CONCEPT_TO_IDX[concept_name], 1)] = raw_score
                    raw_score_rows.append((label, concept_name, task_name, raw_score))

            for task_name in TARGET_CLASSES:
                rho_result = score_method_agreement(records_by_task[task_name], scores_by_task[task_name])
                sign_result = score_sign_agreement(records_by_task[task_name], scores_by_task[task_name])
                results.append((label, task_name, rho_result, sign_result))
                if rho_result is not None:
                    print(f"  [{task_name}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} "
                          f"p={rho_result.spearman_p:.4g} | sign={sign_result.agreement_frac:.1%} "
                          f"({sign_result.n_agree}/{sign_result.n_pairs}) binom_p={sign_result.binom_p:.4g}", flush=True)
                else:
                    print(f"  [{task_name}] too few pairs", flush=True)

    with open(RESULTS_DIR / "cards_celeba_retrieval_strategy_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "target_task", "n_pairs", "spearman_rho", "spearman_p",
                          "sign_agreement", "n_agree", "binom_p"])
        for label, task_name, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([label, task_name, "", "", "", "", "", ""])
            else:
                writer.writerow([label, task_name, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])
    print(f"\nSaved {len(results)} (config, task) rows to results/cards_celeba_retrieval_strategy_ablation.csv")

    with open(RESULTS_DIR / "cards_celeba_retrieval_strategy_ablation_raw_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "concept_name", "target_task", "raw_score"])
        writer.writerows(raw_score_rows)
    print(f"Saved {len(raw_score_rows)} raw per-concept scores to "
          f"results/cards_celeba_retrieval_strategy_ablation_raw_scores.csv")


if __name__ == "__main__":
    main()
