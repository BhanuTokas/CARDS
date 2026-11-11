"""Official-train counterpart of `ablate_cards_celeba_masking_hybrid_
joint_orth_demean_encoder.py` -- same joint encoder x orthogonalize x
demean_query ablation (z-score threshold alpha=1.0, K=50), but for the
`celeba_official_train_attractive_male` (ResNet18) classifier instead
of `celeba_attractive_young`, prompted directly ("Can you build an
encoder ablation with ResNet on the official train pipeline?" -> "Joint
encoder x orthogonalize x demean_query" scope confirmation).

**No reuse shortcut here, unlike the CelebA-HQ-track original** -- that
script could skip 8/16 cells because a prior siglip/CLIP/open_clip_h/
Perception x orth sweep already existed for `celeba_attractive_young`'s
own official-val pool. No such prior sweep exists for the official-
train classifier at any encoder besides SigLIP (only `run_cards_celeba_
masking_hybrid_official_train.py`'s own single siglip/orth=True/
demean=True run has ever been computed) -- every one of the 4 (encoder)
x 2 (orth) x 2 (demean) = 16 cells is computed fresh here.

Mirrors `run_cards_celeba_masking_hybrid_official_train.py`'s own
official-train specifics exactly: SAME official-val-minus-HQ-overlap
retrieval pool (`build_clean_official_val_paths`), SAME TaskBlackBox /
TASK_SLICE ([Attractive]=1, [Male]=3 within the 4-way head), SAME two
ground-truth versions (non_overlapping, previously_used) both scored
per cell. `celeba_attractive_young`'s CLIP-RN50 exclusion ("Drop
CLIP-RN50") carried over unchanged -- same 4-encoder set.

Per (encoder, demean) pair, the pool is built/encoded ONCE (retrieval
embeddings don't depend on orthogonalize), then BOTH orthogonalize
values reuse it -- 4 encoders x 2 demean = 8 pool builds, each driving
2 full process_concept passes (orth=True/False) = 16 total expensive
passes over the 26-concept bank, each scored against both tasks x both
GT versions = 64 correlation cells.

HPC-portable (CELEBA_HQ_ROOT/CELEBA_ROOT/CARDS_RESULTS_DIR/
CARDS_CACHE_DIR env-var overrides), matching every other official-train
script's own convention -- this is real GPU compute (16x a single
masking-hybrid pass), expected to run on Sol via the generic pipeline
SLURM template (`CARDS_SCRIPT=scripts/celeba/ablate/ablate_cards_
celeba_masking_hybrid_joint_orth_demean_encoder_official_train.py`, no
`CARDS_BACKBONE_NAME` override needed -- this script is ResNet18-only
by request, not backbone-parameterized like the main pipeline scripts).
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT
from run_cards_celeba_masking_hybrid_official_val_zscore import build_clean_official_val_paths

from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.models.backbones import BACKBONES
from cards.pipeline import (
    instantiate_encoder,
    orthogonalize_queries,
    process_concept,
    score_masking_hybrid_concepts,
)
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

CELEBA_ROOT = Path(os.environ.get("CELEBA_ROOT", r"C:\Users\btokas\Projects\Datasets\CelebA\celeba"))
RESULTS_DIR = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
CACHE_DIR = Path(os.environ.get("CARDS_CACHE_DIR", "embedding_cache"))
BACKBONE_NAME = "celeba_official_train_attractive_male"  # ResNet18 only, per direct request
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
ALPHA = 1.0
SEED = 0
TASKS = ["Attractive", "Male"]
TASK_SLICE = {"Attractive": 1, "Male": 3}
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
GT_VERSIONS = {
    "non_overlapping": "celeba_official_train_faithfulness_non_overlapping.csv",
    "previously_used": "celeba_official_train_faithfulness_previously_used.csv",
}

ENCODER_CFGS = {
    "siglip": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
               "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
    "clip": {"name": "clip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
             "model_name": "ViT-B-32", "pretrained": "openai", "device": DEVICE},
    "open_clip_h": {"name": "open_clip_h", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                     "model_name": "ViT-H-14", "pretrained": "laion2b_s32b_b79k", "device": DEVICE},
    "perception_encoder": {"name": "perception_encoder", "_target_": "cards.encoders.perception_encoder.PerceptionEncoder",
                            "model_name": "PE-Core-B16-224", "perception_models_path": "../perception_models", "device": DEVICE},
}


class TaskBlackBox:
    def __init__(self, native_model, task_name: str, preprocess, device: str):
        self.model = native_model
        self.task_idx = TASK_SLICE[task_name]
        self._preprocess = preprocess
        self.device = device

    def preprocess(self, image):
        return self._preprocess(image.convert("RGB"))

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch.to(self.device))[:, self.task_idx].detach().cpu()


def load_records(gt_filename: str, task_name: str) -> list[FaithfulnessResult]:
    records = []
    with open(RESULTS_DIR / gt_filename, newline="") as f:
        for row in csv.DictReader(f):
            if row["target_task"] != task_name:
                continue
            records.append(FaithfulnessResult(
                image=row["image"], concept_number=CONCEPT_TO_IDX[row["concept_name"]], category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return records


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    spec = BACKBONES[BACKBONE_NAME]
    native_model = spec.load_native().to(DEVICE).eval()

    official_paths = build_clean_official_val_paths()
    pairs = [(p, 0) for p in official_paths]

    all_rows = []  # (encoder, orth, demean, task, gt_version, n_pairs, rho, rho_p, sign_frac, n_agree, sign_p)
    raw_rows = []  # (encoder, orth, demean, task, concept_name, hybrid_raw_score)

    for encoder_name, encoder_cfg in ENCODER_CFGS.items():
        cfg_base = OmegaConf.create({
            "seed": SEED, "device": DEVICE, "encoder": encoder_cfg, "cache_dir": str(CACHE_DIR),
            "retrieval": {"strategy": "naive"}, "k": K,
            "masking_hybrid": {"threshold_method": "zscore", "alpha": ALPHA},
        })
        encoder = instantiate_encoder(cfg_base)
        text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)
        raw_queries = {c: build_concept_query(CONCEPT_QUERY_TEXT[c], encoder) for c in GROUNDABLE_CONCEPTS}

        pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": encoder_cfg, "cache_dir": str(CACHE_DIR)})
        pool_cfg.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}
        pool_cfg.pool_source = "val"
        pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
        print(f"\n{'=' * 20} encoder={encoder_name}  pool: {len(pool.paths)} images {'=' * 20}", flush=True)

        for demean in (True, False):
            queries_base = ({c: demean_query(q, text_center) for c, q in raw_queries.items()}
                             if demean else raw_queries)

            for orth in (True, False):
                print(f"  --- demean_query={demean} orthogonalize={orth} ---", flush=True)
                queries = orthogonalize_queries(queries_base) if orth else queries_base

                results = []
                for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
                    result = process_concept(cfg_base, encoder, pool, concept_name, query=queries[concept_name])
                    results.append(result)
                    if (concept_idx + 1) % 10 == 0:
                        print(f"    retrieval [{concept_idx + 1}/{len(GROUNDABLE_CONCEPTS)}]", flush=True)

                for task_name in TASKS:
                    black_box = TaskBlackBox(native_model, task_name, spec.preprocess, DEVICE)
                    hybrid_results = score_masking_hybrid_concepts(
                        cfg_base, encoder, black_box, pool, GROUNDABLE_CONCEPTS, results)
                    scores = {(CONCEPT_TO_IDX[c], 1): r.raw_score for c, r in hybrid_results.items()}
                    for c, r in hybrid_results.items():
                        raw_rows.append((encoder_name, orth, demean, task_name, c, r.raw_score))

                    for gt_name, gt_filename in GT_VERSIONS.items():
                        records_task = load_records(gt_filename, task_name)
                        rho_r = score_method_agreement(records_task, scores)
                        sign_r = score_sign_agreement(records_task, scores)
                        if rho_r is None:
                            print(f"    [{task_name}/{gt_name}] too few pairs", flush=True)
                            continue
                        print(f"    [{task_name}/{gt_name}] n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} "
                              f"(p={rho_r.spearman_p:.4g})  sign={sign_r.agreement_frac:.1%} (p={sign_r.binom_p:.4g})",
                              flush=True)
                        all_rows.append((encoder_name, orth, demean, task_name, gt_name, rho_r.n_pairs,
                                          rho_r.spearman_rho, rho_r.spearman_p, sign_r.agreement_frac,
                                          sign_r.n_agree, sign_r.binom_p))

    out_path = RESULTS_DIR / "cards_celeba_masking_hybrid_joint_orth_demean_encoder_official_train_ablation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["encoder", "orthogonalize", "demean_query", "target_task", "gt_version", "n_pairs",
                          "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {out_path}", flush=True)

    raw_path = RESULTS_DIR / "cards_celeba_masking_hybrid_joint_orth_demean_encoder_official_train_ablation_raw_scores.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["encoder", "orthogonalize", "demean_query", "target_task", "concept_name", "hybrid_raw_score"])
        writer.writerows(raw_rows)
    print(f"Saved {len(raw_rows)} raw rows to {raw_path}", flush=True)

    print("\n=== full 32-cell grid (non_overlapping GT), sorted by |rho| ===")
    non_overlap_rows = [r for r in all_rows if r[4] == "non_overlapping"]
    for r in sorted(non_overlap_rows, key=lambda r: -abs(r[6])):
        print(f"  encoder={r[0]:<20s} orth={r[1]!s:<5s} demean={r[2]!s:<5s} task={r[3]:<10s} "
              f"rho={r[6]:+.4f} sign={r[8]:.1%}")


if __name__ == "__main__":
    main()
