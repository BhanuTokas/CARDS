"""Joint orthogonalize x demean_query x encoder ablation for ConceptMask
on CelebA (official-val pool, z-score threshold alpha=1.0, K=50) --
direct request ("Ablations need to check orthogonalization, modality
shift, encoder choice together").

Built on the REAL cards.pipeline integration (cfg.scoring_mode==
"masking_hybrid", masking_hybrid.threshold_method="zscore" --
process_concept + score_masking_hybrid_concepts, reused directly)
instead of yet another hand-rolled copy of the localize->mask->score
loop every prior CelebA ablation script in this directory duplicated.

Scope, decided directly with the user:
  - Encoders: siglip, clip (ViT-B-32), open_clip_h, perception_encoder.
    CLIP-RN50 explicitly excluded ("Drop CLIP-RN50").
  - demean_query=True x {orthogonalize=True,False} x all 4 encoders is
    ALREADY DONE (results/cards_celeba_masking_hybrid_encoder_zscore_
    official_val_ablation*.csv for orth=True; the _no_orth_ variant for
    orth=False) -- reused/tabulated here via REUSED_DEMEAN_TRUE below,
    NOT recomputed ("use SIGLIP's existing number if available" --
    applied to all 4 encoders, not just siglip, for the same reason).
  - demean_query=False was NEVER run for CelebA at all (confirmed
    directly -- only a CUB one exists, irrelevant here). The 8 NEW
    cells this script actually computes: demean_query=False x
    {orthogonalize=True,False} x 4 encoders.

Ground truth: `celeba_full_faithfulness_attribute_conditioned.csv` (the
CORRECTED, attribute-conditioned version), per direct correction
("Shouldn't we use the corrected GT?") -- NOT the original
`celeba_full_faithfulness.csv` this script first used, which silently
mismatched the 0.6704 SigLIP official-val number already known from
`score_methods_against_attribute_conditioned_gt.py`'s own corrected-GT
run (that script's number; this one's first draft reused a DIFFERENT,
original-GT-scored 0.4715 for the same nominal config, caught directly
by the user, not self-caught).

Critically, this does NOT require re-running the expensive masking
pipeline for the 8 already-computed demean=True cells: each method's
`raw_score` is a per-(concept,class) scalar, independent of which
ground truth it gets correlated against (score_method_agreement's own
documented design, established earlier in this same investigation) --
so REUSED_DEMEAN_TRUE below loads the EXISTING per-concept raw_score
tables (`..._ablation_raw_scores.csv` / `..._no_orth_..._raw_scores.csv`
/ `cards_celeba_masking_hybrid_official_val_k1_scores.csv` for siglip's
own orth=True cell) and re-correlates them against the corrected GT
fresh -- cheap, no GPU, no re-masking. Only the 8 NEW demean_query=False
cells need the actual expensive pipeline run, scored against the
corrected GT directly since they're computed from scratch anyway.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT, TASK_POSITIVE_LOGIT_INDEX
from run_cards_celeba_masking_hybrid_official_val_zscore import build_clean_official_val_paths

from cards.concepts.prompts import (
    build_concept_query,
)
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS, TARGET_CLASSES
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

CELEBA_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebA\celeba")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
ALPHA = 1.0
SEED = 0
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}

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

# demean_query=True x {orth=True, orth=False} x all 4 encoders -- the
# expensive masking/localization pass is already done for every one of
# these 8 cells (see this file's own docstring for exactly which raw-
# score CSV covers which cell); only the final correlation against the
# corrected ground truth is computed fresh, by load_demean_true_raw_scores
# below, since that step is cheap and GT-dependent.
REUSED_DEMEAN_TRUE_CELLS = [(encoder_name, orth) for encoder_name in ENCODER_CFGS for orth in (True, False)]


def load_demean_true_raw_scores(encoder_name: str, orth: bool) -> dict[str, dict[tuple[int, int], float]]:
    """Per-task {(concept_idx, 1): hybrid_raw_score} for one already-computed
    demean_query=True (encoder, orth) cell, read from whichever of the three
    existing raw-score CSVs actually covers it."""
    scores_by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    if encoder_name == "siglip" and orth:
        path = RESULTS_DIR / "cards_celeba_masking_hybrid_official_val_k1_scores.csv"
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                key = (CONCEPT_TO_IDX[row["concept_name"]], 1)
                scores_by_task[row["target_task"]][key] = float(row["hybrid_raw_score"])
    elif orth:
        path = RESULTS_DIR / "cards_celeba_masking_hybrid_encoder_zscore_official_val_ablation_raw_scores.csv"
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["encoder"] != encoder_name:
                    continue
                key = (CONCEPT_TO_IDX[row["concept_name"]], 1)
                scores_by_task[row["target_task"]][key] = float(row["hybrid_raw_score"])
    else:
        path = RESULTS_DIR / "cards_celeba_masking_hybrid_encoder_zscore_no_orth_official_val_ablation_raw_scores.csv"
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["encoder"] != encoder_name or row["alpha"] != "1.0":
                    continue
                key = (CONCEPT_TO_IDX[row["concept_name"]], 1)
                scores_by_task[row["target_task"]][key] = float(row["hybrid_raw_score"])
    return scores_by_task


class TaskBlackBox:
    """Wraps the native celeba_attractive_young checkpoint's single-task
    logit as `b(x) -> scalar`, the BlackBoxModel contract."""

    def __init__(self, native_model, task_name: str, preprocess, device: str):
        self.model = native_model
        self.task_idx = TASK_POSITIVE_LOGIT_INDEX[task_name]
        self._preprocess = preprocess
        self.device = device

    def preprocess(self, image):
        return self._preprocess(image.convert("RGB"))

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch.to(self.device))[:, self.task_idx].detach().cpu()


def load_records_by_task() -> dict[str, list[FaithfulnessResult]]:
    by_task: dict[str, list[FaithfulnessResult]] = {t: [] for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "celeba_full_faithfulness_attribute_conditioned.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]].append(FaithfulnessResult(
                image=row["image"], concept_number=CONCEPT_TO_IDX[row["concept_name"]], category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return by_task


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    records_by_task = load_records_by_task()

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()

    official_paths = build_clean_official_val_paths()
    pairs = [(p, 0) for p in official_paths]

    all_rows = []  # (encoder, orthogonalize, demean, target_task, n_pairs, rho, rho_p, sign_frac, n_agree, sign_p, source)

    # Reused demean=True cells -- raw scores loaded from disk (already
    # computed), re-correlated fresh against the corrected GT here.
    print(f"\n{'=' * 20} reusing demean_query=True raw scores, re-scoring against corrected GT {'=' * 20}", flush=True)
    for encoder_name, orth in REUSED_DEMEAN_TRUE_CELLS:
        scores_by_task = load_demean_true_raw_scores(encoder_name, orth)
        for task_name in TARGET_CLASSES:
            rho_r = score_method_agreement(records_by_task[task_name], scores_by_task[task_name])
            sign_r = score_sign_agreement(records_by_task[task_name], scores_by_task[task_name])
            if rho_r is None:
                print(f"  [{encoder_name} orth={orth} {task_name}] too few pairs", flush=True)
                continue
            print(f"  [{encoder_name} orth={orth} {task_name}] n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} "
                  f"(p={rho_r.spearman_p:.4g})  sign={sign_r.agreement_frac:.1%} (p={sign_r.binom_p:.4g})", flush=True)
            all_rows.append((encoder_name, orth, True, task_name, rho_r.n_pairs, rho_r.spearman_rho,
                              rho_r.spearman_p, sign_r.agreement_frac, sign_r.n_agree, sign_r.binom_p, "reused"))

    # NEW: demean_query=False, both orthogonalize values, all 4 encoders.
    cfg_base = OmegaConf.create({
        "seed": SEED, "device": DEVICE,
        "retrieval": {"strategy": "naive"}, "k": K,
        "masking_hybrid": {"threshold_method": "zscore", "alpha": ALPHA},
    })

    for encoder_name, encoder_cfg in ENCODER_CFGS.items():
        print(f"\n{'=' * 20} encoder={encoder_name}, demean_query=False {'=' * 20}", flush=True)
        cfg = OmegaConf.create({**cfg_base, "encoder": encoder_cfg})
        encoder = instantiate_encoder(cfg)

        pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": encoder_cfg, "cache_dir": "embedding_cache"})
        pool_cfg.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}
        pool_cfg.pool_source = "val"
        pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
        print(f"  pool: {len(pool.paths)} images", flush=True)

        # RAW (non-demeaned) queries -- demean_query=False for this whole block.
        raw_queries = {c: build_concept_query(CONCEPT_QUERY_TEXT[c], encoder) for c in GROUNDABLE_CONCEPTS}

        for orth in (True, False):
            print(f"  --- orthogonalize={orth} ---", flush=True)
            queries = orthogonalize_queries(raw_queries) if orth else raw_queries

            results = []
            for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
                result = process_concept(cfg, encoder, pool, concept_name, query=queries[concept_name])
                results.append(result)
                if (concept_idx + 1) % 10 == 0:
                    print(f"    retrieval [{concept_idx + 1}/{len(GROUNDABLE_CONCEPTS)}]", flush=True)

            for task_name in TARGET_CLASSES:
                black_box = TaskBlackBox(native_model, task_name, spec.preprocess, DEVICE)
                hybrid_results = score_masking_hybrid_concepts(
                    cfg, encoder, black_box, pool, GROUNDABLE_CONCEPTS, results)
                scores = {(CONCEPT_TO_IDX[c], 1): r.raw_score for c, r in hybrid_results.items()}

                rho_r = score_method_agreement(records_by_task[task_name], scores)
                sign_r = score_sign_agreement(records_by_task[task_name], scores)
                if rho_r is None:
                    print(f"    [{task_name}] too few pairs", flush=True)
                    continue
                print(f"    [{task_name}] n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} (p={rho_r.spearman_p:.4g})  "
                      f"sign={sign_r.agreement_frac:.1%} (p={sign_r.binom_p:.4g})", flush=True)
                all_rows.append((encoder_name, orth, False, task_name, rho_r.n_pairs, rho_r.spearman_rho,
                                  rho_r.spearman_p, sign_r.agreement_frac, sign_r.n_agree, sign_r.binom_p, "computed"))

    out_path = RESULTS_DIR / "cards_celeba_masking_hybrid_joint_orth_demean_encoder_ablation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["encoder", "orthogonalize", "demean_query", "target_task", "n_pairs",
                          "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p", "source"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows (8 reused + {len(all_rows) - 16} newly computed) to {out_path}")

    print("\n=== full 16-cell grid, Attractive task, sorted by |rho| ===")
    attractive_rows = [r for r in all_rows if r[3] == "Attractive"]
    for r in sorted(attractive_rows, key=lambda r: -abs(r[5])):
        print(f"  encoder={r[0]:<20s} orth={r[1]!s:<5s} demean={r[2]!s:<5s} rho={r[5]:+.4f} sign={r[7]:.1%} [{r[10]}]")


if __name__ == "__main__":
    main()
