"""Prompt-phrasing ablation for the masking hybrid on CelebA official-val,
at the now-confirmed winning config (SigLIP, orthogonalize=True,
demean_query=True, K=50, z-score threshold alpha=1.0 --
`ablate_cards_celeba_masking_hybrid_joint_orth_demean_encoder.py`'s own
v113 result, rho=+0.6704, the best cell of that 16-cell grid).

Scope, decided directly with the user ("Do you test across all prompts
or only 1 prompt vs ensemble?" -> offered baseline-vs-ensemble-of-2
(cheap, reuses `ablate_cards_celeba_masking_hybrid_query.py`'s own
ALT_CONCEPT_QUERY_TEXT) vs a per-phrasing sweep vs both -> "Both"):
three conditions, not the full 2x2x2 grid that script's own docstring
mentions (demean/orthogonalize are ALREADY settled by the joint
ablation, held fixed here at their winning values, not re-swept):
  - "baseline": CONCEPT_QUERY_TEXT only.
  - "alt": ALT_CONCEPT_QUERY_TEXT only (the individual-phrasing
    condition -- there is exactly one alt phrasing per concept, so
    "each phrasing tested individually" reduces to this single
    alt-alone run, not a larger sweep).
  - "ensemble": mean of baseline + alt (L2-renormalized), matching
    the query-ablation script's own `phrasing="ensemble"` definition.

Built on the REAL cards.pipeline integration (process_concept +
score_masking_hybrid_concepts), official-val pool, and the CORRECTED
attribute-conditioned ground truth -- matching v113's own setup, NOT
`ablate_cards_celeba_masking_hybrid_query.py`'s older HQ-val pool +
top_pct threshold + original (uncorrected) ground truth.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from ablate_cards_celeba_masking_hybrid_query import ALT_CONCEPT_QUERY_TEXT
from run_cards_celeba_full import CONCEPT_QUERY_TEXT, TASK_POSITIVE_LOGIT_INDEX
from run_cards_celeba_masking_hybrid_official_val_zscore import build_clean_official_val_paths

from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
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
PHRASINGS = ["baseline", "alt", "ensemble"]


class TaskBlackBox:
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


def build_query(concept: str, encoder, phrasing: str, text_center: torch.Tensor) -> torch.Tensor:
    if phrasing == "baseline":
        t_c = build_concept_query(CONCEPT_QUERY_TEXT[concept], encoder)
    elif phrasing == "alt":
        t_c = build_concept_query(ALT_CONCEPT_QUERY_TEXT[concept], encoder)
    else:
        embeddings = torch.stack([
            build_concept_query(CONCEPT_QUERY_TEXT[concept], encoder),
            build_concept_query(ALT_CONCEPT_QUERY_TEXT[concept], encoder),
        ])
        t_c = F.normalize(embeddings.mean(dim=0), dim=0)
    return demean_query(t_c, text_center)


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

    cfg = OmegaConf.create({
        "seed": SEED, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
        "retrieval": {"strategy": "naive"}, "k": K,
        "masking_hybrid": {"threshold_method": "zscore", "alpha": ALPHA},
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()

    official_paths = build_clean_official_val_paths()
    pairs = [(p, 0) for p in official_paths]

    pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": cfg.encoder, "cache_dir": "embedding_cache"})
    pool_cfg.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}
    pool_cfg.pool_source = "val"
    pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images", flush=True)

    all_rows = []  # (phrasing, target_task, n_pairs, rho, rho_p, sign_frac, n_agree, sign_p)
    raw_rows = []  # (phrasing, concept_name, target_task, hybrid_raw_score)

    for phrasing in PHRASINGS:
        print(f"\n{'=' * 20} phrasing={phrasing} {'=' * 20}", flush=True)
        raw_queries = {c: build_query(c, encoder, phrasing, text_center) for c in GROUNDABLE_CONCEPTS}
        queries = orthogonalize_queries(raw_queries)

        results = []
        for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
            result = process_concept(cfg, encoder, pool, concept_name, query=queries[concept_name])
            results.append(result)
            if (concept_idx + 1) % 10 == 0:
                print(f"  retrieval [{concept_idx + 1}/{len(GROUNDABLE_CONCEPTS)}]", flush=True)

        for task_name in TARGET_CLASSES:
            black_box = TaskBlackBox(native_model, task_name, spec.preprocess, DEVICE)
            hybrid_results = score_masking_hybrid_concepts(cfg, encoder, black_box, pool, GROUNDABLE_CONCEPTS, results)
            scores = {(CONCEPT_TO_IDX[c], 1): r.raw_score for c, r in hybrid_results.items()}
            for c, r in hybrid_results.items():
                raw_rows.append((phrasing, c, task_name, r.raw_score))

            rho_r = score_method_agreement(records_by_task[task_name], scores)
            sign_r = score_sign_agreement(records_by_task[task_name], scores)
            if rho_r is None:
                print(f"  [{task_name}] too few pairs", flush=True)
                continue
            print(f"  [{task_name}] n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} (p={rho_r.spearman_p:.4g})  "
                  f"sign={sign_r.agreement_frac:.1%} (p={sign_r.binom_p:.4g})", flush=True)
            all_rows.append((phrasing, task_name, rho_r.n_pairs, rho_r.spearman_rho, rho_r.spearman_p,
                              sign_r.agreement_frac, sign_r.n_agree, sign_r.binom_p))

    out_path = RESULTS_DIR / "cards_celeba_masking_hybrid_prompt_ablation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phrasing", "target_task", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        writer.writerows(all_rows)

    raw_path = RESULTS_DIR / "cards_celeba_masking_hybrid_prompt_ablation_raw_scores.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phrasing", "concept_name", "target_task", "hybrid_raw_score"])
        writer.writerows(raw_rows)

    print(f"\nSaved {len(all_rows)} rows to {out_path}")
    print("\n=== summary, Attractive task ===")
    for r in sorted([r for r in all_rows if r[1] == "Attractive"], key=lambda r: -abs(r[3])):
        print(f"  phrasing={r[0]:<10s} rho={r[3]:+.4f} (p={r[4]:.4g})  sign={r[5]:.1%} (p={r[7]:.4g})")


if __name__ == "__main__":
    main()
