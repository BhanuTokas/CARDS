"""Masking hybrid on the Male task, at v113's own established winning
config (SigLIP, orthogonalize=True, demean_query=True, K=50, z-score
threshold alpha=1.0, `baseline` phrasing, official-val pool) --
prompted directly ("Can we extend to add Male to our analysis?" ->
"Full 3-method comparison"). Scores against the NEW `celeba_male_
faithfulness_attribute_conditioned.csv` ground truth via the NEW
`celeba_attractive_young_male` backbone's own 3rd (Male) logit block.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
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

CELEBA_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebA\celeba")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
ALPHA = 1.0
SEED = 0
TASK_NAME = "Male"
MALE_POSITIVE_LOGIT_INDEX = 5
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}


class MaleBlackBox:
    def __init__(self, native_model, preprocess, device: str):
        self.model = native_model
        self._preprocess = preprocess
        self.device = device

    def preprocess(self, image):
        return self._preprocess(image.convert("RGB"))

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch.to(self.device))[:, MALE_POSITIVE_LOGIT_INDEX].detach().cpu()


def load_records() -> list[FaithfulnessResult]:
    records = []
    with open(RESULTS_DIR / "celeba_male_faithfulness_attribute_conditioned.csv", newline="") as f:
        for row in csv.DictReader(f):
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
    records = load_records()

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

    raw_queries = {c: build_concept_query(CONCEPT_QUERY_TEXT[c], encoder) for c in GROUNDABLE_CONCEPTS}
    demeaned = {c: demean_query(q, text_center) for c, q in raw_queries.items()}
    queries = orthogonalize_queries(demeaned)

    spec = BACKBONES["celeba_attractive_young_male"]
    native_model = spec.load_native().to(DEVICE).eval()
    black_box = MaleBlackBox(native_model, spec.preprocess, DEVICE)

    official_paths = build_clean_official_val_paths()
    pairs = [(p, 0) for p in official_paths]

    pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": cfg.encoder, "cache_dir": "embedding_cache"})
    pool_cfg.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}
    pool_cfg.pool_source = "val"
    pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images", flush=True)

    results = []
    for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
        result = process_concept(cfg, encoder, pool, concept_name, query=queries[concept_name])
        results.append(result)
        if (concept_idx + 1) % 10 == 0:
            print(f"  retrieval [{concept_idx + 1}/{len(GROUNDABLE_CONCEPTS)}]", flush=True)

    hybrid_results = score_masking_hybrid_concepts(cfg, encoder, black_box, pool, GROUNDABLE_CONCEPTS, results)
    scores = {(CONCEPT_TO_IDX[c], 1): r.raw_score for c, r in hybrid_results.items()}

    all_rows = [(c, r.raw_score) for c, r in hybrid_results.items()]
    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_male_raw_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "hybrid_raw_score"])
        writer.writerows(all_rows)

    rho_r = score_method_agreement(records, scores)
    sign_r = score_sign_agreement(records, scores)
    if rho_r is None:
        print("too few pairs", flush=True)
        return
    print(f"\n[{TASK_NAME}] n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} (p={rho_r.spearman_p:.4g})  "
          f"sign={sign_r.agreement_frac:.1%} ({sign_r.n_agree}/{sign_r.n_pairs}, p={sign_r.binom_p:.4g})", flush=True)

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_male_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_task", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        writer.writerow([TASK_NAME, rho_r.n_pairs, rho_r.spearman_rho, rho_r.spearman_p,
                          sign_r.agreement_frac, sign_r.n_agree, sign_r.binom_p])
    print("\nSaved results/cards_celeba_masking_hybrid_male_ablation.csv")


if __name__ == "__main__":
    main()
