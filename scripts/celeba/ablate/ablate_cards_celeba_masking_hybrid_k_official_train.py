"""K ablation for the masking hybrid on the official-train CelebA
classifier (ResNet18) -- prompted directly ("Can you also do separate
ablations on K, alpha and prompt variations with the official set?").

Fixed config: SigLIP, orthogonalize=True, demean_query=**False**, z-score
alpha=1.0 -- the NEW best-known setting for this classifier, per
`ablate_cards_celeba_masking_hybrid_joint_orth_demean_encoder_official_
train.py`'s own 16-cell result (rho=+0.751 Attractive, the single best
cell of that whole grid), NOT the older demean_query=True default the
rest of the official-train pipeline still uses -- see that ablation's
own docstring for the full reasoning. If the main pipeline's own default
is later updated to match, this script's fixed config won't need to
change.

K=50's own cell is REUSED, not recomputed -- it's exactly the (siglip,
orth=True, demean=False) cell already computed by the joint ablation
(`cards_celeba_masking_hybrid_joint_orth_demean_encoder_official_train_
ablation_raw_scores.csv`), same "don't redo a K=50 pass you already
have" pattern the CelebA-HQ track's own K-ablation scripts established.
Only K=15 and K=30 are computed fresh here.

Built on the REAL cards.pipeline integration (process_concept +
score_masking_hybrid_concepts), matching every other official-train
ablation in this directory.
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

from cards.concepts.prompts import build_concept_query
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
BACKBONE_NAME = "celeba_official_train_attractive_male"  # ResNet18 only
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 1.0
SEED = 0
TASKS = ["Attractive", "Male"]
TASK_SLICE = {"Attractive": 1, "Male": 3}
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
GT_VERSIONS = {
    "non_overlapping": "celeba_official_train_faithfulness_non_overlapping.csv",
    "previously_used": "celeba_official_train_faithfulness_previously_used.csv",
}
K_VALUES_NEW = [15, 30]
K_REUSED = 50
REUSED_RAW_CSV = RESULTS_DIR / "cards_celeba_masking_hybrid_joint_orth_demean_encoder_official_train_ablation_raw_scores.csv"


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


def load_reused_k50_scores() -> dict[str, dict[tuple[int, int], float]]:
    scores_by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TASKS}
    with open(REUSED_RAW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["encoder"] != "siglip" or row["orthogonalize"] != "True" or row["demean_query"] != "False":
                continue
            key = (CONCEPT_TO_IDX[row["concept_name"]], 1)
            scores_by_task[row["target_task"]][key] = float(row["hybrid_raw_score"])
    return scores_by_task


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    spec = BACKBONES[BACKBONE_NAME]
    native_model = spec.load_native().to(DEVICE).eval()

    official_paths = build_clean_official_val_paths()
    pairs = [(p, 0) for p in official_paths]

    all_rows = []  # (K, task, gt_version, n_pairs, rho, rho_p, sign_frac, n_agree, sign_p, source)
    raw_rows = []  # (K, task, concept_name, hybrid_raw_score, source)

    # K=50, reused from the joint encoder ablation (same config, no recompute).
    print(f"\n{'=' * 20} K={K_REUSED} (reused from joint encoder ablation) {'=' * 20}", flush=True)
    reused_scores = load_reused_k50_scores()
    for task_name in TASKS:
        for c, score in reused_scores[task_name].items():
            raw_rows.append((K_REUSED, task_name, GROUNDABLE_CONCEPTS[c[0]], score, "reused"))
        for gt_name, gt_filename in GT_VERSIONS.items():
            records_task = load_records(gt_filename, task_name)
            rho_r = score_method_agreement(records_task, reused_scores[task_name])
            sign_r = score_sign_agreement(records_task, reused_scores[task_name])
            if rho_r is None:
                continue
            print(f"  [{task_name}/{gt_name}] n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} "
                  f"sign={sign_r.agreement_frac:.1%}", flush=True)
            all_rows.append((K_REUSED, task_name, gt_name, rho_r.n_pairs, rho_r.spearman_rho, rho_r.spearman_p,
                              sign_r.agreement_frac, sign_r.n_agree, sign_r.binom_p, "reused"))

    # New K values.
    cfg = OmegaConf.create({
        "seed": SEED, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": str(CACHE_DIR),
        "retrieval": {"strategy": "naive"}, "k": K_REUSED,
        "masking_hybrid": {"threshold_method": "zscore", "alpha": ALPHA},
    })
    encoder = instantiate_encoder(cfg)
    raw_queries = {c: build_concept_query(CONCEPT_QUERY_TEXT[c], encoder) for c in GROUNDABLE_CONCEPTS}
    queries = orthogonalize_queries(raw_queries)  # orth=True, demean=False (the new best config)

    pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": cfg.encoder, "cache_dir": str(CACHE_DIR)})
    pool_cfg.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}
    pool_cfg.pool_source = "val"
    pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
    print(f"\npool: {len(pool.paths)} images", flush=True)

    for K in K_VALUES_NEW:
        print(f"\n{'=' * 20} K={K} {'=' * 20}", flush=True)
        cfg.k = K

        results = []
        for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
            result = process_concept(cfg, encoder, pool, concept_name, query=queries[concept_name])
            results.append(result)
            if (concept_idx + 1) % 10 == 0:
                print(f"  retrieval [{concept_idx + 1}/{len(GROUNDABLE_CONCEPTS)}]", flush=True)

        for task_name in TASKS:
            black_box = TaskBlackBox(native_model, task_name, spec.preprocess, DEVICE)
            hybrid_results = score_masking_hybrid_concepts(cfg, encoder, black_box, pool, GROUNDABLE_CONCEPTS, results)
            scores = {(CONCEPT_TO_IDX[c], 1): r.raw_score for c, r in hybrid_results.items()}
            for c, r in hybrid_results.items():
                raw_rows.append((K, task_name, c, r.raw_score, "computed"))

            for gt_name, gt_filename in GT_VERSIONS.items():
                records_task = load_records(gt_filename, task_name)
                rho_r = score_method_agreement(records_task, scores)
                sign_r = score_sign_agreement(records_task, scores)
                if rho_r is None:
                    print(f"  [{task_name}/{gt_name}] too few pairs", flush=True)
                    continue
                print(f"  [{task_name}/{gt_name}] n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} "
                      f"(p={rho_r.spearman_p:.4g})  sign={sign_r.agreement_frac:.1%} (p={sign_r.binom_p:.4g})",
                      flush=True)
                all_rows.append((K, task_name, gt_name, rho_r.n_pairs, rho_r.spearman_rho, rho_r.spearman_p,
                                  sign_r.agreement_frac, sign_r.n_agree, sign_r.binom_p, "computed"))

    out_path = RESULTS_DIR / "cards_celeba_masking_hybrid_k_official_train_ablation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "target_task", "gt_version", "n_pairs", "spearman_rho", "spearman_p",
                          "sign_agreement", "n_agree", "binom_p", "source"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {out_path}", flush=True)

    raw_path = RESULTS_DIR / "cards_celeba_masking_hybrid_k_official_train_ablation_raw_scores.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "target_task", "concept_name", "hybrid_raw_score", "source"])
        writer.writerows(raw_rows)
    print(f"Saved {len(raw_rows)} raw rows to {raw_path}", flush=True)

    print("\n=== summary (non_overlapping GT), sorted by |rho| ===")
    non_overlap = [r for r in all_rows if r[2] == "non_overlapping"]
    for r in sorted(non_overlap, key=lambda r: -abs(r[4])):
        print(f"  K={r[0]:<3d} task={r[1]:<10s} rho={r[4]:+.4f} sign={r[6]:.1%} [{r[9]}]")


if __name__ == "__main__":
    main()
