"""Cohen's d standardization (v120's own d_z = mean(delta_scores) /
std(delta_scores, ddof=1)) applied to the LFW/UTKFace cross-dataset
pools with the NEW official-train classifier, prompted directly ("Can
you recreate the pivoted table with the Male and Attractive collapsed
using Fisher and Cohen's d transform applied?" -- Cohen's d was only
computed for official-val so far; this fills in LFW/UTKFace so the
pivoted table's CARDS columns are consistently Cohen's-d-based).
Mirrors `ablate_cards_celeba_masking_hybrid_official_train_cross_
dataset.py` but captures per-image delta_scores instead of just the
aggregate raw mean.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT

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

RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
ALPHA = 1.0
SEED = 0
TASKS = ["Attractive", "Male"]
TASK_SLICE = {"Attractive": 1, "Male": 3}
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

POOLS = {
    "lfw": Path(r"C:\Users\btokas\Projects\Datasets\LFW\images"),
    "utkface": Path(r"C:\Users\btokas\Projects\Datasets\UTKFace\images"),
}
GT_VERSIONS = {
    "non_overlapping": "celeba_official_train_faithfulness_non_overlapping.csv",
    "previously_used": "celeba_official_train_faithfulness_previously_used.csv",
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

    spec = BACKBONES["celeba_official_train_attractive_male"]
    native_model = spec.load_native().to(DEVICE).eval()

    all_rows = []  # (pool, task, gt_version, metric, n_pairs, rho, rho_p, sign_frac, n_agree, sign_p)
    raw_rows = []  # (pool, task, concept_name, raw_mean, cohens_d, n_deltas)

    for pool_name, image_dir in POOLS.items():
        image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        print(f"\n{'=' * 20} pool={pool_name} (n={len(image_paths)} images) {'=' * 20}", flush=True)
        pairs = [(p, 0) for p in image_paths]

        pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": cfg.encoder, "cache_dir": "embedding_cache"})
        pool_cfg.dataset = {"name": f"{pool_name}_pool", "root": str(image_dir)}
        pool_cfg.pool_source = "val"
        pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
        print(f"  pool built/loaded: {len(pool.paths)} images", flush=True)

        results = []
        for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
            result = process_concept(cfg, encoder, pool, concept_name, query=queries[concept_name])
            results.append(result)
            if (concept_idx + 1) % 10 == 0:
                print(f"  retrieval [{concept_idx + 1}/{len(GROUNDABLE_CONCEPTS)}]", flush=True)

        for task_name in TASKS:
            black_box = TaskBlackBox(native_model, task_name, spec.preprocess, DEVICE)
            hybrid_results = score_masking_hybrid_concepts(cfg, encoder, black_box, pool, GROUNDABLE_CONCEPTS, results)

            raw_scores, cohens_d_scores = {}, {}
            for c, r in hybrid_results.items():
                deltas = np.array(r.delta_scores)
                raw_mean = float(np.mean(deltas)) if len(deltas) else 0.0
                sd = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
                d_z = raw_mean / sd if sd > 1e-12 else 0.0
                raw_scores[(CONCEPT_TO_IDX[c], 1)] = raw_mean
                cohens_d_scores[(CONCEPT_TO_IDX[c], 1)] = d_z
                raw_rows.append((pool_name, task_name, c, raw_mean, d_z, len(deltas)))

            for gt_name, gt_filename in GT_VERSIONS.items():
                records = load_records(gt_filename, task_name)
                for metric_name, scores in [("raw_mean", raw_scores), ("cohens_d", cohens_d_scores)]:
                    rho_r = score_method_agreement(records, scores)
                    sign_r = score_sign_agreement(records, scores)
                    if rho_r is None:
                        print(f"  [{task_name}/{gt_name}/{metric_name}] too few pairs", flush=True)
                        continue
                    print(f"  [{task_name}/{gt_name}/{metric_name}] n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} "
                          f"(p={rho_r.spearman_p:.4g})  sign={sign_r.agreement_frac:.1%} (p={sign_r.binom_p:.4g})", flush=True)
                    all_rows.append((pool_name, task_name, gt_name, metric_name, rho_r.n_pairs, rho_r.spearman_rho,
                                      rho_r.spearman_p, sign_r.agreement_frac, sign_r.n_agree, sign_r.binom_p))

    out_path = RESULTS_DIR / "cards_celeba_masking_hybrid_official_train_cross_dataset_cohens_d.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pool", "task", "gt_version", "metric", "n_pairs", "spearman_rho", "spearman_p",
                          "sign_agreement", "n_agree", "binom_p"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")

    raw_path = RESULTS_DIR / "cards_celeba_masking_hybrid_official_train_cross_dataset_cohens_d_raw.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pool", "task", "concept_name", "raw_mean", "cohens_d", "n_deltas"])
        writer.writerows(raw_rows)
    print(f"Saved {len(raw_rows)} raw rows to {raw_path}")


if __name__ == "__main__":
    main()
