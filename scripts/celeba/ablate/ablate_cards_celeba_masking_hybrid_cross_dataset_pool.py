"""Cross-dataset retrieval-pool generalization check for the masking
hybrid, prompted directly ("test if ConceptMask still works if we use a
different dataset for querying, i.e. FFHQ instead of celeba-official
validation set" -> FFHQ unavailable/too large to acquire quickly ->
"Let's try both LFW and UTK if possible?").

Every pool-size ablation so far (v115-v115c) still drew its candidate
pool from CelebA itself (official-val, a subset of the SAME source
distribution the ground-truth classifier trained on). This is a
genuinely different question: does ConceptMask's retrieval + masking
still work when the CANDIDATE IMAGES come from a completely different
face dataset/distribution (LFW's in-the-wild candid photos, UTKFace's
wide age/ethnicity range), never seen by SigLIP's retrieval space in
any CelebA-specific way and never seen by the celeba_attractive_young
classifier during training at all?

Ground truth stays the SAME CelebA-based corrected attribute-
conditioned faithfulness records throughout -- only the retrieval POOL
changes, matching every prior pool-swap script's own convention (v98's
official-val pool, v115's fractional subsamples). SigLIP queries/
winning config (orthogonalize=True, demean_query=True, K=50, zscore
alpha=1.0, `baseline` phrasing) held fixed at v113's own best config.

Full available image set used for each pool (no subsampling -- v115c
already showed pool size barely matters down to n=169, so there's no
reason to think either LFW's 13,233 or UTKFace's 23,708 own images
need trimming).
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

RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
ALPHA = 1.0
SEED = 0
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

POOLS = {
    "lfw": Path(r"C:\Users\btokas\Projects\Datasets\LFW\images"),
    "utkface": Path(r"C:\Users\btokas\Projects\Datasets\UTKFace\images"),
}


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

    raw_queries = {c: build_concept_query(CONCEPT_QUERY_TEXT[c], encoder) for c in GROUNDABLE_CONCEPTS}
    demeaned = {c: demean_query(q, text_center) for c, q in raw_queries.items()}
    queries = orthogonalize_queries(demeaned)

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()

    all_rows = []  # (pool_name, target_task, n_pool, n_pairs, rho, rho_p, sign_frac, n_agree, sign_p)
    raw_rows = []  # (pool_name, target_task, concept_name, hybrid_raw_score)

    for pool_name, image_dir in POOLS.items():
        image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        print(f"\n{'=' * 20} pool={pool_name} (n={len(image_paths)} images from {image_dir}) {'=' * 20}", flush=True)
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

        for task_name in TARGET_CLASSES:
            black_box = TaskBlackBox(native_model, task_name, spec.preprocess, DEVICE)
            hybrid_results = score_masking_hybrid_concepts(cfg, encoder, black_box, pool, GROUNDABLE_CONCEPTS, results)
            scores = {(CONCEPT_TO_IDX[c], 1): r.raw_score for c, r in hybrid_results.items()}
            for c, r in hybrid_results.items():
                raw_rows.append((pool_name, task_name, c, r.raw_score))

            rho_r = score_method_agreement(records_by_task[task_name], scores)
            sign_r = score_sign_agreement(records_by_task[task_name], scores)
            if rho_r is None:
                print(f"  [{task_name}] too few pairs", flush=True)
                continue
            print(f"  [{task_name}] n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} (p={rho_r.spearman_p:.4g})  "
                  f"sign={sign_r.agreement_frac:.1%} (p={sign_r.binom_p:.4g})", flush=True)
            all_rows.append((pool_name, task_name, len(pool.paths), rho_r.n_pairs, rho_r.spearman_rho,
                              rho_r.spearman_p, sign_r.agreement_frac, sign_r.n_agree, sign_r.binom_p))

    out_path = RESULTS_DIR / "cards_celeba_masking_hybrid_cross_dataset_pool_ablation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pool_name", "target_task", "n_pool", "n_pairs", "spearman_rho", "spearman_p",
                          "sign_agreement", "n_agree", "binom_p"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")

    raw_path = RESULTS_DIR / "cards_celeba_masking_hybrid_cross_dataset_pool_raw_scores.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pool_name", "target_task", "concept_name", "hybrid_raw_score"])
        writer.writerows(raw_rows)
    print(f"Saved {len(raw_rows)} raw per-concept rows to {raw_path}")

    print("\n=== summary, Attractive task, vs official-val's own 0.6704/0.6670 baseline ===")
    for r in [r for r in all_rows if r[1] == "Attractive"]:
        print(f"  pool={r[0]:<10s} n_pool={r[2]:<6d} rho={r[4]:+.4f} (p={r[5]:.4g})  sign={r[6]:.1%} (p={r[8]:.4g})")


if __name__ == "__main__":
    main()
