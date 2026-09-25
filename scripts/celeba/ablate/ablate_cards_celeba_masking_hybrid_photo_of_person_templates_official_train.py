"""Second CLIP-style template ablation for the masking hybrid on the
official-train CelebA classifier (ResNet18) -- prompted directly
("Can we try the prompt variation experiment, but some changes to the
prompt" -> "Fix the noun-phrase issue and modify them to be more like
'a photo of a person with eyeglasses.' instead 'a photo of a
eyeglasses.'").

Fixes `ablate_cards_celeba_masking_hybrid_clip_templates_official_
train.py`'s own known grammar issue (templates 2/3 there assumed a
bare singular noun, breaking for plural/mass-noun concepts like
"arched eyebrows" or "eyeglasses") by filling {concept} with the FULL
baseline SUBJECT PHRASE from `CONCEPT_QUERY_TEXT` (e.g. "a person
wearing eyeglasses", not just "eyeglasses") -- that phrase was already
hand-crafted per concept to read grammatically, so wrapping it in a
generic "a photo of ..." prefix can't reintroduce the same bug.

Fixed config: SigLIP, orthogonalize=True, demean_query=**False**, K=50,
z-score alpha=1.0 -- same as every other official-train prompt/K/alpha
ablation in this directory.

None of these 3 conditions reuse anything from the joint encoder
ablation or the earlier CLIP-templates ablation -- all 3 are new
phrasings, computed fresh.
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
BACKBONE_NAME = "celeba_official_train_attractive_male"
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

# {concept} filled with CONCEPT_QUERY_TEXT's own full subject phrase
# (e.g. "a person wearing eyeglasses"), not a bare noun -- see module
# docstring for why this avoids the earlier template ablation's
# grammar bug.
TEMPLATES = [
    "a photo of {concept}.",
    "a photo showing {concept}.",
    "an image of {concept}.",
]


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
        "cache_dir": str(CACHE_DIR),
        "retrieval": {"strategy": "naive"}, "k": K,
        "masking_hybrid": {"threshold_method": "zscore", "alpha": ALPHA},
    })
    encoder = instantiate_encoder(cfg)

    spec = BACKBONES[BACKBONE_NAME]
    native_model = spec.load_native().to(DEVICE).eval()

    official_paths = build_clean_official_val_paths()
    pairs = [(p, 0) for p in official_paths]

    pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": cfg.encoder, "cache_dir": str(CACHE_DIR)})
    pool_cfg.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}
    pool_cfg.pool_source = "val"
    pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images", flush=True)

    all_rows = []  # (template, task, gt_version, n_pairs, rho, rho_p, sign_frac, n_agree, sign_p)
    raw_rows = []  # (template, task, concept_name, hybrid_raw_score)

    for template in TEMPLATES:
        print(f"\n{'=' * 20} template={template!r} {'=' * 20}", flush=True)
        raw_queries = {
            c: build_concept_query(template.format(concept=CONCEPT_QUERY_TEXT[c]), encoder)
            for c in GROUNDABLE_CONCEPTS
        }
        queries = orthogonalize_queries(raw_queries)

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
                raw_rows.append((template, task_name, c, r.raw_score))

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
                all_rows.append((template, task_name, gt_name, rho_r.n_pairs, rho_r.spearman_rho, rho_r.spearman_p,
                                  sign_r.agreement_frac, sign_r.n_agree, sign_r.binom_p))

    out_path = RESULTS_DIR / "cards_celeba_masking_hybrid_photo_of_person_templates_official_train_ablation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["template", "target_task", "gt_version", "n_pairs", "spearman_rho", "spearman_p",
                          "sign_agreement", "n_agree", "binom_p"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {out_path}", flush=True)

    raw_path = RESULTS_DIR / "cards_celeba_masking_hybrid_photo_of_person_templates_official_train_ablation_raw_scores.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["template", "target_task", "concept_name", "hybrid_raw_score"])
        writer.writerows(raw_rows)
    print(f"Saved {len(raw_rows)} raw rows to {raw_path}", flush=True)

    print("\n=== pooled (naive mean, Attractive+Male), previously_used GT ===")
    for template in TEMPLATES:
        rho_a = next(r[4] for r in all_rows if r[0] == template and r[1] == "Attractive" and r[2] == "previously_used")
        rho_m = next(r[4] for r in all_rows if r[0] == template and r[1] == "Male" and r[2] == "previously_used")
        print(f"  {template!r:<40s} pooled_rho={((rho_a + rho_m) / 2):+.4f}  (Attractive={rho_a:+.4f} Male={rho_m:+.4f})")


if __name__ == "__main__":
    main()
