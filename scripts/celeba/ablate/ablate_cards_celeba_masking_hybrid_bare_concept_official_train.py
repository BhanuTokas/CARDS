"""Bare-concept-only phrasing ablation for the masking hybrid on the
official-train CelebA classifier (ResNet18) -- prompted directly ("Can
we also run an ablation with just "<concept>" itself?"), the most
minimal phrasing condition tried in this whole prompt-ablation track:
NO template wrapping at all, just the noun phrase itself as the entire
query text (e.g. "eyeglasses", not "a person wearing eyeglasses" or
"a photo of eyeglasses").

`CONCEPT_NOUN_PHRASE` is the SAME per-concept bare noun-phrase mapping
from `ablate_cards_celeba_masking_hybrid_clip_templates_official_train.
py`, inlined here directly rather than cross-imported -- that script
lives on a different branch (`main`) than this one was written on
(`mask_strategy`), the same recurring cross-branch sync issue seen
throughout this project; every other single-purpose ablation script in
this directory is self-contained for exactly this reason, so this one
follows suit rather than depending on a script that may not be
checked out. No double-article/grammar concern here since there's no
template to double up with -- the noun phrase alone is passed straight
to `build_concept_query`.

Fixed config: SigLIP, orthogonalize=True, demean_query=**False**, K=50,
z-score alpha=1.0 -- same as every other official-train prompt/K/alpha
ablation. Only ONE new condition, computed fresh.
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

# Bare noun phrase per concept -- the ENTIRE query text for this ablation,
# no template wrapping. Inlined from the CLIP-templates ablation (see
# module docstring for why it's duplicated here rather than imported).
CONCEPT_NOUN_PHRASE: dict[str, str] = {
    "Arched_Eyebrows": "arched eyebrows",
    "Bushy_Eyebrows": "bushy eyebrows",
    "Bags_Under_Eyes": "bags under the eyes",
    "Narrow_Eyes": "narrow eyes",
    "Big_Nose": "a big nose",
    "Pointy_Nose": "a pointy nose",
    "Big_Lips": "big lips",
    "Wearing_Lipstick": "lipstick",
    "Mouth_Slightly_Open": "a slightly open mouth",
    "Smiling": "a smile",
    "Bald": "a bald head",
    "Bangs": "bangs",
    "Black_Hair": "black hair",
    "Blond_Hair": "blond hair",
    "Brown_Hair": "brown hair",
    "Gray_Hair": "gray hair",
    "Straight_Hair": "straight hair",
    "Wavy_Hair": "wavy hair",
    "Receding_Hairline": "a receding hairline",
    "Pale_Skin": "pale skin",
    "Rosy_Cheeks": "rosy cheeks",
    "Eyeglasses": "eyeglasses",
    "Wearing_Earrings": "earrings",
    "Wearing_Hat": "a hat",
    "Wearing_Necklace": "a necklace",
    "Wearing_Necktie": "a necktie",
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

    print("Bare-concept queries:", flush=True)
    for c in GROUNDABLE_CONCEPTS[:5]:
        print(f"  {c}: {CONCEPT_NOUN_PHRASE[c]!r}", flush=True)

    raw_queries = {c: build_concept_query(CONCEPT_NOUN_PHRASE[c], encoder) for c in GROUNDABLE_CONCEPTS}
    queries = orthogonalize_queries(raw_queries)

    results = []
    for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
        result = process_concept(cfg, encoder, pool, concept_name, query=queries[concept_name])
        results.append(result)
        if (concept_idx + 1) % 10 == 0:
            print(f"  retrieval [{concept_idx + 1}/{len(GROUNDABLE_CONCEPTS)}]", flush=True)

    all_rows = []  # (task, gt_version, n_pairs, rho, rho_p, sign_frac, n_agree, sign_p)
    raw_rows = []  # (task, concept_name, hybrid_raw_score)

    for task_name in TASKS:
        black_box = TaskBlackBox(native_model, task_name, spec.preprocess, DEVICE)
        hybrid_results = score_masking_hybrid_concepts(cfg, encoder, black_box, pool, GROUNDABLE_CONCEPTS, results)
        scores = {(CONCEPT_TO_IDX[c], 1): r.raw_score for c, r in hybrid_results.items()}
        for c, r in hybrid_results.items():
            raw_rows.append((task_name, c, r.raw_score))

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
            all_rows.append((task_name, gt_name, rho_r.n_pairs, rho_r.spearman_rho, rho_r.spearman_p,
                              sign_r.agreement_frac, sign_r.n_agree, sign_r.binom_p))

    out_path = RESULTS_DIR / "cards_celeba_masking_hybrid_bare_concept_official_train_ablation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_task", "gt_version", "n_pairs", "spearman_rho", "spearman_p",
                          "sign_agreement", "n_agree", "binom_p"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {out_path}", flush=True)

    raw_path = RESULTS_DIR / "cards_celeba_masking_hybrid_bare_concept_official_train_ablation_raw_scores.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_task", "concept_name", "hybrid_raw_score"])
        writer.writerows(raw_rows)
    print(f"Saved {len(raw_rows)} raw rows to {raw_path}", flush=True)

    rho_a = next(r[3] for r in all_rows if r[0] == "Attractive" and r[1] == "previously_used")
    rho_m = next(r[3] for r in all_rows if r[0] == "Male" and r[1] == "previously_used")
    print(f"\npooled_rho (previously_used) = {((rho_a + rho_m) / 2):+.4f}  "
          f"(Attractive={rho_a:+.4f} Male={rho_m:+.4f})")


if __name__ == "__main__":
    main()
