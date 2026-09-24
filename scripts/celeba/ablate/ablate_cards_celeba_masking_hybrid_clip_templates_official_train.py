"""Generic CLIP-style template ablation for the masking hybrid on the
official-train CelebA classifier (ResNet18) -- prompted directly ("Can
you try it with these templates? templates=['a photo of {concept}',
'a photo of a {concept}', 'an image with a {concept} in it']"), a
third phrasing style alongside the existing hand-crafted baseline/alt
sentences (`ablate_cards_celeba_masking_hybrid_prompt_official_train.
py`) -- these 3 are generic template SLOTS filled with a per-concept
noun phrase, not full hand-written sentences.

Fixed config: SigLIP, orthogonalize=True, demean_query=**False**, K=50,
z-score alpha=1.0 -- the same best-known setting used by the other two
official-train prompt/K/alpha ablations.

**CONCEPT_NOUN_PHRASE is a NEW mapping, not reused from anywhere** --
derived directly from each concept's own baseline sentence (stripping
the "a person with/wearing" scaffolding down to just the noun phrase,
e.g. Arched_Eyebrows's "a person with arched eyebrows" -> "arched
eyebrows"), since these 3 templates need a noun phrase to fill their
own {concept} slot, not the full baseline/alt sentences. Worth noting
directly rather than silently: template 2 ("a photo of a {concept}")
assumes a singular countable noun, which several concepts genuinely
aren't (e.g. "arched eyebrows", "narrow eyes", "black hair" are
plural/mass nouns) -- this produces grammatically rough prompts like
"a photo of a arched eyebrows" for those concepts. Left as-is rather
than special-cased per concept, since the whole point of testing
generic templates is exactly that they're generic (not hand-tuned
per concept the way baseline/alt already are) -- SigLIP/CLIP encoders
are trained on enough noisy alt-text that they're generally robust to
this, but it's a known, real property of this specific template set,
not a bug to silently paper over.

None of these 3 conditions can reuse anything from the joint encoder
ablation (that only covers the ORIGINAL baseline phrasing) -- all 3
templates are computed fresh here.
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

TEMPLATES = [
    "a photo of {concept}",
    "a photo of a {concept}",
    "an image with a {concept} in it",
]

# Per-concept noun phrase to fill {concept} -- derived from CONCEPT_QUERY_TEXT's
# own baseline sentences, stripped down to just the noun phrase (see module
# docstring for the "template 2 assumes singular" caveat this produces).
# 9/26 phrases start with their own "a " (e.g. "a big nose", "a smile") --
# templates 2/3 ALSO supply their own article ("a photo of a {concept}"), so
# using these as-is there would literally double up ("a photo of a a big
# nose"). Fixed below via CONCEPT_NOUN_PHRASE_BARE, not by rewriting these.
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

# Article-stripped version for templates 2/3, which already supply their own
# "a" -- e.g. "a big nose" -> "big nose", so "a photo of a {concept}" comes
# out as "a photo of a big nose", not "a photo of a a big nose".
CONCEPT_NOUN_PHRASE_BARE: dict[str, str] = {
    c: phrase[2:] if phrase.startswith("a ") else phrase
    for c, phrase in CONCEPT_NOUN_PHRASE.items()
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

    all_rows = []  # (template, task, gt_version, n_pairs, rho, rho_p, sign_frac, n_agree, sign_p)
    raw_rows = []  # (template, task, concept_name, hybrid_raw_score)

    for template in TEMPLATES:
        print(f"\n{'=' * 20} template={template!r} {'=' * 20}", flush=True)
        # template 1 has no article of its own -- use the phrase as-is (with
        # whatever article it naturally carries); templates 2/3 already
        # supply "a", so use the article-stripped phrase to avoid doubling up.
        phrase_map = CONCEPT_NOUN_PHRASE if template == TEMPLATES[0] else CONCEPT_NOUN_PHRASE_BARE
        raw_queries = {
            c: build_concept_query(template.format(concept=phrase_map[c]), encoder)
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

    out_path = RESULTS_DIR / "cards_celeba_masking_hybrid_clip_templates_official_train_ablation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["template", "target_task", "gt_version", "n_pairs", "spearman_rho", "spearman_p",
                          "sign_agreement", "n_agree", "binom_p"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {out_path}", flush=True)

    raw_path = RESULTS_DIR / "cards_celeba_masking_hybrid_clip_templates_official_train_ablation_raw_scores.csv"
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
