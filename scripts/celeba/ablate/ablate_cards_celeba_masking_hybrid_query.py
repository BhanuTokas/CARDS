"""2x2x2 ablation of the masking hybrid's own query construction --
demean_query x orthogonalize x prompt-phrasing -- prompted directly
("Should we ablate orthogonalization, demeaning and prompt variations?"
-> "Full 2x2x2 grid, both datasets" -> then scoped down to "let's try
demean ablation first"). Everything downstream of the query (retrieval,
patch-similarity localization, best-of-7-strategy masking, classifier
scoring) is UNCHANGED across all 8 configs -- only how `t_c` itself gets
built varies, isolating this as a query-construction-only ablation.

CURRENTLY SCOPED to demean_query only (`configs` below is a 1-item
list, not the full itertools.product(...) grid) -- every prior hybrid
run (v67 CUB, v81/v82 CelebA) already applies demean_query
UNCONDITIONALLY, so those existing results already ARE the demean=True
condition; only demean=False needs a fresh run here, compared against
the existing CSV as baseline rather than recomputing it. Expand
`configs` back to the full grid (orthogonalize, phrasing) as a later
step if this axis looks worth pursuing further.

- demean_query: subtracts GENERIC_REFERENCE_CONCEPTS' text center before
  use (already CARDS' production default for retrieval; never tested
  for the LOCALIZATION use case specifically before).
- orthogonalize: jointly Lowdin-orthogonalizes ALL 26 concepts' queries
  at once (`cards.pipeline.orthogonalize_queries`, reused directly, not
  reimplemented) -- needs every query built up front, so this script
  restructures the per-concept loop into "build all queries for this
  config, then loop concepts" rather than building-and-immediately-using
  one query at a time.
- phrasing: "baseline" (CONCEPT_QUERY_TEXT) vs "ensemble" (averaged with
  ALT_CONCEPT_QUERY_TEXT below, a second, differently-structured phrasing
  per concept -- mirrors CUB's own ALT_PREFIX_TEMPLATES precedent from
  `ablate_cards_cub_attributes.py`).

8 configs x 26 concepts x 50 images each -- the same per-image cost as
one full hybrid run, x8.
"""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from localize_concept_patches_celeba import patch_similarity_grid, upsample_to_mask
from run_cards_celeba_full import CONCEPT_QUERY_TEXT, TASK_POSITIVE_LOGIT_INDEX

from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS, TARGET_CLASSES
from cards.data.datasets import load_celeba
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import FaithfulnessResult, mask_region, score_method_agreement, score_sign_agreement

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
TOP_PCT = 15
SEED = 42
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}

# A second, differently-structured phrasing per concept -- mirrors CUB's
# own ALT_PREFIX_TEMPLATES precedent (ablate_cards_cub_attributes.py).
ALT_CONCEPT_QUERY_TEXT: dict[str, str] = {
    "Arched_Eyebrows": "someone whose eyebrows are arched",
    "Bushy_Eyebrows": "someone with thick, bushy eyebrows",
    "Bags_Under_Eyes": "someone with under-eye bags",
    "Narrow_Eyes": "someone whose eyes are narrow",
    "Big_Nose": "someone with a large nose",
    "Pointy_Nose": "someone whose nose comes to a point",
    "Big_Lips": "someone with full, large lips",
    "Wearing_Lipstick": "someone with lipstick on",
    "Mouth_Slightly_Open": "someone whose mouth is slightly parted",
    "Smiling": "someone smiling",
    "Bald": "someone who is bald",
    "Bangs": "someone with a fringe of bangs",
    "Black_Hair": "someone with hair that is black",
    "Blond_Hair": "someone with hair that is blond",
    "Brown_Hair": "someone with hair that is brown",
    "Gray_Hair": "someone with hair that is gray",
    "Straight_Hair": "someone with straight, unwavy hair",
    "Wavy_Hair": "someone with wavy, curly hair",
    "Receding_Hairline": "someone whose hairline is receding",
    "Pale_Skin": "someone with fair, pale skin",
    "Rosy_Cheeks": "someone with rosy, flushed cheeks",
    "Eyeglasses": "someone wearing glasses",
    "Wearing_Earrings": "someone with earrings on",
    "Wearing_Hat": "someone with a hat on their head",
    "Wearing_Necklace": "someone with a necklace on",
    "Wearing_Necktie": "someone wearing a tie",
}


def build_query(concept: str, encoder, phrasing: str, text_center: torch.Tensor | None) -> torch.Tensor:
    if phrasing == "baseline":
        t_c = build_concept_query(CONCEPT_QUERY_TEXT[concept], encoder)
    else:
        embeddings = torch.stack([
            build_concept_query(CONCEPT_QUERY_TEXT[concept], encoder),
            build_concept_query(ALT_CONCEPT_QUERY_TEXT[concept], encoder),
        ])
        t_c = F.normalize(embeddings.mean(dim=0), dim=0)
    if text_center is not None:
        t_c = demean_query(t_c, text_center)
    return t_c


def load_records_by_task() -> dict[str, list[FaithfulnessResult]]:
    by_task: dict[str, list[FaithfulnessResult]] = {t: [] for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "celeba_full_faithfulness.csv", newline="") as f:
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
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
    })
    encoder = instantiate_encoder(cfg)
    siglip_model, siglip_preprocess = encoder.model, encoder.preprocess
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    cfg.dataset = {"name": "celeba", "root": str(CELEBA_HQ_ROOT)}
    cfg.pool_source = "val"
    pairs = load_celeba(CELEBA_HQ_ROOT, split="val")
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images", flush=True)

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()

    results = []  # (config_label, task_name, rho_result, sign_result)
    all_rows = []  # (config_label, concept_name, target_task, hybrid_raw_score)

    # Phrasing-only scope now (demean v86, orthogonalize v87 already
    # ablated separately) -- demean=True/orthogonalize=False held fixed,
    # matching the ORIGINAL v82 baseline, so this isolates the phrasing
    # effect cleanly rather than conflating it with orthogonalize's
    # already-known effect. Restore to itertools.product([False, True],
    # [False, True], ["baseline", "ensemble"]) for the full 2x2x2 grid.
    configs = [(True, False, "ensemble")]
    for use_demean, use_orth, phrasing in configs:
        label = f"demean{use_demean}_orth{use_orth}_{phrasing}"
        print(f"\n=== {label} ===", flush=True)

        tc_for_demean = text_center if use_demean else None
        raw_queries = {c: build_query(c, encoder, phrasing, tc_for_demean) for c in GROUNDABLE_CONCEPTS}
        queries = orthogonalize_queries(raw_queries) if use_orth else raw_queries

        hybrid_scores_by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}

        for concept_name in GROUNDABLE_CONCEPTS:
            t_c = queries[concept_name]
            t_c_dev = t_c.to(DEVICE)

            present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
            delta_logits = {t: [] for t in TARGET_CLASSES}

            for idx in present_indices:
                image = Image.open(pool.paths[idx]).convert("RGB")
                sim_grid = patch_similarity_grid(siglip_model, siglip_preprocess, image, t_c)
                sim_map = upsample_to_mask(sim_grid, (image.height, image.width))
                thresh = np.percentile(sim_map, 100 - TOP_PCT)
                mask = sim_map >= thresh
                if not mask.any() or mask.all():
                    continue

                rng = np.random.default_rng(SEED + CONCEPT_TO_IDX[concept_name] * 10_000 + int(idx))
                candidates = [mask_region(image, mask, strategy=s, rng=rng) for s in FILL_STRATEGIES]
                with torch.no_grad():
                    embeds = encoder.encode_images([image] + candidates).to(DEVICE)
                embed_orig = embeds[0]
                best_angle, best_i = None, None
                for i in range(len(FILL_STRATEGIES)):
                    diff = embed_orig - embeds[1 + i]
                    diff_unit = diff / diff.norm()
                    cos_sim = float(torch.clamp(diff_unit @ t_c_dev, -1.0, 1.0))
                    angle_deg = float(np.degrees(np.arccos(cos_sim)))
                    if best_angle is None or angle_deg < best_angle:
                        best_angle, best_i = angle_deg, i
                masked_image = candidates[best_i]

                pixels_orig = spec.preprocess(image).unsqueeze(0)
                pixels_masked = spec.preprocess(masked_image).unsqueeze(0)
                batch = torch.cat([pixels_orig, pixels_masked], dim=0).to(DEVICE)
                with torch.no_grad():
                    logits = native_model(batch)

                for task_name in TARGET_CLASSES:
                    task_idx = TASK_POSITIVE_LOGIT_INDEX[task_name]
                    delta_logits[task_name].append((logits[0, task_idx] - logits[1, task_idx]).item())

            for task_name in TARGET_CLASSES:
                score = float(np.mean(delta_logits[task_name]))
                hybrid_scores_by_task[task_name][(CONCEPT_TO_IDX[concept_name], 1)] = score
                all_rows.append((label, concept_name, task_name, score))

        for task_name in TARGET_CLASSES:
            rho_result = score_method_agreement(records_by_task[task_name], hybrid_scores_by_task[task_name])
            sign_result = score_sign_agreement(records_by_task[task_name], hybrid_scores_by_task[task_name])
            results.append((label, task_name, rho_result, sign_result))
            if rho_result is not None:
                print(f"  [{task_name}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} "
                      f"p={rho_result.spearman_p:.4g} | sign={sign_result.agreement_frac:.1%} "
                      f"({sign_result.n_agree}/{sign_result.n_pairs}) binom_p={sign_result.binom_p:.4g}", flush=True)

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_query_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "target_task", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for label, task_name, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([label, task_name, "", "", "", "", "", ""])
            else:
                writer.writerow([label, task_name, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_query_ablation_raw_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "concept_name", "target_task", "hybrid_raw_score"])
        writer.writerows(all_rows)

    print(f"\nSaved {len(results)} (config, task) rows to results/cards_celeba_masking_hybrid_query_ablation.csv")


if __name__ == "__main__":
    main()
