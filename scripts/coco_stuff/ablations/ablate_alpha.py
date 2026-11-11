"""Ablation of z-score localization threshold (alpha) for COCO-Stuff.

Mirrors scripts/celeba/ablate/ablate_cards_celeba_masking_hybrid_threshold_zscore.py
but for the 143 Broden concepts × 20 COCO-Stuff biased classes.

Design:
  For each concept:
    Pass 1 — localize all K=50 present-set images ONCE (cached). Sim-maps
             are identical across all alpha values so this avoids 5x
             redundant encoder forward passes.
    Pass 2 — for each alpha in ALPHA_VALUES: compute concept_zscore_cutoff
             on the pooled sim-maps, threshold each cached map, mask/fill/
             score with best-of-7.
  After all concepts: evaluate every (alpha, class) combination against the
  pre-computed faithfulness CSV and report Spearman rho + sign agreement.

Prerequisites:
  - scripts/coco_stuff/build/train_coco_stuff_classifier.py (ResNet18 ckpt)
  - scripts/coco_stuff/build/compute_coco_stuff_faithfulness.py (faith CSV)
  - CARDS pool embeddings (built on first run of any run_cards_* script)

Outputs (in results/coco_stuff/):
  alpha_ablation_scores.csv           -- (alpha, concept, class, raw_score)
  alpha_ablation_summary.csv          -- (alpha, class, n_pairs, rho, rho_p, sign, sign_p,
                                          pooled_rho, pooled_p, pooled_n)
  alpha_ablation_cutoffs.csv          -- (alpha, concept, cutoff, mean_area_pct)
  alpha_ablation_per_image_scores.csv -- (alpha, concept, image_path, class, delta_logit)

Usage (interactive GPU node on SOL):
    module load gcc/12.5.0 && module load cuda-12.4.1-gcc-12.1.0
    cd /scratch/rnair21/CARDS
    export PYTHONNOUSERSITE=1
    /scratch/rnair21/envs/cards_env/bin/python \\
        scripts/coco_stuff/ablations/ablate_alpha.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from scipy import stats

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.coco_stuff import CO_OCCUR_BRODEN_CONCEPTS, COCO_STUFF_20_CLASSES
from cards.data.datasets import list_broden_concepts, load_coco_stuff
from cards.models.coco_stuff_black_box import _coco_stuff_preprocess
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import mask_region

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COCO_ROOT   = Path("/scratch/rnair21/LLMConceptEditing_old/coco")
BRODEN_ROOT = Path("/scratch/rnair21/LLMConceptEditing_old/broden_concepts")
CKPT_PATH   = Path("trained_models_new/coco_stuff/resnet18_coco_stuff.pt")
RESULTS_DIR = Path("results/coco_stuff")
FAITH_CSV   = RESULTS_DIR / "coco_stuff_faithfulness.csv"

BRODEN_DROPPED_CONCEPTS: set[str] = {
    "air_conditioner", "apron", "awning", "bathroom_s", "beak", "bedroom_s",
    "blotchy", "bumper", "can", "candlestick", "cap", "clock", "dining_room_s",
    "drinking_glass", "eye", "eyebrow", "faucet", "figurine", "handle",
    "inside_arm", "jar", "knob", "loudspeaker", "mouse", "mouth",
    "outside_arm", "street_s",
}

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
K              = 50
SEED           = 42
ALPHA_VALUES   = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50]
MIN_SAMPLES    = 3    # minimum faithfulness records per (concept, class) for rho
MIN_FAITH_IMGS = 100  # minimum unique images for a concept to be included in rho

FILL_STRATEGIES = [
    "blur", "zero_fill", "mean_fill", "hue_shift",
    "white_fill", "zero_fill_noise", "noise_then_blur",
]


# ---------------------------------------------------------------------------
# Faithfulness loader
# ---------------------------------------------------------------------------

STRATEGIES = ("black_mask", "black_bbox")


def load_faithfulness(
    path: Path, strategy: str = "black_mask"
) -> tuple[dict[tuple[str, str], list[float]], dict[str, int]]:
    col = f"delta_p_{strategy}"
    # pass 1: collect unique images per concept
    concept_imgs: dict[str, set[str]] = defaultdict(set)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            concept_imgs[row["concept_name"]].add(row["image_path"])
    # select exactly MIN_FAITH_IMGS images per qualifying concept (sorted = deterministic)
    valid_imgs: dict[str, set[str]] = {
        c: set(sorted(imgs)[:MIN_FAITH_IMGS])
        for c, imgs in concept_imgs.items()
        if len(imgs) >= MIN_FAITH_IMGS
    }
    # pass 2: load faithfulness for selected images only
    faith: dict[tuple[str, str], list[float]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            c, cl, img = row["concept_name"], row["class_name"], row["image_path"]
            if c in valid_imgs and img in valid_imgs[c]:
                faith[(c, cl)].append(float(row[col]))
    image_counts = {c: MIN_FAITH_IMGS for c in valid_imgs}
    return dict(faith), image_counts


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _sign_pval(n_agree: int, n: int) -> float:
    return stats.binomtest(n_agree, n, p=0.5, alternative="two-sided").pvalue


def evaluate_alpha(
    scores:       dict[tuple[str, str], float],
    faith:        dict[tuple[str, str], list[float]],
    image_counts: dict[str, int],
) -> tuple[list[dict], float, float, float, float, int]:
    """Per-class rho/sign rows + mean rho + mean sign + pooled rho + pooled p + n_pooled."""
    valid = {c for c, n in image_counts.items() if n >= MIN_FAITH_IMGS}
    rows: list[dict] = []
    class_rhos, class_signs = [], []
    pooled_s, pooled_g = [], []

    for cls_name in COCO_STUFF_20_CLASSES:
        cards_vals, gt_vals = [], []
        for key, dp_list in faith.items():
            if key[1] != cls_name or key[0] not in valid or len(dp_list) < MIN_SAMPLES:
                continue
            if key not in scores:
                continue
            cards_vals.append(scores[key])
            gt_vals.append(float(np.mean(dp_list)))
        pooled_s.extend(cards_vals)
        pooled_g.extend(gt_vals)

        n = len(cards_vals)
        if n < 2:
            rows.append(dict(
                class_name=cls_name, n_pairs=n,
                spearman_rho=float("nan"), spearman_p=float("nan"),
                sign_agreement=float("nan"), sign_p=float("nan"),
            ))
            continue

        rho, rho_p = stats.spearmanr(cards_vals, gt_vals)
        arr        = np.array(cards_vals)
        gt         = np.array(gt_vals)
        n_agree    = int(np.sum(np.sign(arr) == np.sign(gt)))
        sign_agr   = n_agree / n
        sign_p     = _sign_pval(n_agree, n)

        rows.append(dict(
            class_name=cls_name, n_pairs=n,
            spearman_rho=float(rho), spearman_p=float(rho_p),
            sign_agreement=sign_agr, sign_p=sign_p,
        ))
        class_rhos.append(float(rho))
        class_signs.append(sign_agr)

    mean_rho  = float(np.mean(class_rhos))  if class_rhos  else float("nan")
    mean_sign = float(np.mean(class_signs)) if class_signs else float("nan")
    if len(pooled_s) >= 2:
        pr, pp = stats.spearmanr(pooled_s, pooled_g)
    else:
        pr, pp = float("nan"), float("nan")
    return rows, mean_rho, mean_sign, float(pr), float(pp), len(pooled_s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not FAITH_CSV.exists():
        raise FileNotFoundError(
            f"Faithfulness CSV not found: {FAITH_CSV}\n"
            "Run scripts/coco_stuff/build/compute_coco_stuff_faithfulness.py first."
        )

    # --- Encoder ---
    cfg = OmegaConf.create({
        "seed": SEED, "device": DEVICE,
        "encoder": {
            "name": "siglip",
            "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
            "model_name": "ViT-B-16-SigLIP",
            "pretrained": "webli",
            "device": DEVICE,
        },
        "cache_dir": "embedding_cache",
    })
    encoder     = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)
    print(f"Encoder: SigLIP ViT-B-16-SigLIP  device: {DEVICE}", flush=True)

    # --- Pool ---
    cfg.dataset    = {"name": "coco_stuff", "root": str(COCO_ROOT)}
    cfg.pool_source = "val"
    pairs = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="val")
    pool  = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"Pool: {len(pool.paths)} COCO val2017 images", flush=True)

    # --- ResNet18 ---
    import torch.nn as nn
    from torchvision.models import resnet18
    raw_model = resnet18(weights=None)
    raw_model.fc = nn.Linear(raw_model.fc.in_features, len(COCO_STUFF_20_CLASSES))
    raw_model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    raw_model  = raw_model.to(DEVICE).eval()
    preprocess = _coco_stuff_preprocess()
    print(f"ResNet18 loaded: {CKPT_PATH}", flush=True)

    # --- Concepts ---
    concepts    = [c for c in list_broden_concepts(BRODEN_ROOT) if c not in BRODEN_DROPPED_CONCEPTS]
    concept_idx = {c: i for i, c in enumerate(concepts)}
    print(f"Concepts: {len(concepts)}  orthogonalize=True", flush=True)

    raw_queries = {
        c: demean_query(build_concept_query(c, encoder), text_center)
        for c in concepts
    }
    queries = orthogonalize_queries(raw_queries)

    # --- Storage across alphas ---
    scores_by_alpha: dict[float, dict[tuple[str, str], float]] = {
        a: {} for a in ALPHA_VALUES
    }
    score_rows:     list[tuple] = []
    cutoff_rows:    list[tuple] = []
    per_image_rows: list[tuple] = []

    # --- Per-concept two-pass loop ---
    for c_i, concept_name in enumerate(concepts):
        t_c     = queries[concept_name]
        t_c_dev = t_c.to(DEVICE)

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)

        # Pass 1: localize once, cache (image, sim_map) for all present images
        cached: list[tuple[int, Image.Image, np.ndarray]] = []
        for idx in present_indices:
            image   = Image.open(pool.paths[idx]).convert("RGB")
            sim_map = localize_concept(encoder, image, t_c, (image.height, image.width))
            cached.append((idx, image, sim_map))
        sim_maps = [sm for _, _, sm in cached]

        print(
            f"[{c_i + 1:>3d}/{len(concepts)}] {concept_name:<25s} "
            f"retrieved={len(cached)}",
            flush=True,
        )

        # Pass 2: for each alpha, threshold cached maps, mask/fill/score
        for alpha in ALPHA_VALUES:
            cutoff = concept_zscore_cutoff(sim_maps, alpha)
            delta_logits: dict[str, list[float]] = {cls: [] for cls in COCO_STUFF_20_CLASSES}
            areas: list[float] = []
            n_skipped = 0

            for idx, image, sim_map in cached:
                mask = threshold_mask(sim_map, method="fixed", cutoff=cutoff)
                if not mask.any() or mask.all():
                    n_skipped += 1
                    continue
                areas.append(float(mask.mean()))

                rng        = np.random.default_rng(SEED + concept_idx[concept_name] * 10_000 + int(idx))
                candidates = [mask_region(image, mask, strategy=s, rng=rng) for s in FILL_STRATEGIES]

                with torch.no_grad():
                    embeds = encoder.encode_images([image] + candidates).to(DEVICE)
                embed_orig = embeds[0]

                best_angle, best_i = None, 0
                for i in range(len(FILL_STRATEGIES)):
                    diff      = embed_orig - embeds[1 + i]
                    diff_unit = diff / (diff.norm() + 1e-8)
                    cos_sim   = float(torch.clamp(diff_unit @ t_c_dev, -1.0, 1.0))
                    angle_deg = float(np.degrees(np.arccos(cos_sim)))
                    if best_angle is None or angle_deg < best_angle:
                        best_angle, best_i = angle_deg, i

                masked_image  = candidates[best_i]
                pixels_orig   = preprocess(image).unsqueeze(0)
                pixels_masked = preprocess(masked_image).unsqueeze(0)
                batch = torch.cat([pixels_orig, pixels_masked], dim=0).to(DEVICE)
                with torch.no_grad():
                    logits = raw_model(batch)   # (2, 20)

                for cls_idx, cls_name in enumerate(COCO_STUFF_20_CLASSES):
                    dl = (logits[0, cls_idx] - logits[1, cls_idx]).item()
                    delta_logits[cls_name].append(dl)
                    per_image_rows.append((alpha, concept_name, str(pool.paths[idx]), cls_name, dl))

            mean_area = 100.0 * float(np.mean(areas)) if areas else 0.0
            cutoff_rows.append((alpha, concept_name, cutoff, mean_area))

            for cls_name in COCO_STUFF_20_CLASSES:
                score = float(np.mean(delta_logits[cls_name])) if delta_logits[cls_name] else 0.0
                scores_by_alpha[alpha][(concept_name, cls_name)] = score
                score_rows.append((alpha, concept_name, cls_name, score))

            print(
                f"  alpha={alpha:.2f}  cutoff={cutoff:+.4f}  "
                f"mean_area={mean_area:.1f}%  "
                f"n={len(cached) - n_skipped}  skipped={n_skipped}",
                flush=True,
            )

    # --- Evaluate all alphas × both strategies ---
    summary_rows: list[dict] = []

    for strategy in STRATEGIES:
        faith, image_counts = load_faithfulness(FAITH_CSV, strategy=strategy)
        n_valid = sum(1 for n in image_counts.values() if n >= MIN_FAITH_IMGS)
        print(f"\n{'=' * 70}", flush=True)
        print(
            f"Strategy: {strategy}  |  faithfulness pairs: {len(faith)}  "
            f"valid concepts (≥{MIN_FAITH_IMGS} imgs): {n_valid}",
            flush=True,
        )

        best_alpha_rho  = None
        best_mean_rho   = -float("inf")
        best_alpha_sign = None
        best_mean_sign  = -float("inf")
        best_alpha_pr   = None
        best_pooled_rho = -float("inf")

        for alpha in ALPHA_VALUES:
            print(f"\n=== alpha={alpha:.2f}  strategy={strategy} ===", flush=True)
            rows, mean_rho, mean_sign, pr, pp, pn = evaluate_alpha(
                scores_by_alpha[alpha], faith, image_counts
            )
            for r in rows:
                r["alpha"]      = alpha
                r["strategy"]   = strategy
                r["pooled_rho"] = pr
                r["pooled_p"]   = pp
                r["pooled_n"]   = pn
                summary_rows.append(r)
                if not np.isnan(r["spearman_rho"]):
                    print(
                        f"  [{r['class_name']:<15s}] n={r['n_pairs']:>4d}  "
                        f"rho={r['spearman_rho']:+.3f} (p={r['spearman_p']:.3g})  "
                        f"sign={r['sign_agreement']:.1%} (p={r['sign_p']:.3g})",
                        flush=True,
                    )

            co_scores = {k: v for k, v in scores_by_alpha[alpha].items()
                         if k[0] in CO_OCCUR_BRODEN_CONCEPTS}
            _, co_rho, co_sign, _, _, _ = evaluate_alpha(co_scores, faith, image_counts)
            print(
                f"  --- mean rho={mean_rho:+.3f}  pooled rho={pr:+.3f}  "
                f"mean sign={mean_sign:.1%}  |  "
                f"mean rho (co-occur)={co_rho:+.3f}  "
                f"mean sign (co-occur)={co_sign:.1%}",
                flush=True,
            )

            if mean_rho > best_mean_rho:
                best_mean_rho, best_alpha_rho = mean_rho, alpha
            if mean_sign > best_mean_sign:
                best_mean_sign, best_alpha_sign = mean_sign, alpha
            if not np.isnan(pr) and pr > best_pooled_rho:
                best_pooled_rho, best_alpha_pr = pr, alpha

        print(
            f"\n[{strategy}] Best alpha by mean rho  : alpha={best_alpha_rho:.2f}  rho={best_mean_rho:+.3f}",
            flush=True,
        )
        print(
            f"[{strategy}] Best alpha by pooled rho : alpha={best_alpha_pr:.2f}  rho={best_pooled_rho:+.3f}",
            flush=True,
        )
        print(
            f"[{strategy}] Best alpha by mean sign  : alpha={best_alpha_sign:.2f}  sign={best_mean_sign:.1%}",
            flush=True,
        )

    # --- Save CSVs ---
    scores_path = RESULTS_DIR / "alpha_ablation_scores.csv"
    with open(scores_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "concept_name", "class_name", "raw_score"])
        writer.writerows(score_rows)
    print(f"\nSaved {scores_path}", flush=True)

    summary_path = RESULTS_DIR / "alpha_ablation_summary.csv"
    fields = ["strategy", "alpha", "class_name", "n_pairs", "spearman_rho", "spearman_p",
              "sign_agreement", "sign_p", "pooled_rho", "pooled_p", "pooled_n"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved {summary_path}", flush=True)

    cutoffs_path = RESULTS_DIR / "alpha_ablation_cutoffs.csv"
    with open(cutoffs_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "concept_name", "cutoff", "mean_area_pct"])
        writer.writerows(cutoff_rows)
    print(f"Saved {cutoffs_path}", flush=True)

    per_image_path = RESULTS_DIR / "alpha_ablation_per_image_scores.csv"
    with open(per_image_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "concept_name", "image_path", "class_name", "delta_logit"])
        writer.writerows(per_image_rows)
    print(f"Saved {per_image_path}", flush=True)


if __name__ == "__main__":
    main()
