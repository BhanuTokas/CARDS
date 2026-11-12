"""CARDS model-variant experiment for COCO-Stuff.

Tests CARDS faithfulness across three black-box classifier architectures:
  coco_stuff            -- ResNet18 (baseline)
  coco_stuff_vit_b_16   -- ViT-B/16
  coco_stuff_convnext_tiny -- ConvNeXt-Tiny

Architecture is selected via the BACKBONE_NAME environment variable (default:
"coco_stuff"). All three use identical CARDS settings (SigLIP encoder, K=50,
ALPHA=1.5, demean=True, orth=True, best-of-7 fill) so any rho difference is
attributable solely to the black-box model.

Output: results/coco_stuff/model_variants/{BACKBONE_NAME}_cards_scores.csv
  concept_name, class_name, raw_score

Rho vs. coco_stuff_faithfulness.csv is printed at end (same evaluation
protocol as all other COCO-Stuff experiments).

Usage:
    cd /scratch/rnair21/CARDS
    BACKBONE_NAME=coco_stuff_vit_b_16 \\
        /scratch/rnair21/envs/cards_env/bin/python \\
        scripts/coco_stuff/run/run_cards_coco_stuff_model_variants.py
"""

from __future__ import annotations

import csv
import os
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
from cards.data.coco_stuff import COCO_STUFF_20_CLASSES
from cards.data.datasets import list_broden_concepts, load_coco_stuff
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import mask_region

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COCO_ROOT   = Path("/scratch/rnair21/LLMConceptEditing_old/coco")
BRODEN_ROOT = Path("/scratch/rnair21/LLMConceptEditing_old/broden_concepts")
RESULTS_DIR = Path("results/coco_stuff")
OUT_DIR     = Path("results/coco_stuff/model_variants")
COUNTS_CSV  = RESULTS_DIR / "coco_stuff_faithfulness_detection_counts.csv"
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
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
K               = 50
ALPHA           = 1.5
SEED            = 42
MIN_FAITH_IMGS  = 100
MIN_SAMPLES     = 3
STRATEGIES      = ("black_mask", "black_bbox")

FILL_STRATEGIES = [
    "blur", "zero_fill", "mean_fill", "hue_shift",
    "white_fill", "zero_fill_noise", "noise_then_blur",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_balanced_concepts(counts_csv: Path, min_images: int) -> set[str]:
    balanced: set[str] = set()
    with open(counts_csv, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["n_images"]) >= min_images:
                balanced.add(row["concept_name"])
    return balanced


def load_faithfulness(
    path: Path, strategy: str
) -> tuple[dict[tuple[str, str], list[float]], dict[str, int]]:
    col = f"delta_p_{strategy}"
    concept_imgs: dict[str, set[str]] = defaultdict(set)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            concept_imgs[row["concept_name"]].add(row["image_path"])
    valid_imgs: dict[str, set[str]] = {
        c: set(sorted(imgs)[:MIN_FAITH_IMGS])
        for c, imgs in concept_imgs.items()
        if len(imgs) >= MIN_FAITH_IMGS
    }
    faith: dict[tuple[str, str], list[float]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            c, cl, img = row["concept_name"], row["class_name"], row["image_path"]
            if c in valid_imgs and img in valid_imgs[c]:
                faith[(c, cl)].append(float(row[col]))
    image_counts = {c: MIN_FAITH_IMGS for c in valid_imgs}
    return dict(faith), image_counts


def _sign_pval(n_agree: int, n: int) -> float:
    return stats.binomtest(n_agree, n, p=0.5, alternative="two-sided").pvalue


def evaluate(
    scores:       dict[tuple[str, str], float],
    faith:        dict[tuple[str, str], list[float]],
    image_counts: dict[str, int],
) -> tuple[list[dict], float, float, float, float, int]:
    valid = {c for c, n in image_counts.items() if n >= MIN_FAITH_IMGS}
    rows: list[dict] = []
    class_rhos, class_signs = [], []
    pooled_s, pooled_g = [], []

    for cls_name in COCO_STUFF_20_CLASSES:
        s_vals, g_vals = [], []
        for key, dp_list in faith.items():
            if key[1] != cls_name or key[0] not in valid or len(dp_list) < MIN_SAMPLES:
                continue
            if key not in scores:
                continue
            s_vals.append(scores[key])
            g_vals.append(float(np.mean(dp_list)))
        pooled_s.extend(s_vals)
        pooled_g.extend(g_vals)

        n = len(s_vals)
        if n < 2:
            rows.append(dict(class_name=cls_name, n_pairs=n,
                             spearman_rho=float("nan"), spearman_p=float("nan"),
                             sign_agreement=float("nan"), sign_p=float("nan")))
            continue

        rho, rho_p = stats.spearmanr(s_vals, g_vals)
        n_agree  = int(np.sum(np.sign(np.array(s_vals)) == np.sign(np.array(g_vals))))
        sign_agr = n_agree / n
        rows.append(dict(class_name=cls_name, n_pairs=n,
                         spearman_rho=float(rho), spearman_p=float(rho_p),
                         sign_agreement=sign_agr, sign_p=_sign_pval(n_agree, n)))
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
    backbone_name = os.environ.get("BACKBONE_NAME", "coco_stuff")
    if backbone_name not in BACKBONES:
        raise ValueError(
            f"Unknown BACKBONE_NAME={backbone_name!r}. "
            f"Available: {list(BACKBONES)}"
        )
    spec = BACKBONES[backbone_name]
    print(f"Backbone: {backbone_name}  embed_dim={spec.embed_dim}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Encoder + pool ---
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

    cfg.dataset    = {"name": "coco_stuff", "root": str(COCO_ROOT)}
    cfg.pool_source = "val"
    pairs = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="val")
    pool  = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"Pool: {len(pool.paths)} COCO val2017 images", flush=True)

    # --- Black-box model ---
    model      = spec.load_native().to(DEVICE).eval()
    preprocess = spec.preprocess
    print(f"Model loaded.  preprocess={preprocess}", flush=True)

    # --- Concepts ---
    # Orthogonalize over all 143 concepts (matching ablate_alpha.py); score only
    # the balanced subset for rho evaluation.
    balanced     = _load_balanced_concepts(COUNTS_CSV, MIN_FAITH_IMGS)
    all_concepts = [c for c in list_broden_concepts(BRODEN_ROOT)
                    if c not in BRODEN_DROPPED_CONCEPTS]
    concepts     = [c for c in all_concepts if c in balanced]
    concept_idx  = {c: i for i, c in enumerate(concepts)}
    print(f"All concepts: {len(all_concepts)}  scored (balanced): {len(concepts)}", flush=True)

    raw_queries = {
        c: demean_query(build_concept_query(c, encoder), text_center)
        for c in all_concepts
    }
    queries = orthogonalize_queries(raw_queries)

    # --- Per-concept scoring ---
    all_scores: dict[tuple[str, str], float] = {}
    score_rows: list[tuple] = []
    per_image_rows: list[tuple] = []   # (concept, image idx, class, delta_logit) for Cohen's d
    suffix = os.environ.get("OUT_SUFFIX", "")   # e.g. _dz keeps the existing score/summary files untouched

    for c_i, concept_name in enumerate(concepts):
        t_c     = queries[concept_name]
        t_c_dev = t_c.to(DEVICE)

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
        cached: list[tuple[int, Image.Image, np.ndarray]] = []
        for idx in present_indices:
            image   = Image.open(pool.paths[idx]).convert("RGB")
            sim_map = localize_concept(encoder, image, t_c, (image.height, image.width))
            cached.append((idx, image, sim_map))

        cutoff = concept_zscore_cutoff([sm for _, _, sm in cached], ALPHA)
        delta_logits: dict[str, list[float]] = {cls: [] for cls in COCO_STUFF_20_CLASSES}
        n_skipped = 0

        for idx, image, sim_map in cached:
            mask = threshold_mask(sim_map, method="fixed", cutoff=cutoff)
            if not mask.any() or mask.all():
                n_skipped += 1
                continue

            rng = np.random.default_rng(
                SEED + concept_idx[concept_name] * 10_000 + int(idx)
            )
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

            pix_orig   = preprocess(image).unsqueeze(0)
            pix_masked = preprocess(candidates[best_i]).unsqueeze(0)
            with torch.no_grad():
                logits = model(torch.cat([pix_orig, pix_masked], dim=0).to(DEVICE))

            for cls_i, cls_name in enumerate(COCO_STUFF_20_CLASSES):
                d_ = (logits[0, cls_i] - logits[1, cls_i]).item()
                delta_logits[cls_name].append(d_)
                per_image_rows.append((concept_name, int(idx), cls_name, d_))

        for cls_name in COCO_STUFF_20_CLASSES:
            dl  = delta_logits[cls_name]
            raw = float(np.mean(dl)) if dl else 0.0
            all_scores[(concept_name, cls_name)] = raw
            score_rows.append((concept_name, cls_name, raw))

        if (c_i + 1) % 10 == 0 or c_i == 0:
            print(
                f"  [{c_i + 1:>3d}/{len(concepts)}] {concept_name:<25s} "
                f"retrieved={len(cached)}  skipped={n_skipped}",
                flush=True,
            )

    # --- Save scores ---
    out_path = OUT_DIR / f"{backbone_name}_cards_scores{suffix}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "class_name", "raw_score"])
        writer.writerows(score_rows)
    print(f"\nSaved {len(score_rows)} rows -> {out_path}", flush=True)
    pi_path = OUT_DIR / f"{backbone_name}_cards_per_image{suffix}.csv"
    with open(pi_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "image_idx", "class_name", "delta_logit"])
        writer.writerows(per_image_rows)
    print(f"Saved {len(per_image_rows)} per-image rows -> {pi_path}", flush=True)

    # --- Evaluate rho ---
    print("\n=== Faithfulness evaluation ===", flush=True)
    summary_rows: list[dict] = []
    for strategy in STRATEGIES:
        faith, image_counts = load_faithfulness(FAITH_CSV, strategy)
        rows, mean_rho, mean_sign, pr, pp, pn = evaluate(all_scores, faith, image_counts)
        print(
            f"  strategy={strategy:<12s}  mean_rho={mean_rho:+.4f}  "
            f"mean_sign={mean_sign:.4f}  "
            f"pooled_rho={pr:+.4f}  pooled_p={pp:.4g}  n_pairs={pn}",
            flush=True,
        )
        for r in rows:
            if not np.isnan(r["spearman_rho"]):
                print(
                    f"    [{r['class_name']:<15s}] n={r['n_pairs']:>4d}  "
                    f"rho={r['spearman_rho']:+.3f}  sign={r['sign_agreement']:.1%}",
                    flush=True,
                )
            summary_rows.append({
                "backbone": backbone_name, "strategy": strategy, **r,
                "pooled_rho": pr, "pooled_p": pp, "pooled_n": pn,
            })

    summary_path = OUT_DIR / f"{backbone_name}_summary{suffix}.csv"
    fields = ["backbone", "strategy", "class_name", "n_pairs",
              "spearman_rho", "spearman_p", "sign_agreement", "sign_p",
              "pooled_rho", "pooled_p", "pooled_n"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
