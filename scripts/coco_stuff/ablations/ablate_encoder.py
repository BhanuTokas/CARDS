"""Encoder ablation for COCO-Stuff CARDS.

Varies the vision-language encoder used for retrieval and localization:
  siglip             -- ViT-B-16-SigLIP/webli (loaded from alpha_ablation_scores.csv)
  clip               -- ViT-B-32/openai
  open_clip_h        -- ViT-H-14/laion2b_s32b_b79k
  perception_encoder -- PE-Core-B16-224

Fixed across all conditions: K=50, alpha=1.0, demean=True, orth=True.

Outputs (results/coco_stuff/):
  encoder_ablation_scores.csv   -- (encoder, concept_name, class_name, raw_score)
  encoder_ablation_summary.csv  -- (encoder, strategy, class_name, n_pairs,
                                    spearman_rho, spearman_p, sign_agreement,
                                    sign_p, pooled_rho, pooled_p, pooled_n)

Usage:
    cd /scratch/rnair21/CARDS
    /scratch/rnair21/envs/cards_env/bin/python \\
        scripts/coco_stuff/ablations/ablate_encoder.py
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
from cards.data.coco_stuff import COCO_STUFF_20_CLASSES
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
ALPHA_CSV   = RESULTS_DIR / "alpha_ablation_scores.csv"

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
ALPHA          = 1.5
SEED           = 42
MIN_SAMPLES    = 3
MIN_FAITH_IMGS = 100
STRATEGIES     = ("black_mask", "black_bbox")

FILL_STRATEGIES = [
    "blur", "zero_fill", "mean_fill", "hue_shift",
    "white_fill", "zero_fill_noise", "noise_then_blur",
]

ENCODERS_TO_RUN: dict[str, dict] = {
    "clip": {
        "name": "clip",
        "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
        "model_name": "ViT-B-32",
        "pretrained": "openai",
        "device": DEVICE,
    },
    "open_clip_h": {
        "name": "open_clip_h",
        "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
        "model_name": "ViT-H-14",
        "pretrained": "laion2b_s32b_b79k",
        "device": DEVICE,
    },
    "perception_encoder": {
        "name": "perception_encoder",
        "_target_": "cards.encoders.perception_encoder.PerceptionEncoder",
        "model_name": "PE-Core-B16-224",
        "perception_models_path": "../perception_models",
        "device": DEVICE,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_siglip_scores(path: Path, alpha: float) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if abs(float(row["alpha"]) - alpha) < 1e-6:
                scores[(row["concept_name"], row["class_name"])] = float(row["raw_score"])
    return scores


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


def run_encoder(
    encoder_name: str,
    encoder_cfg:  dict,
    concepts:     list[str],
    concept_idx:  dict[str, int],
) -> dict[tuple[str, str], float]:
    import torch.nn as nn
    from torchvision.models import resnet18

    print(f"\n{'=' * 60}", flush=True)
    print(f"Encoder: {encoder_name}", flush=True)

    cfg = OmegaConf.create({
        "seed": SEED, "device": DEVICE,
        "encoder": encoder_cfg,
        "cache_dir": "embedding_cache",
    })
    encoder    = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    cfg.dataset    = {"name": "coco_stuff", "root": str(COCO_ROOT)}
    cfg.pool_source = "val"
    pairs = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="val")
    pool  = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"Pool: {len(pool.paths)} images", flush=True)

    raw_model = resnet18(weights=None)
    raw_model.fc = nn.Linear(raw_model.fc.in_features, len(COCO_STUFF_20_CLASSES))
    raw_model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    raw_model = raw_model.to(DEVICE).eval()
    preprocess = _coco_stuff_preprocess()

    raw_queries = {
        c: demean_query(build_concept_query(c, encoder), text_center)
        for c in concepts
    }
    queries = orthogonalize_queries(raw_queries)
    scores: dict[tuple[str, str], float] = {}

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

            rng = np.random.default_rng(SEED + concept_idx[concept_name] * 10_000 + int(idx))
            candidates = [mask_region(image, mask, strategy=s, rng=rng) for s in FILL_STRATEGIES]

            with torch.no_grad():
                embeds = encoder.encode_images([image] + candidates).to(DEVICE)
            embed_orig = embeds[0]
            best_angle, best_i = None, 0
            for i in range(len(FILL_STRATEGIES)):
                diff   = embed_orig - embeds[1 + i]
                diff_u = diff / (diff.norm() + 1e-8)
                cos    = float(torch.clamp(diff_u @ t_c_dev, -1.0, 1.0))
                angle  = float(np.degrees(np.arccos(cos)))
                if best_angle is None or angle < best_angle:
                    best_angle, best_i = angle, i

            px_o = preprocess(image).unsqueeze(0)
            px_m = preprocess(candidates[best_i]).unsqueeze(0)
            with torch.no_grad():
                logits = raw_model(torch.cat([px_o, px_m], dim=0).to(DEVICE))

            for cls_i, cls_name in enumerate(COCO_STUFF_20_CLASSES):
                delta_logits[cls_name].append((logits[0, cls_i] - logits[1, cls_i]).item())

        for cls_name in COCO_STUFF_20_CLASSES:
            dl = delta_logits[cls_name]
            scores[(concept_name, cls_name)] = float(np.mean(dl)) if dl else 0.0

        print(
            f"  [{c_i + 1:>3d}/{len(concepts)}] {concept_name:<25s} "
            f"retrieved={len(cached)}  skipped={n_skipped}",
            flush=True,
        )

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    concepts    = [c for c in list_broden_concepts(BRODEN_ROOT) if c not in BRODEN_DROPPED_CONCEPTS]
    concept_idx = {c: i for i, c in enumerate(concepts)}
    print(f"Concepts: {len(concepts)}", flush=True)

    siglip_scores = load_siglip_scores(ALPHA_CSV, alpha=ALPHA)
    print(f"SigLIP scores loaded from {ALPHA_CSV}  (alpha={ALPHA})", flush=True)

    all_scores: dict[str, dict[tuple[str, str], float]] = {"siglip": siglip_scores}
    for enc_name, enc_cfg in ENCODERS_TO_RUN.items():
        all_scores[enc_name] = run_encoder(enc_name, enc_cfg, concepts, concept_idx)

    # Save raw scores
    scores_csv = RESULTS_DIR / "encoder_ablation_scores.csv"
    with open(scores_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["encoder", "concept_name", "class_name", "raw_score"])
        for enc_name, scores in all_scores.items():
            for (concept, cls), score in scores.items():
                writer.writerow([enc_name, concept, cls, score])
    print(f"\nSaved -> {scores_csv}", flush=True)

    # Evaluate
    summary_rows: list[dict] = []
    for strategy in STRATEGIES:
        faith, image_counts = load_faithfulness(FAITH_CSV, strategy)
        print(f"\n{'=' * 60}  strategy={strategy}", flush=True)
        for enc_name, scores in all_scores.items():
            rows, mean_rho, mean_sign, pr, pp, pn = evaluate(scores, faith, image_counts)
            print(
                f"  {enc_name:<22s}  mean_rho={mean_rho:+.4f}  "
                f"mean_sign={mean_sign:.4f}  "
                f"pooled_rho={pr:+.4f}  pooled_p={pp:.4g}  n_pairs={pn}",
                flush=True,
            )
            for r in rows:
                summary_rows.append({
                    "encoder": enc_name, "strategy": strategy, **r,
                    "pooled_rho": pr, "pooled_p": pp, "pooled_n": pn,
                })

    summary_csv = RESULTS_DIR / "encoder_ablation_summary.csv"
    fields = ["encoder", "strategy", "class_name", "n_pairs",
              "spearman_rho", "spearman_p", "sign_agreement", "sign_p",
              "pooled_rho", "pooled_p", "pooled_n"]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved -> {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
