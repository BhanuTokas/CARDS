"""Masking hybrid (k=1.0 z-score threshold, the best-known config from
the threshold ablation) run on a completely DIFFERENT image pool:
standard CelebA's own OFFICIAL val partition (list_eval_partition.txt,
partition==1), not CelebAMask-HQ's custom 85/15 split -- prompted
directly ("I want to see the concept attribution calculated by Hybrid
method using official CelebA val set of images").

Two overlaps are excluded from the 19,867 official-val images before
building this pool:
  - 2,516 images that are ALSO in CelebAMask-HQ's own TRAIN split --
    MUST exclude: the celeba_attractive_young classifier trained on
    those exact photos (at HQ resolution), so scoring on them would be
    train-set leakage, not a held-out evaluation.
  - 477 images that are ALSO in CelebAMask-HQ's own VAL split -- exclude
    per the user's earlier explicit request ("omitting the overlapping
    images"), so this is a genuinely fresh set, disjoint from every
    image used anywhere else in this investigation.
Leaves 16,874 clean images, at standard CelebA's own lower-resolution
face-aligned crops (img_align_celeba/, not HQ's re-derived 1024x1024s)
-- a real, deliberate resolution difference from every other run in this
track, not an oversight.

NO ground-truth comparison here (rho/sign against real faithfulness):
CelebAMask-HQ's real per-pixel segmentation masks only exist for its own
30,000 curated images -- none of these 16,874 official-val images have
one, so compute_faithfulness can't run on them. This script reports the
hybrid's own raw attribution scores only, plus a cross-pool consistency
check against the ALREADY-computed k=1.0 HQ-val raw scores (`results/
cards_celeba_masking_hybrid_threshold_zscore_orthogonalize_ablation_
raw_scores.csv`) -- does the hybrid rank concepts similarly on two
disjoint image pools, the closest thing to a validity check available
without real masks on this pool.

Same settings as the k=1.0 zscore ablation otherwise: demean_query=True,
orthogonalize=True, K=50, SigLIP, the same 7-strategy fill family.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT, TASK_POSITIVE_LOGIT_INDEX

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba import load_celebamask_hq_image_paths, split_celebamask_hq
from cards.data.celeba_attributes import (
    GROUNDABLE_CONCEPTS,
    TARGET_CLASSES,
    load_attribute_labels,
    load_attribute_names,
)
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import mask_region

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
CELEBA_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebA\celeba")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
SEED = 42
ALPHA = 1.0  # the best-known threshold config (see notes v94 addendum)
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}


def build_clean_official_val_paths() -> list[Path]:
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]
    hq_train_indices, hq_val_indices = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices)

    hq_to_orig = {}
    with open(CELEBA_HQ_ROOT / "CelebA-HQ-to-CelebA-mapping.txt") as f:
        next(f)
        for line in f:
            parts = line.split()
            hq_to_orig[int(parts[0])] = parts[2]
    hq_train_orig = {hq_to_orig[i] for i in hq_train_indices}
    hq_val_orig = {hq_to_orig[i] for i in hq_val_indices}

    official_val = []
    with open(CELEBA_ROOT / "list_eval_partition.txt") as f:
        for line in f:
            fname, part = line.split()
            if part == "1":
                official_val.append(fname)

    clean = [f for f in official_val if f not in hq_train_orig and f not in hq_val_orig]
    print(f"official CelebA val: {len(official_val)}  "
          f"minus HQ-train overlap ({len(set(official_val) & hq_train_orig)}, leakage)  "
          f"minus HQ-val overlap ({len(set(official_val) & hq_val_orig)}, already tested)  "
          f"= {len(clean)} clean images", flush=True)
    return [CELEBA_ROOT / "img_align_celeba" / f for f in clean]


def load_baseline_k1_scores() -> dict[str, dict[str, float]]:
    """The already-computed k=1.0 raw scores on CelebAMask-HQ's own val
    pool -- for the cross-pool consistency check, not ground truth."""
    scores = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_threshold_zscore_orthogonalize_ablation_raw_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            if float(row["k"]) == ALPHA:
                scores[row["target_task"]][row["concept_name"]] = float(row["hybrid_raw_score"])
    return scores


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    paths = build_clean_official_val_paths()

    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    cfg.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}
    cfg.pool_source = "val"
    pairs = [(p, 0) for p in paths]  # label unused by retrieval/scoring, placeholder only
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images (fresh encode if first run -- this is a NEW cache key)", flush=True)

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()

    raw_queries = {
        c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], encoder), text_center)
        for c in GROUNDABLE_CONCEPTS
    }
    queries = orthogonalize_queries(raw_queries)

    all_rows = []  # (concept_name, target_task, hybrid_raw_score)
    for c_i, concept_name in enumerate(GROUNDABLE_CONCEPTS):
        t_c = queries[concept_name]
        t_c_dev = t_c.to(DEVICE)
        concept_idx = CONCEPT_TO_IDX[concept_name]

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)

        cached = []
        for idx in present_indices:
            image = Image.open(pool.paths[idx]).convert("RGB")
            sim_map = localize_concept(encoder, image, t_c, (image.height, image.width))
            cached.append((idx, image, sim_map))
        cutoff = concept_zscore_cutoff([sm for _, _, sm in cached], ALPHA)

        delta_logits = {t: [] for t in TARGET_CLASSES}
        n_skipped = 0
        for idx, image, sim_map in cached:
            mask = threshold_mask(sim_map, method="fixed", cutoff=cutoff)
            if not mask.any() or mask.all():
                n_skipped += 1
                continue

            rng = np.random.default_rng(SEED + concept_idx * 10_000 + int(idx))
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
            score = float(np.mean(delta_logits[task_name])) if delta_logits[task_name] else 0.0
            all_rows.append((concept_name, task_name, score))

        print(f"[{c_i + 1:>2d}/{len(GROUNDABLE_CONCEPTS)}] {concept_name:<20s} cutoff={cutoff:+.4f} "
              f"n={len(present_indices) - n_skipped:>3d} skipped={n_skipped}", flush=True)

    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_official_val_k1_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "target_task", "hybrid_raw_score"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to results/cards_celeba_masking_hybrid_official_val_k1_scores.csv", flush=True)

    print("\n=== attribution scores (this run, official CelebA val) ===")
    for task_name in TARGET_CLASSES:
        print(f"\n  -- {task_name} --")
        task_rows = sorted((r for r in all_rows if r[1] == task_name), key=lambda r: -r[2])
        for concept_name, _t, score in task_rows:
            print(f"    {concept_name:<20s} {score:+.4f}")

    print("\n=== cross-pool consistency check vs. the already-computed k=1.0 HQ-val scores (NOT ground truth) ===")
    baseline = load_baseline_k1_scores()
    new_scores = {t: {} for t in TARGET_CLASSES}
    for concept_name, task_name, score in all_rows:
        new_scores[task_name][concept_name] = score
    for task_name in TARGET_CLASSES:
        concepts_common = [c for c in GROUNDABLE_CONCEPTS if c in baseline[task_name]]
        a = [baseline[task_name][c] for c in concepts_common]
        b = [new_scores[task_name][c] for c in concepts_common]
        rho, p = stats.spearmanr(a, b)
        print(f"  [{task_name}] Spearman rho between HQ-val k=1.0 scores and official-val k=1.0 scores: "
              f"rho={rho:+.4f} p={p:.4g} (n={len(concepts_common)} concepts)")


if __name__ == "__main__":
    main()
