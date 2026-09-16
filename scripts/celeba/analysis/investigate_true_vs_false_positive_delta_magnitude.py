"""Tests a specific hypothesis raised directly after v115f's decoupling
puzzle: "Maybe the correct samples are dominating the final values?" --
i.e. even though precision@50 collapses at the 1%-pool (n=169) size,
maybe the FEW genuinely true-positive images in a concept's present set
produce a much LARGER |b(orig)-b(masked)| masking effect than the many
false-positive images (masking an irrelevant region on a non-target
image should do little), so the mean raw_score end up dominated by a
small number of real matches rather than diluted by the many wrong
ones -- which would explain v115c-f's rho robustness directly, without
needing precision itself to predict per-image rank accuracy.

Reimplements masking_score's own per-image loop (cards.attribution.
masking_mode.masking_score, ported here rather than called directly)
because that function only returns the AGGREGATE raw_score / lists
without attaching each per-image delta_score to which pool index it
came from after degenerate-mask skipping -- this script needs that
join to label each surviving delta_score true-positive or
false-positive against the REAL attribute label.

Single 1%-pool draw (seed=0, matching v115c/d/e/f's own seed=0 for
direct comparability), all 26 concepts, Attractive task only.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT, TASK_POSITIVE_LOGIT_INDEX
from run_cards_celeba_masking_hybrid_official_val_zscore import build_clean_official_val_paths

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.attribution.masking_mode import DEFAULT_FILL_STRATEGIES
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba_attributes import (
    GROUNDABLE_CONCEPTS,
    load_attribute_labels,
    load_attribute_names,
)
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.validation.broden_faithfulness import mask_region

CELEBA_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebA\celeba")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
ALPHA = 1.0
FRACTION = 0.01
SEED = 0
TASK = "Attractive"
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    attr_names = load_attribute_names(CELEBA_ROOT / "list_attr_celeba.txt")
    attr_labels = load_attribute_labels(CELEBA_ROOT / "list_attr_celeba.txt")
    concept_col = {c: attr_names.index(c) for c in GROUNDABLE_CONCEPTS}

    cfg = OmegaConf.create({
        "seed": SEED, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()
    task_idx = TASK_POSITIVE_LOGIT_INDEX[TASK]

    @torch.no_grad()
    def black_box(batch: torch.Tensor) -> torch.Tensor:
        return native_model(batch.to(DEVICE))[:, task_idx].detach().cpu()

    official_paths = build_clean_official_val_paths()
    pairs = [(p, 0) for p in official_paths]

    pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": cfg.encoder, "cache_dir": "embedding_cache"})
    pool_cfg.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}
    pool_cfg.pool_source = "val"
    full_pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
    n_full = len(full_pool.paths)
    n_sub = round(n_full * FRACTION)

    rng_sub = np.random.default_rng(SEED)
    sub_idx = rng_sub.choice(n_full, size=n_sub, replace=False)
    sub_paths = [full_pool.paths[i] for i in sub_idx]
    sub_embeds = full_pool.embeddings[sub_idx]
    sub_labels = np.zeros((n_sub, len(GROUNDABLE_CONCEPTS)), dtype=bool)
    for i, p in enumerate(sub_paths):
        row = attr_labels.get(p.name)
        if row is not None:
            sub_labels[i] = [row[concept_col[c]] for c in GROUNDABLE_CONCEPTS]
    print(f"1% subsample: {n_sub} images", flush=True)

    raw_queries = {c: build_concept_query(CONCEPT_QUERY_TEXT[c], encoder) for c in GROUNDABLE_CONCEPTS}
    demeaned = {c: demean_query(q, text_center) for c, q in raw_queries.items()}
    queries = orthogonalize_queries(demeaned)

    all_rows = []  # (concept, is_true_positive, delta_score, abs_delta_score)

    for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
        t_c = queries[concept_name]
        sims = (sub_embeds @ t_c.to(sub_embeds.dtype)).numpy()
        present_indices = np.argsort(-sims)[:K].tolist()

        images = {idx: Image.open(sub_paths[idx]).convert("RGB") for idx in present_indices}
        sim_maps = {idx: localize_concept(encoder, images[idx], t_c, (images[idx].height, images[idx].width))
                    for idx in present_indices}
        cutoff = concept_zscore_cutoff(list(sim_maps.values()), ALPHA)

        n_tp = int(sub_labels[present_indices, concept_idx].sum())
        n_kept = 0
        for idx in present_indices:
            image = images[idx]
            mask = threshold_mask(sim_maps[idx], method="fixed", cutoff=cutoff)
            if not mask.any() or mask.all():
                continue
            n_kept += 1

            img_rng = np.random.default_rng(SEED + concept_idx * 10_000 + int(idx))
            candidates = [mask_region(image, mask, strategy=s, rng=img_rng) for s in DEFAULT_FILL_STRATEGIES]
            with torch.no_grad():
                embeds = encoder.encode_images([image] + candidates)
            embed_orig = embeds[0]
            t_c_dev = t_c.to(embed_orig.device)
            best_angle, best_i = None, 0
            for i in range(len(DEFAULT_FILL_STRATEGIES)):
                diff = embed_orig - embeds[1 + i]
                diff_unit = diff / diff.norm()
                cos_sim = float(torch.clamp(diff_unit @ t_c_dev, -1.0, 1.0))
                angle_deg = float(np.degrees(np.arccos(cos_sim)))
                if best_angle is None or angle_deg < best_angle:
                    best_angle, best_i = angle_deg, i
            masked_image = candidates[best_i]

            pixels_orig = spec.preprocess(image).unsqueeze(0)
            pixels_masked = spec.preprocess(masked_image).unsqueeze(0)
            batch = torch.cat([pixels_orig, pixels_masked], dim=0)
            outputs = black_box(batch)
            delta = (outputs[0] - outputs[1]).item()

            is_tp = bool(sub_labels[idx, concept_idx])
            all_rows.append((concept_name, is_tp, delta, abs(delta)))

        print(f"  [{concept_idx + 1:>2d}/26] {concept_name:<20s} n_true_pos={n_tp:>2d}/{K} kept={n_kept}", flush=True)

    with open(RESULTS_DIR / "celeba_pool_size_1pct_tp_vs_fp_delta_magnitude.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "is_true_positive", "delta_score", "abs_delta_score"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} per-image rows", flush=True)

    tp_abs = [r[3] for r in all_rows if r[1]]
    fp_abs = [r[3] for r in all_rows if not r[1]]
    print("\n=== pooled across all 26 concepts ===")
    print(f"  true-positive images:  n={len(tp_abs)}  mean|delta|={np.mean(tp_abs):.4f}  median={np.median(tp_abs):.4f}")
    print(f"  false-positive images: n={len(fp_abs)}  mean|delta|={np.mean(fp_abs):.4f}  median={np.median(fp_abs):.4f}")
    if tp_abs and fp_abs:
        u, p = mannwhitneyu(tp_abs, fp_abs, alternative="two-sided")
        print(f"  Mann-Whitney U test (TP |delta| vs FP |delta|): U={u:.1f} p={p:.4g}")
        print(f"  ratio mean|delta_tp| / mean|delta_fp| = {np.mean(tp_abs) / np.mean(fp_abs):.3f}")

    print("\n=== per-concept: mean|delta| for TP vs FP images ===")
    for c in GROUNDABLE_CONCEPTS:
        c_tp = [r[3] for r in all_rows if r[0] == c and r[1]]
        c_fp = [r[3] for r in all_rows if r[0] == c and not r[1]]
        if not c_tp or not c_fp:
            print(f"  {c:<20s} n_tp={len(c_tp):>2d} n_fp={len(c_fp):>2d}  (one group empty, skipped)")
            continue
        print(f"  {c:<20s} n_tp={len(c_tp):>2d} mean|delta_tp|={np.mean(c_tp):.4f}   "
              f"n_fp={len(c_fp):>2d} mean|delta_fp|={np.mean(c_fp):.4f}   "
              f"ratio={np.mean(c_tp) / np.mean(c_fp):.2f}")


if __name__ == "__main__":
    main()
