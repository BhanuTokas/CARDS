"""Follow-up to v115g, prompted directly ("Could it be that some of the
false positives are actually true?"): CelebA's own attribute labels are
known to be noisy (crowd/classifier-derived, not hand-verified per
image) -- some of what got counted as a "false positive" in v115g might
actually have the attribute and just be mislabeled, which would inflate
false-positive |delta_score| and understate how much true positives
actually dominate.

Reproduces the SAME 1%-pool (seed=0), SAME queries, for a small set of
concepts, but this time saves the actual image PATH alongside each
present-set image's label and |delta_score| -- so the most "suspicious"
false positives (large masking effect, behaving like a true positive)
can be visually inspected directly, rather than just counted.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

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
CONCEPTS_TO_CHECK = ["Eyeglasses", "Gray_Hair", "Bald", "Wearing_Hat"]


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

    raw_queries = {c: build_concept_query(CONCEPT_QUERY_TEXT[c], encoder) for c in GROUNDABLE_CONCEPTS}
    demeaned = {c: demean_query(q, text_center) for c, q in raw_queries.items()}
    queries = orthogonalize_queries(demeaned)

    all_rows = []  # (concept, image_path, is_true_positive, delta_score, abs_delta_score, sim)

    for concept_name in CONCEPTS_TO_CHECK:
        concept_idx = CONCEPT_TO_IDX[concept_name]
        t_c = queries[concept_name]
        sims = (sub_embeds @ t_c.to(sub_embeds.dtype)).numpy()
        present_indices = np.argsort(-sims)[:K].tolist()

        images = {idx: Image.open(sub_paths[idx]).convert("RGB") for idx in present_indices}
        sim_maps = {idx: localize_concept(encoder, images[idx], t_c, (images[idx].height, images[idx].width))
                    for idx in present_indices}
        cutoff = concept_zscore_cutoff(list(sim_maps.values()), ALPHA)

        for idx in present_indices:
            image = images[idx]
            mask = threshold_mask(sim_maps[idx], method="fixed", cutoff=cutoff)
            if not mask.any() or mask.all():
                continue

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
            all_rows.append((concept_name, str(sub_paths[idx]), is_tp, delta, abs(delta), float(sims[idx])))

    with open(RESULTS_DIR / "celeba_suspicious_false_positives.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_name", "image_path", "is_true_positive", "delta_score", "abs_delta_score", "similarity"])
        writer.writerows(all_rows)
    print(f"Saved {len(all_rows)} rows", flush=True)

    for concept_name in CONCEPTS_TO_CHECK:
        rows = [r for r in all_rows if r[0] == concept_name]
        fps = sorted((r for r in rows if not r[2]), key=lambda r: -r[4])
        tps = sorted((r for r in rows if r[2]), key=lambda r: -r[4])
        print(f"\n=== {concept_name}: top 5 FALSE POSITIVES by |delta_score| (most TP-like behavior) ===")
        for r in fps[:5]:
            print(f"  |delta|={r[4]:.4f}  sim={r[5]:.4f}  {r[1]}")
        print("  -- for comparison, top 5 TRUE positives: --")
        for r in tps[:5]:
            print(f"  |delta|={r[4]:.4f}  sim={r[5]:.4f}  {r[1]}")


if __name__ == "__main__":
    main()
