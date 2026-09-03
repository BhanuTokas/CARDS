"""Evaluates the masking hybrid's own concept attribution against all 4
shortcut-injected classifiers (0%/33%/67%/100%, see
scripts/celeba/build/train_attractive_shortcut_classifiers.py) --
the actual point of the shortcut experiment: does the attribution
method correctly detect DECLINING reliance on real semantic concepts as
a model becomes more shortcut-reliant?

No ground-truth comparison here (no celeba_full_faithfulness.csv) --
this experiment isn't measuring correlation with real per-pixel masks,
it's comparing the method's own attribution OUTPUT across 4 differently-
trained models. Retrieval/scoring runs against the SAME, always-CLEAN
CelebA-HQ val pool every other config in this track uses -- raw images
loaded straight from disk never pass through the shortcut-injecting
Dataset class, so this deliberately tests whether the method still
finds real concepts meaningful on natural (shortcut-free) images, not
just whether masking a concept region happens to also touch the
(spatially unrelated, untouched-by-masking) shortcut block.

Config: SigLIP, demean=True, orthogonalize=True, z-score alpha=1.0, K=50
-- the current best-validated HQ-val config (v96). Scores the full
26-concept GROUNDABLE_CONCEPTS bank for each of the 4 checkpoints.

Reports three views: (1) each model's own top-5 concepts by raw_score,
(2) the SAME fixed 5 concepts (the 0%-model's own top-5) tracked across
all 4 rates for a clean apples-to-apples decline check, (3) mean
|raw_score| across all 26 concepts per rate, the single cleanest summary
number for the "steady decline" hypothesis.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.data.datasets import load_celeba
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import mask_region

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
CKPT_DIR = Path("trained_models_new/celeba")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
ALPHA = 1.0
SEED = 42
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
RATES_PCT = [0, 33, 67, 100]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ShortcutBlackBox:
    """Wraps one rate's checkpoint (single 2-way head, positive-class
    logit at index 1) as `b(x) -> scalar` -- the BlackBoxModel contract."""

    def __init__(self, rate_pct: int, device: str):
        self.device = device
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        state = torch.load(CKPT_DIR / f"resnet18_attractive_shortcut_{rate_pct}.pt", map_location="cpu")
        model.load_state_dict(state)
        self.model = model.to(device).eval()
        self._preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image.convert("RGB"))

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch.to(self.device))[:, 1].detach().cpu()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    cfg.dataset = {"name": "celeba", "root": str(CELEBA_HQ_ROOT)}
    cfg.pool_source = "val"
    pairs = load_celeba(CELEBA_HQ_ROOT, split="val")
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images (always clean -- shortcut injection never touches this pool)", flush=True)

    raw_queries = {
        c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], encoder), text_center)
        for c in GROUNDABLE_CONCEPTS
    }
    queries = orthogonalize_queries(raw_queries)

    # cache localization ONCE across all 4 rates -- retrieval/localization
    # depend only on (encoder, image, t_c), never on which classifier is
    # being explained, so this is shared work across the whole rate sweep.
    concept_cache: dict[str, tuple[list[int], list[tuple]]] = {}
    for concept_name in GROUNDABLE_CONCEPTS:
        t_c = queries[concept_name]
        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
        cached = []
        for idx in present_indices:
            image = Image.open(pool.paths[idx]).convert("RGB")
            sim_map = localize_concept(encoder, image, t_c, (image.height, image.width))
            cached.append((idx, image, sim_map))
        concept_cache[concept_name] = (present_indices, cached)
        print(f"localized {concept_name}", flush=True)

    all_rows = []  # (rate_pct, concept_name, raw_score)
    scores_by_rate: dict[int, dict[str, float]] = {}

    for rate_pct in RATES_PCT:
        print(f"\n=== rate={rate_pct}% ===", flush=True)
        black_box = ShortcutBlackBox(rate_pct, DEVICE)
        scores: dict[str, float] = {}

        for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
            t_c = queries[concept_name]
            t_c_dev = t_c.to(DEVICE)
            _present_indices, cached = concept_cache[concept_name]
            sim_maps = [sm for _, _, sm in cached]
            cutoff = concept_zscore_cutoff(sim_maps, ALPHA)

            delta_scores = []
            for idx, image, sim_map in cached:
                mask = threshold_mask(sim_map, method="fixed", cutoff=cutoff)
                if not mask.any() or mask.all():
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

                pixels_orig = black_box.preprocess(image).unsqueeze(0)
                pixels_masked = black_box.preprocess(masked_image).unsqueeze(0)
                batch = torch.cat([pixels_orig, pixels_masked], dim=0)
                outputs = black_box(batch)
                delta_scores.append((outputs[0] - outputs[1]).item())

            score = float(np.mean(delta_scores)) if delta_scores else 0.0
            scores[concept_name] = score
            all_rows.append((rate_pct, concept_name, score))

        scores_by_rate[rate_pct] = scores
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        print(f"  top-5: {[(c, round(s, 4)) for c, s in ranked[:5]]}", flush=True)
        print(f"  mean |raw_score| across all 26 concepts: {np.mean([abs(s) for s in scores.values()]):.4f}", flush=True)

    with open(RESULTS_DIR / "cards_celeba_shortcut_experiment_raw_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rate_pct", "concept_name", "hybrid_raw_score"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to results/cards_celeba_shortcut_experiment_raw_scores.csv")

    # --- view 1: each model's own top-5 ---
    print("\n=== each model's own top-5 concepts ===")
    for rate_pct in RATES_PCT:
        ranked = sorted(scores_by_rate[rate_pct].items(), key=lambda kv: -kv[1])
        print(f"  rate={rate_pct:>3d}%: " + ", ".join(f"{c}={s:+.3f}" for c, s in ranked[:5]))

    # --- view 2: the 0%-model's own top-5, tracked across all rates ---
    baseline_top5 = [c for c, _ in sorted(scores_by_rate[0].items(), key=lambda kv: -kv[1])[:5]]
    print(f"\n=== rate=0% baseline's own top-5 concepts, tracked across all rates ({baseline_top5}) ===")
    for concept_name in baseline_top5:
        trajectory = [scores_by_rate[r][concept_name] for r in RATES_PCT]
        print(f"  {concept_name:<20s} " + " -> ".join(f"{v:+.4f}" for v in trajectory))

    # --- view 3: mean |raw_score| across all 26 concepts, per rate ---
    print("\n=== mean |raw_score| across all 26 concepts, per rate (the headline decline check) ===")
    for rate_pct in RATES_PCT:
        mean_abs = np.mean([abs(s) for s in scores_by_rate[rate_pct].values()])
        print(f"  rate={rate_pct:>3d}%: mean |raw_score| = {mean_abs:.4f}")


if __name__ == "__main__":
    main()
