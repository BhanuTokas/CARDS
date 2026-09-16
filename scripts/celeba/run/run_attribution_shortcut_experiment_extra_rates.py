"""Recovers the 33%/67% rows for the masking hybrid's shortcut experiment
(scripts/celeba/run/run_attribution_shortcut_experiment.py), whose original
0/33/67/100 run was overwritten by the later "smaller intervals" rerun
(range(0,101,10), which never touches 33/67). Identical logic to that
script, restricted to RATES_PCT = [33, 67] and APPENDING to the existing
results/cards_celeba_shortcut_experiment_raw_scores.csv rather than
overwriting it (the 0/10/.../100 rows already there are untouched).
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
from run_cards_celeba_masking_hybrid_official_val_zscore import build_clean_official_val_paths

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import mask_region

CELEBA_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebA\celeba")
RESULTS_DIR = Path("results")
CKPT_DIR = Path("trained_models_new/celeba")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
ALPHA = 1.0
SEED = 42
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}
RATES_PCT = [33, 67]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ShortcutBlackBox:
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

    official_paths = build_clean_official_val_paths()
    cfg.dataset = {"name": "celeba_official_val_clean", "root": str(CELEBA_ROOT)}
    cfg.pool_source = "val"
    pairs = [(p, 0) for p in official_paths]
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images (official CelebA val, always clean)", flush=True)

    raw_queries = {
        c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], encoder), text_center)
        for c in GROUNDABLE_CONCEPTS
    }
    queries = orthogonalize_queries(raw_queries)

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

    all_rows = []
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
        print(f"  mean |raw_score| across all 26 concepts: {np.mean([abs(s) for s in scores.values()]):.4f}", flush=True)

    with open(RESULTS_DIR / "cards_celeba_shortcut_experiment_raw_scores.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)
    print(f"\nAppended {len(all_rows)} rows to results/cards_celeba_shortcut_experiment_raw_scores.csv")


if __name__ == "__main__":
    main()
