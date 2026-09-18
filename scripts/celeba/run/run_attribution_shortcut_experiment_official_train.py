"""Official-train counterpart of `run_attribution_shortcut_experiment.py`
-- same question (does the masking hybrid's own attribution correctly
DECLINE toward real semantic concepts as a model becomes more shortcut-
reliant?), now against the 22 official-train shortcut checkpoints
(`train_{attractive,male}_shortcut_classifiers_official_train.py`: 2
tasks x 11 rates, 0/10/.../100%) instead of the original 4 HQ-trained
ones.

SigLIP only, same validated config as the original (demean=True,
orthogonalize=True, z-score alpha=1.0, K=50) -- this experiment tests
attribution behavior across classifiers, not correlation-with-ground-
truth optimality, so the exact threshold config matters less here than
in the tuning ablations. Prompted directly ("Can we do both 1. and 2.?"
-- item 2 being "Run the ConceptMask/masking-hybrid shortcut-reliance
analysis ... extend the existing shortcut-detection scripts ... to
these new official-train Attractive+Male checkpoints").

Retrieval/localization against each of the 26 GROUNDABLE_CONCEPTS is
CLASSIFIER- and TASK-independent (depends only on encoder/pool/query),
so it's cached ONCE and reused across BOTH tasks and all 11 rates --
same reasoning and mechanism as the original script's own single-task
cache, just widened by one more shared axis (task).

HPC-portable: CELEBA_ROOT/CARDS_RESULTS_DIR/CARDS_CKPT_DIR are env-var
overridable, same convention as the training scripts. Uses ONLY official
CelebA (val partition, `list_eval_partition.txt`==1) -- no CelebA-HQ
dependency at all (CelebA-HQ isn't available on the HPC cluster this
runs on; confirmed directly, "I do not have the CELEBA_HQ downloaded
there"), unlike the original script's own `build_clean_official_val_
paths()` call, which excluded CelebA-HQ-train-leaked images. That
exclusion isn't needed here anyway: these classifiers train on official
partition==0 only, already disjoint from partition==1 by construction.
"""

from __future__ import annotations

import csv
import os
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
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import mask_region

CELEBA_ROOT = Path(os.environ.get("CELEBA_ROOT", r"C:\Users\btokas\Projects\Datasets\CelebA\celeba"))
RESULTS_DIR = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
CKPT_DIR = Path(os.environ.get("CARDS_CKPT_DIR", "trained_models_new/celeba"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
ALPHA = 1.0
SEED = 42
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
TASKS = ["Attractive", "Male"]
RATES_PCT = list(range(0, 101, 10))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_official_val_paths() -> list[Path]:
    """Plain official CelebA val partition (list_eval_partition.txt,
    partition==1) -- no CelebA-HQ leakage exclusion needed here, unlike
    build_clean_official_val_paths(): these official-train shortcut
    classifiers train on official partition==0 only, which is disjoint
    from partition==1 by construction, so there's no leakage to exclude
    (confirmed directly -- "I do not have the CELEBA_HQ downloaded
    there," prompting this switch away from CelebA-HQ entirely)."""
    val_paths = []
    with open(CELEBA_ROOT / "list_eval_partition.txt") as f:
        for line in f:
            fname, part = line.split()
            if part == "1":
                val_paths.append(CELEBA_ROOT / "img_align_celeba" / fname)
    return val_paths


class ShortcutBlackBox:
    """Wraps one (task, rate)'s checkpoint (single 2-way head, positive-
    class logit at index 1) as `b(x) -> scalar`."""

    def __init__(self, task: str, rate_pct: int, device: str):
        self.device = device
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        ckpt_path = CKPT_DIR / f"resnet18_official_train_{task.lower()}_shortcut_{rate_pct}.pt"
        state = torch.load(ckpt_path, map_location="cpu")
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

    official_paths = build_official_val_paths()
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

    # Cache localization ONCE across BOTH tasks and all 11 rates -- depends
    # only on (encoder, image, t_c), never on which classifier is explained.
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

    all_rows = []  # (task, rate_pct, concept_name, raw_score)
    scores_by_task_rate: dict[str, dict[int, dict[str, float]]] = {t: {} for t in TASKS}

    for task in TASKS:
        for rate_pct in RATES_PCT:
            print(f"\n=== task={task} rate={rate_pct}% ===", flush=True)
            black_box = ShortcutBlackBox(task, rate_pct, DEVICE)
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
                all_rows.append((task, rate_pct, concept_name, score))

            scores_by_task_rate[task][rate_pct] = scores
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            print(f"  top-5: {[(c, round(s, 4)) for c, s in ranked[:5]]}", flush=True)
            print(f"  mean |raw_score| across all {len(GROUNDABLE_CONCEPTS)} concepts: "
                  f"{np.mean([abs(s) for s in scores.values()]):.4f}", flush=True)

    out_path = RESULTS_DIR / "cards_celeba_official_train_shortcut_experiment_raw_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "rate_pct", "concept_name", "hybrid_raw_score"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")

    print("\n=== mean |raw_score| across all concepts, per task, per rate (the headline decline check) ===")
    for task in TASKS:
        for rate_pct in RATES_PCT:
            mean_abs = np.mean([abs(s) for s in scores_by_task_rate[task][rate_pct].values()])
            print(f"  task={task:<10s} rate={rate_pct:>3d}%: mean |raw_score| = {mean_abs:.4f}")


if __name__ == "__main__":
    main()
