"""Tests orthogonalization meaningfully, unlike a bare cfg.orthogonalize
toggle -- per direct verification against cards/pipeline.py, orthogonalized
directions currently only get saved to disk (save_directions), never
consumed by global_score/normalize_score (Steps 6-7), so flipping that
flag alone would produce byte-identical raw_score numbers. This script
instead builds the missing consumer: a projection-based retrieval mode
that uses a concept's own embedding-space DIRECTION (rather than raw
cosine-similarity-to-text-query) to select present/absent images, so
orthogonalizing that direction against the other 86 concepts' own
directions can actually change which images get selected and therefore
the resulting score.

Three conditions per concept, same K=30/no-demean/baseline-phrasing
settings used everywhere else (isolating retrieval-ranking-criterion as
the one variable under test, not re-crossing the full K/demean/phrasing
grid):
1. "text_query": the original method -- rank the pool by cosine
   similarity to the concept's own text query (what every prior CARDS-
   on-CUB run in this investigation has used).
2. "direction": rank the pool by projection onto the concept's OWN
   difference-in-means direction (P_c/N_c centroids from condition 1's
   own retrieval) -- direction-based ranking, not yet orthogonalized.
3. "direction_orthogonal": same, but the direction is first Lowdin-
   orthogonalized against all 86 other concepts' own directions
   (cards.directions.orthogonalize.lowdin_orthogonalize) before ranking.

retrieve_top_bottom_k already ranks by `pool.embeddings @ query_vector`
(cosine similarity, since both are L2-normalized) -- a concept's own
unit_vector is exactly this shape, so conditions 2/3 reuse it directly,
no new retrieval code needed.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ablate"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_cub_attributes import PREFIX_TEMPLATES  # noqa: E402

from cards.data.cub_attributes import groundable_attributes, load_attribute_names  # noqa: E402
from cards.data.cub_parts import load_images_txt  # noqa: E402
from cards.directions.estimate import ConceptDirection, estimate_direction  # noqa: E402
from cards.directions.orthogonalize import lowdin_orthogonalize  # noqa: E402
from cards.models.backbones import BACKBONES  # noqa: E402
from cards.pipeline import instantiate_encoder  # noqa: E402
from cards.retrieval.confound import matched_retrieval  # noqa: E402
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool  # noqa: E402
from cards.retrieval.retrieve import retrieve_top_bottom_k  # noqa: E402
from cards.concepts.prompts import build_concept_query  # noqa: E402
from cards.validation.broden_faithfulness import (  # noqa: E402
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
ATTRIBUTE_NAMES_PATH = CUB_ROOT / "attributes" / "new_attributes.txt"
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 30


def readable(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


def load_faithfulness_records() -> list[FaithfulnessResult]:
    records = []
    with open(RESULTS_DIR / "cub_attribute_faithfulness.csv", newline="") as f:
        for row in csv.DictReader(f):
            records.append(
                FaithfulnessResult(
                    image=row["image"], concept_number=int(row["concept_number"]), category=row["category"],
                    predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                    delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                    random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                    z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
                )
            )
    return records


def score_from_indices(spec, native_model, pool, present_indices, absent_indices) -> list[float]:
    present_paths = [pool.paths[i] for i in present_indices]
    absent_paths = [pool.paths[i] for i in absent_indices]
    present_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in present_paths]).to(DEVICE)
    absent_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in absent_paths]).to(DEVICE)
    with torch.no_grad():
        present_logits = native_model(present_batch)
        absent_logits = native_model(absent_batch)
    return (present_logits.mean(dim=0) - absent_logits.mean(dim=0)).tolist()


def evaluate_and_log(label, records, scores, results):
    rho_result = score_method_agreement(records, scores, min_samples_per_pair=3)
    sign_result = score_sign_agreement(records, scores, min_samples_per_pair=3)
    results.append((label, rho_result, sign_result))
    if rho_result is None:
        print(f"[{label}] too few pairs", flush=True)
    else:
        print(f"[{label}] n={rho_result.n_pairs} rho={rho_result.spearman_rho:+.4f} p={rho_result.spearman_p:.4g} "
              f"| sign={sign_result.agreement_frac:.1%} ({sign_result.n_agree}/{sign_result.n_pairs}) "
              f"binom_p={sign_result.binom_p:.4g}", flush=True)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    faithfulness_records = load_faithfulness_records()
    print(f"{len(faithfulness_records)} attribute-level faithfulness records loaded (unchanged ground truth).", flush=True)

    attribute_names = load_attribute_names(ATTRIBUTE_NAMES_PATH)
    groundable = groundable_attributes(attribute_names)
    print(f"{len(groundable)} groundable attributes.", flush=True)

    image_paths = load_images_txt(CUB_ROOT)
    class_labels = {}
    for line in (CUB_ROOT / "image_class_labels.txt").read_text().splitlines():
        image_id, class_id = line.split()
        class_labels[image_id] = int(class_id) - 1
    test_ids = [
        line.split()[0]
        for line in (CUB_ROOT / "train_test_split.txt").read_text().splitlines()
        if line.split()[1] == "0"
    ]

    print("\nLoading SigLIP + pool...", flush=True)
    encoder_cfg = OmegaConf.create({"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                                     "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE})
    encoder = instantiate_encoder(OmegaConf.create({"encoder": encoder_cfg, "device": DEVICE}))

    pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": encoder_cfg, "cache_dir": "embedding_cache"})
    pool_cfg.dataset = {"name": "cub", "root": str(CUB_ROOT)}
    pool_cfg.pool_source = "test"
    pairs = [(image_paths[i], class_labels[i]) for i in test_ids]
    pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)

    spec = BACKBONES["resnet18_cub"]
    native_model = spec.load_native().to(DEVICE).eval()

    print("\n=== Step 1: text-query retrieval + direction estimation per concept ===", flush=True)
    text_query_scores: dict[tuple[int, int], float] = {}
    directions: dict[int, ConceptDirection] = {}
    present_by_attr: dict[int, list[int]] = {}
    absent_by_attr: dict[int, list[int]] = {}

    for attr_idx, (prefix, _part_names) in groundable.items():
        value = attribute_names[attr_idx].split("::", 1)[1]
        text = PREFIX_TEMPLATES[prefix].format(value=readable(value))
        t_c = build_concept_query(text, encoder)

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
        absent_indices = matched_retrieval(pool, present_indices, t_c)
        present_by_attr[attr_idx] = present_indices
        absent_by_attr[attr_idx] = absent_indices

        scores = score_from_indices(spec, native_model, pool, present_indices, absent_indices)
        for native_idx, s in enumerate(scores):
            text_query_scores[(attr_idx, native_idx)] = s

        directions[attr_idx] = estimate_direction(
            str(attr_idx), pool.embeddings[present_indices], pool.embeddings[absent_indices]
        )

    print(f"{len(directions)} directions estimated.", flush=True)

    print("\n=== Step 2: direction-based retrieval (no orthogonalization) ===", flush=True)
    direction_scores: dict[tuple[int, int], float] = {}
    for attr_idx, direction in directions.items():
        present_indices, absent_indices = retrieve_top_bottom_k(pool, direction.unit_vector, K)
        scores = score_from_indices(spec, native_model, pool, present_indices, absent_indices)
        for native_idx, s in enumerate(scores):
            direction_scores[(attr_idx, native_idx)] = s

    print("\n=== Step 3: Lowdin orthogonalization across all 87 directions ===", flush=True)
    ordered_idx = list(directions.keys())
    ordered_directions = [directions[i] for i in ordered_idx]
    try:
        orthogonalized = lowdin_orthogonalize(ordered_directions)
        print("Orthogonalization succeeded.", flush=True)
    except ValueError as e:
        print(f"Orthogonalization FAILED: {e}", flush=True)
        orthogonalized = None

    results = []
    evaluate_and_log("text_query (baseline)", faithfulness_records, text_query_scores, results)
    evaluate_and_log("direction (no orthogonalization)", faithfulness_records, direction_scores, results)

    if orthogonalized is not None:
        print("\n=== Step 4: orthogonalized-direction-based retrieval ===", flush=True)
        ortho_scores: dict[tuple[int, int], float] = {}
        for attr_idx, direction in zip(ordered_idx, orthogonalized):
            present_indices, absent_indices = retrieve_top_bottom_k(pool, direction.unit_vector, K)
            scores = score_from_indices(spec, native_model, pool, present_indices, absent_indices)
            for native_idx, s in enumerate(scores):
                ortho_scores[(attr_idx, native_idx)] = s
        evaluate_and_log("direction_orthogonal", faithfulness_records, ortho_scores, results)

    with open(RESULTS_DIR / "cards_cub_orthogonalize_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for label, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([label, "", "", "", "", "", ""])
            else:
                writer.writerow([label, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])

    print(f"\nSaved {len(results)} configs to results/cards_cub_orthogonalize_ablation.csv")


if __name__ == "__main__":
    main()
