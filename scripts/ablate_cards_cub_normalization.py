"""Ablates Step 7's score normalization -- never wired into any CUB
script so far (every CARDS-on-CUB run has scored raw `raw_score`
directly, bypassing normalize_score entirely, the same gap
orthogonalization had before v46). Prompted directly: "Can we run
ablations on normalization techniques."

Five conditions, computed from the SAME retrieval per concept (current
default: aligned_retrieval, K=50, demean_query=True, baseline phrasing,
SigLIP -- v47's winning config) so this isolates normalization alone,
no retrieval cost paid twice:
- raw: raw_score directly (what every prior run has used).
- variance: Cohen's-d-style, raw_score / pooled_std(present union absent
  outputs) -- per class, since present/absent logits are (k, 200).
- embedding_distance_euclidean: raw_score / delta_c, delta_c = Step 4's
  ||mean(P_c)-mean(N_c)|| in embedding space (one value per concept,
  reused across all 200 classes).
- embedding_distance_angular (origin-relative): raw_score / delta_c,
  delta_c = 1 - cosine_similarity(present_centroid, absent_centroid)
  instead of the Euclidean norm (Step 7's own angular sub-ablation,
  cards.attribution.normalization.angular_distance) -- cosine similarity
  is implicitly "angle as seen from the origin."
- embedding_distance_angular (cluster-centered): the same angular
  calculation, but both centroids first have the POOL's own mean image
  embedding subtracted -- prompted directly ("Should we consider the
  angular distance from the center of image cluster or the origin?").
  CLIP-style joint embedding spaces are known to cluster in a narrow
  cone far from the origin (the same "modality gap" demean_query already
  corrects for on the text-query side, see feedback_demean_query_per_
  dataset_encoder memory) -- raw origin-relative cosine similarities
  between two centroids both sitting deep in that cone can be
  compressed/poorly calibrated; centering on the cluster's own mean
  first is the standard fix, tested here as its own condition rather
  than assumed to help.
- embedding_distance_query_projection: delta_c = t_c_unit . d_c (a
  SIGNED linear projection of the raw difference vector d_c =
  mean(P_c)-mean(N_c) onto the ORIGINAL TEXT QUERY's own unit direction,
  not d_c's own direction) -- prompted directly ("Can we also ablate for
  the distance along the concept vector?"). Distinct from
  embedding_distance_euclidean: projecting d_c onto its OWN direction
  trivially returns ||d_c|| (already tested), so "distance along the
  concept vector" is only a new quantity if "concept vector" means t_c,
  the direction actually asked for -- exactly the quantity
  cards.retrieval.aligned's own docstring names as what naive bottom-k
  retrieval already optimizes (t_c . d_c). Unlike every other condition
  here, this divisor CAN be negative (t_c and d_c aren't guaranteed to
  point the same way, especially under `matched`/`naive` where v45 found
  them ~80-83 degrees apart) -- a genuine, checkable difference from the
  other conditions, all of which use a non-negative divisor and are
  therefore mathematically guaranteed to tie on sign agreement.

Scored against the 87-attribute bank (Part 2, the primary evidence base
per v48).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_cards_cub_attributes import PREFIX_TEMPLATES  # noqa: E402

from cards.attribution.normalization import angular_distance, embedding_distance_normalize, variance_normalize  # noqa: E402
from cards.data.cub_attributes import groundable_attributes, load_attribute_names  # noqa: E402
from cards.data.cub_parts import load_images_txt  # noqa: E402
from cards.directions.estimate import estimate_direction  # noqa: E402
from cards.models.backbones import BACKBONES  # noqa: E402
from cards.pipeline import instantiate_encoder  # noqa: E402
from cards.retrieval.aligned import aligned_retrieval  # noqa: E402
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool  # noqa: E402
from cards.retrieval.retrieve import retrieve_top_bottom_k  # noqa: E402
from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query  # noqa: E402
from cards.validation.broden_faithfulness import (  # noqa: E402
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
ATTRIBUTE_NAMES_PATH = CUB_ROOT / "attributes" / "new_attributes.txt"
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50


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
    print(f"{len(faithfulness_records)} faithfulness records loaded.", flush=True)

    attribute_names = load_attribute_names(ATTRIBUTE_NAMES_PATH)
    groundable = groundable_attributes(attribute_names)

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
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    pool_mean = pool.embeddings.mean(dim=0)
    print(f"pool mean embedding norm: {pool_mean.norm().item():.4f} "
          f"(nonzero => the image cluster sits off-origin, consistent with the CLIP modality-gap cone)", flush=True)

    scores_raw: dict[tuple[int, int], float] = {}
    scores_variance: dict[tuple[int, int], float] = {}
    scores_embed_euclidean: dict[tuple[int, int], float] = {}
    scores_embed_angular_origin: dict[tuple[int, int], float] = {}
    scores_embed_angular_centered: dict[tuple[int, int], float] = {}
    scores_embed_query_projection: dict[tuple[int, int], float] = {}

    n_variance_skipped = 0
    n_embed_skipped = 0
    n_embed_centered_skipped = 0
    n_embed_projection_skipped = 0
    n_negative_projection = 0

    for attr_idx, (prefix, _part_names) in groundable.items():
        value = attribute_names[attr_idx].split("::", 1)[1]
        text = PREFIX_TEMPLATES[prefix].format(value=readable(value))
        t_c = build_concept_query(text, encoder)
        t_c = demean_query(t_c, text_center)

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
        absent_indices = aligned_retrieval(pool, present_indices, t_c, K)

        present_paths = [pool.paths[i] for i in present_indices]
        absent_paths = [pool.paths[i] for i in absent_indices]
        present_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in present_paths]).to(DEVICE)
        absent_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in absent_paths]).to(DEVICE)

        with torch.no_grad():
            present_logits = native_model(present_batch)  # (k, 200)
            absent_logits = native_model(absent_batch)

        raw_score_all_classes = (present_logits.mean(dim=0) - absent_logits.mean(dim=0))

        direction = estimate_direction(str(attr_idx), pool.embeddings[present_indices], pool.embeddings[absent_indices])
        delta_c_euclidean = direction.magnitude
        present_centroid = pool.embeddings[present_indices].mean(dim=0)
        absent_centroid = pool.embeddings[absent_indices].mean(dim=0)
        delta_c_angular_origin = angular_distance(present_centroid, absent_centroid)
        delta_c_angular_centered = angular_distance(present_centroid - pool_mean, absent_centroid - pool_mean)

        d_c = present_centroid - absent_centroid  # raw (unnormalized) difference vector
        t_c_unit = F.normalize(t_c, dim=0)
        delta_c_query_projection = float(t_c_unit @ d_c)  # signed -- can be negative, unlike every other delta_c here
        if delta_c_query_projection < 0:
            n_negative_projection += 1

        for native_idx in range(200):
            raw_val = raw_score_all_classes[native_idx].item()
            scores_raw[(attr_idx, native_idx)] = raw_val

            try:
                scores_variance[(attr_idx, native_idx)] = variance_normalize(
                    present_logits[:, native_idx], absent_logits[:, native_idx]
                )
            except ValueError:
                n_variance_skipped += 1

            try:
                scores_embed_euclidean[(attr_idx, native_idx)] = embedding_distance_normalize(raw_val, delta_c_euclidean)
                scores_embed_angular_origin[(attr_idx, native_idx)] = embedding_distance_normalize(raw_val, delta_c_angular_origin)
            except ValueError:
                n_embed_skipped += 1

            try:
                scores_embed_angular_centered[(attr_idx, native_idx)] = embedding_distance_normalize(raw_val, delta_c_angular_centered)
            except ValueError:
                n_embed_centered_skipped += 1

            try:
                scores_embed_query_projection[(attr_idx, native_idx)] = embedding_distance_normalize(raw_val, delta_c_query_projection)
            except ValueError:
                n_embed_projection_skipped += 1

    print(f"\n(variance-normalize skipped {n_variance_skipped} zero-variance (concept,class) pairs; "
          f"embedding-distance skipped {n_embed_skipped} zero-delta_c pairs, "
          f"{n_embed_centered_skipped} zero-delta_c-centered pairs, "
          f"{n_embed_projection_skipped} zero-query-projection pairs)", flush=True)
    print(f"query-projection delta_c was NEGATIVE for {n_negative_projection}/{len(groundable)} concepts "
          f"(t_c and d_c pointing more than 90 degrees apart)", flush=True)

    results = []
    for label, scores in [
        ("raw (no normalization -- what every prior run used)", scores_raw),
        ("variance (Cohen's-d-style)", scores_variance),
        ("embedding_distance (euclidean delta_c)", scores_embed_euclidean),
        ("embedding_distance (angular delta_c, origin-relative)", scores_embed_angular_origin),
        ("embedding_distance (angular delta_c, cluster-centered)", scores_embed_angular_centered),
        ("embedding_distance (query-projection delta_c, signed)", scores_embed_query_projection),
    ]:
        evaluate_and_log(label, faithfulness_records, scores, results)

    with open(RESULTS_DIR / "cards_cub_normalization_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for label, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([label, "", "", "", "", "", ""])
            else:
                writer.writerow([label, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])

    print(f"\nSaved {len(results)} configs to results/cards_cub_normalization_ablation.csv")


if __name__ == "__main__":
    main()
