"""Tests orthogonalization as a REAL, optional pipeline component --
applied to the query vector, after demeaning, before retrieval -- unlike
the existing `cfg.orthogonalize` toggle in `cards/pipeline.py`, which
only overwrites the direction saved to `directions.pt` AFTER raw_score is
already computed (confirmed a no-op for scoring, v44), and unlike
`ablate_cards_cub_orthogonalize.py`'s own test (which orthogonalized an
ESTIMATED direction and used it for projection-based retrieval, a
different retrieval mechanism entirely). Prompted directly ("Can we add
orthogonalization as an optional component post demeaning?").

Mechanism: build all 87 concepts' queries t_c (per-attribute, optionally
demeaned via the existing `demean_query`), then jointly Lowdin-
orthogonalize the FULL 87-vector set via `cards.directions.orthogonalize.
lowdin_orthogonalize` (wrapping each query in a throwaway `ConceptDirection`
just to reuse that function's own Gram-matrix math -- magnitude is unused
here, set to 0.0) -- producing 87 mutually-orthonormal query vectors, one
per concept. Each concept's ORTHOGONALIZED query then drives its own
retrieval (`retrieve_top_bottom_k` + `aligned_retrieval`) exactly as the
plain query would otherwise. `raw_score` is computed from whatever images
that orthogonalized query retrieves -- a genuine, scoring-path-affecting
use of orthogonalization, not a post-hoc no-op.

Fixed at SigLIP / K=50 / baseline phrasing (the current best-known
config, v47/v56) -- demean x orthogonalize is a real 2x2 (4 configs),
covering the exact cell the prompt asked about (demean=True,
orthogonalize=True) plus the other three for context.
"""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ablate"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from ablate_cards_cub_attributes import ENCODER_CONFIGS
from run_cards_cub_attributes import PREFIX_TEMPLATES

from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.cub_attributes import groundable_attributes, load_attribute_names
from cards.data.cub_parts import load_images_txt
from cards.directions.estimate import ConceptDirection
from cards.directions.orthogonalize import lowdin_orthogonalize
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder
from cards.retrieval.aligned import aligned_retrieval
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import (
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
            records.append(FaithfulnessResult(
                image=row["image"], concept_number=int(row["concept_number"]), category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return records


def build_all_queries(groundable, attribute_names, encoder, use_demean: bool, text_center) -> dict[int, torch.Tensor]:
    queries = {}
    for attr_idx, (prefix, _part_names) in groundable.items():
        value = attribute_names[attr_idx].split("::", 1)[1]
        t_c = build_concept_query(PREFIX_TEMPLATES[prefix].format(value=readable(value)), encoder)
        if use_demean:
            t_c = demean_query(t_c, text_center)
        queries[attr_idx] = t_c
    return queries


def orthogonalize_queries(queries: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    import torch.nn.functional as F

    attr_indices = list(queries.keys())
    directions = [ConceptDirection(concept=str(i), unit_vector=F.normalize(queries[i], dim=0), magnitude=0.0)
                  for i in attr_indices]
    orthogonalized = lowdin_orthogonalize(directions)
    return {attr_idx: d.unit_vector for attr_idx, d in zip(attr_indices, orthogonalized)}


def run_config(groundable, pool, spec, native_model, queries: dict[int, torch.Tensor]) -> dict[tuple[int, int], float]:
    scores: dict[tuple[int, int], float] = {}
    for attr_idx in groundable:
        t_c = queries[attr_idx]
        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
        absent_indices = aligned_retrieval(pool, present_indices, t_c, K)

        present_paths = [pool.paths[i] for i in present_indices]
        absent_paths = [pool.paths[i] for i in absent_indices]
        present_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in present_paths]).to(DEVICE)
        absent_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in absent_paths]).to(DEVICE)

        with torch.no_grad():
            present_logits = native_model(present_batch)
            absent_logits = native_model(absent_batch)

        raw_score_all_classes = (present_logits.mean(dim=0) - absent_logits.mean(dim=0)).tolist()
        for native_idx, score in enumerate(raw_score_all_classes):
            scores[(attr_idx, native_idx)] = score
    return scores


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
    print(f"{len(faithfulness_records)} attribute-level faithfulness records loaded.", flush=True)

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

    spec = BACKBONES["resnet18_cub"]
    native_model = spec.load_native().to(DEVICE).eval()

    print("Loading SigLIP + pool...", flush=True)
    siglip_cfg = OmegaConf.create({"device": DEVICE, **ENCODER_CONFIGS["siglip"]})
    siglip_encoder = instantiate_encoder(OmegaConf.create({"encoder": siglip_cfg, "device": DEVICE}))
    pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": siglip_cfg, "cache_dir": "embedding_cache"})
    pool_cfg.dataset = {"name": "cub", "root": str(CUB_ROOT)}
    pool_cfg.pool_source = "test"
    pairs = [(image_paths[i], class_labels[i]) for i in test_ids]
    pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, siglip_encoder)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, siglip_encoder)

    results = []
    for use_demean, use_orth in itertools.product([False, True], [False, True]):
        queries = build_all_queries(groundable, attribute_names, siglip_encoder, use_demean, text_center)
        if use_orth:
            queries = orthogonalize_queries(queries)
        label = f"aligned_siglip_k{K}_demean{use_demean}_orth{use_orth}_baseline"
        scores = run_config(groundable, pool, spec, native_model, queries)
        evaluate_and_log(label, faithfulness_records, scores, results)

    with open(RESULTS_DIR / "cards_cub_query_orthogonalize_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for label, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([label, "", "", "", "", "", ""])
            else:
                writer.writerow([label, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])

    print(f"\nSaved {len(results)} configs to results/cards_cub_query_orthogonalize_ablation.csv")
    print("\n=== summary, sorted by sign agreement descending ===")
    scored = [(label, r, s) for label, r, s in results if r is not None]
    scored.sort(key=lambda t: -t[2].agreement_frac)
    for label, r, s in scored:
        print(f"{label:<55s} rho={r.spearman_rho:+.4f} (p={r.spearman_p:.4g})  "
              f"sign={s.agreement_frac:.1%} (p={s.binom_p:.4g})")


if __name__ == "__main__":
    main()
