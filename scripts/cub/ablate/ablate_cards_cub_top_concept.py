"""Re-runs the exact same 14-config CARDS grid as ablate_cards_cub_
attributes_aligned.py (SigLIP: K in {15,30,50} x demean x phrasing;
CLIP: K=30/baseline x demean), but scores each config against the NEW
top-1/top-3/top-5 per-class concept-IDENTIFICATION metric (top_concept_
per_class_analysis.py) alongside the usual rho/sign-agreement -- does
the config that wins on magnitude/sign agreement (K=50, demean=True,
SigLIP, baseline) also win on "does this method correctly pick out the
single most important concept for a class," or is that a different
question with a different answer? Prompted directly ("Can we ablate
CARDS along various settings to see what works best?" -- for the new
top-concept metric specifically, following "Or if the concept is in
like top 5?").
"""

from __future__ import annotations

import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_cards_cub_attributes import PREFIX_TEMPLATES  # noqa: E402
from ablate_cards_cub_attributes import ALT_PREFIX_TEMPLATES, ENCODER_CONFIGS  # noqa: E402

from cards.data.cub_attributes import groundable_attributes, load_attribute_names  # noqa: E402
from cards.data.cub_parts import load_images_txt  # noqa: E402
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
MIN_SAMPLES = 3
MIN_CONCEPTS_PER_CLASS = 2


def readable(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


def build_query(prefix: str, value: str, encoder, phrasing: str) -> torch.Tensor:
    base_text = PREFIX_TEMPLATES[prefix].format(value=readable(value))
    if phrasing == "baseline":
        return build_concept_query(base_text, encoder)
    alt_text = ALT_PREFIX_TEMPLATES[prefix].format(value=readable(value))
    embeddings = torch.stack([build_concept_query(base_text, encoder), build_concept_query(alt_text, encoder)])
    return F.normalize(embeddings.mean(dim=0), dim=0)


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


def load_ground_truth_by_class(records: list[FaithfulnessResult]) -> dict[int, dict[int, float]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in records:
        grouped[(r.concept_number, r.predicted_class)].append(r.delta_p)
    by_class: dict[int, dict[int, float]] = defaultdict(dict)
    for (concept, cls), deltas in grouped.items():
        if len(deltas) >= MIN_SAMPLES:
            by_class[cls][concept] = float(np.mean(deltas))
    return by_class


def top_concept_metrics(gt_by_class: dict[int, dict[int, float]], scores: dict[tuple[int, int], float]) -> dict:
    scores_by_class: dict[int, dict[int, float]] = defaultdict(dict)
    for (concept, cls), score in scores.items():
        scores_by_class[cls][concept] = score

    top1_hits = top3_hits = top5_hits = n_eval = 0
    for cls, gt_concepts in gt_by_class.items():
        if cls not in scores_by_class:
            continue
        common = [c for c in gt_concepts if c in scores_by_class[cls]]
        if len(common) < MIN_CONCEPTS_PER_CLASS:
            continue
        gt_ranked = sorted(common, key=lambda c: gt_concepts[c], reverse=True)
        method_top1 = max(common, key=lambda c: scores_by_class[cls][c])
        n_eval += 1
        if method_top1 == gt_ranked[0]:
            top1_hits += 1
        if method_top1 in set(gt_ranked[:3]):
            top3_hits += 1
        if method_top1 in set(gt_ranked[:5]):
            top5_hits += 1

    if n_eval == 0:
        return {"n": 0, "top1": float("nan"), "top3": float("nan"), "top5": float("nan")}
    return {"n": n_eval, "top1": top1_hits / n_eval, "top3": top3_hits / n_eval, "top5": top5_hits / n_eval}


GT_TOP_N = 3
RECALL_KS = (3, 5, 10)


def recall_metrics(gt_by_class: dict[int, dict[int, float]], scores: dict[tuple[int, int], float]) -> dict:
    """Of the ground truth's own top-3 most important concepts for a
    class, what fraction show up in the method's own top-K ranked list
    (K in RECALL_KS)? Same restriction-to-common-concepts convention as
    top_concept_metrics."""
    scores_by_class: dict[int, dict[int, float]] = defaultdict(dict)
    for (concept, cls), score in scores.items():
        scores_by_class[cls][concept] = score

    recall_sums = {k: 0.0 for k in RECALL_KS}
    n_eval = 0
    for cls, gt_concepts in gt_by_class.items():
        if cls not in scores_by_class:
            continue
        common = [c for c in gt_concepts if c in scores_by_class[cls]]
        if len(common) < GT_TOP_N:
            continue
        gt_ranked = sorted(common, key=lambda c: gt_concepts[c], reverse=True)
        gt_top_set = set(gt_ranked[:GT_TOP_N])
        method_ranked = sorted(common, key=lambda c: scores_by_class[cls][c], reverse=True)
        n_eval += 1
        for k in RECALL_KS:
            hit = len(gt_top_set & set(method_ranked[:k]))
            recall_sums[k] += hit / len(gt_top_set)

    if n_eval == 0:
        return {"n": 0, **{f"recall{k}": float("nan") for k in RECALL_KS}}
    return {"n": n_eval, **{f"recall{k}": recall_sums[k] / n_eval for k in RECALL_KS}}


def run_config(groundable, attribute_names, encoder, spec, pool, native_model, k: int, use_demean: bool,
                phrasing: str, text_center: torch.Tensor | None) -> dict[tuple[int, int], float]:
    scores: dict[tuple[int, int], float] = {}
    for attr_idx, (prefix, _part_names) in groundable.items():
        value = attribute_names[attr_idx].split("::", 1)[1]
        t_c = build_query(prefix, value, encoder, phrasing)
        if use_demean:
            t_c = demean_query(t_c, text_center)

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, k)
        absent_indices = aligned_retrieval(pool, present_indices, t_c, k)

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


def build_pool_for_encoder(encoder_cfg: dict, encoder, image_paths, class_labels, test_ids):
    cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": encoder_cfg, "cache_dir": "embedding_cache"})
    cfg.dataset = {"name": "cub", "root": str(CUB_ROOT)}
    cfg.pool_source = "test"
    pairs = [(image_paths[i], class_labels[i]) for i in test_ids]
    return load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)


def evaluate_and_log(label, records, gt_by_class, scores, results):
    rho_result = score_method_agreement(records, scores, min_samples_per_pair=3)
    sign_result = score_sign_agreement(records, scores, min_samples_per_pair=3)
    topk = top_concept_metrics(gt_by_class, scores)
    recall = recall_metrics(gt_by_class, scores)
    results.append((label, rho_result, sign_result, topk, recall))
    sign_str = f"sign={sign_result.agreement_frac:.1%}" if rho_result is not None else "sign=n/a"
    print(f"[{label}] {sign_str} | top1={topk['top1']:.1%} top3={topk['top3']:.1%} top5={topk['top5']:.1%} | "
          f"recall@3={recall['recall3']:.1%} recall@5={recall['recall5']:.1%} recall@10={recall['recall10']:.1%}",
          flush=True)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    faithfulness_records = load_faithfulness_records()
    gt_by_class = load_ground_truth_by_class(faithfulness_records)
    print(f"{len(faithfulness_records)} attribute-level faithfulness records loaded.", flush=True)
    print(f"{len(gt_by_class)} classes have ground-truth-measured concepts.", flush=True)

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

    results = []

    print("\nLoading SigLIP + pool...", flush=True)
    siglip_cfg = OmegaConf.create({"device": DEVICE, **ENCODER_CONFIGS["siglip"]})
    siglip_encoder = instantiate_encoder(OmegaConf.create({"encoder": siglip_cfg, "device": DEVICE}))
    siglip_pool = build_pool_for_encoder(ENCODER_CONFIGS["siglip"], siglip_encoder, image_paths, class_labels, test_ids)
    siglip_text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, siglip_encoder)

    configs = list(itertools.product([15, 30, 50], [False, True], ["baseline", "ensemble"]))
    for k, use_demean, phrasing in configs:
        label = f"aligned_siglip_k{k}_demean{use_demean}_{phrasing}"
        scores = run_config(groundable, attribute_names, siglip_encoder, spec, siglip_pool, native_model, k,
                             use_demean, phrasing, siglip_text_center)
        evaluate_and_log(label, faithfulness_records, gt_by_class, scores, results)

    print("\nLoading CLIP (ViT-B-32/openai) + pool...", flush=True)
    clip_cfg = OmegaConf.create({"device": DEVICE, **ENCODER_CONFIGS["clip"]})
    clip_encoder = instantiate_encoder(OmegaConf.create({"encoder": clip_cfg, "device": DEVICE}))
    clip_pool = build_pool_for_encoder(ENCODER_CONFIGS["clip"], clip_encoder, image_paths, class_labels, test_ids)
    clip_text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, clip_encoder)

    for use_demean in [False, True]:
        label = f"aligned_clip_k30_demean{use_demean}_baseline"
        scores = run_config(groundable, attribute_names, clip_encoder, spec, clip_pool, native_model, 30,
                             use_demean, "baseline", clip_text_center)
        evaluate_and_log(label, faithfulness_records, gt_by_class, scores, results)

    with open(RESULTS_DIR / "cards_cub_top_concept_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p",
                          "n_classes", "top1", "top3", "top5", "recall3", "recall5", "recall10"])
        for label, rho_result, sign_result, topk, recall in results:
            if rho_result is None:
                writer.writerow([label, "", "", "", "", "", "", topk["n"], topk["top1"], topk["top3"], topk["top5"],
                                  recall["recall3"], recall["recall5"], recall["recall10"]])
            else:
                writer.writerow([label, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p,
                                  topk["n"], topk["top1"], topk["top3"], topk["top5"],
                                  recall["recall3"], recall["recall5"], recall["recall10"]])

    print(f"\nSaved {len(results)} configs to results/cards_cub_top_concept_ablation.csv")
    print("\n=== summary, sorted by top1 match rate descending ===")
    scored = [(label, r, s, t, rc) for label, r, s, t, rc in results if r is not None]
    scored.sort(key=lambda tup: -tup[3]["top1"])
    for label, r, s, t, rc in scored:
        print(f"{label:<50s} sign={s.agreement_frac:.1%}  top1={t['top1']:.1%}  top3={t['top3']:.1%}  top5={t['top5']:.1%}  "
              f"recall@3={rc['recall3']:.1%}  recall@5={rc['recall5']:.1%}  recall@10={rc['recall10']:.1%}")

    print("\n=== summary, sorted by recall@3 descending ===")
    scored_by_recall = sorted(scored, key=lambda tup: -tup[4]["recall3"])
    for label, r, s, t, rc in scored_by_recall:
        print(f"{label:<50s} recall@3={rc['recall3']:.1%}  recall@5={rc['recall5']:.1%}  recall@10={rc['recall10']:.1%}")


if __name__ == "__main__":
    main()
