"""v44's full demean_query x phrasing-ensembling x encoder grid, re-run
with `stratified_retrieval` (cards.retrieval.confound) in place of
`matched_retrieval` -- the other half of "Run the whole sweep on both
aligned and stratified?"

K sweep range differs from the other grids: `stratified_retrieval` needs
`2*k <= min class test-split size` across ALL 200 CUB species
simultaneously (each species gets its own independent top-k/bottom-k
retrieval) -- checked directly, min=11 images/class on the test split, so
k=5 is the largest value that fits every stratum.

Cut down from the original full [2,3,4,5] k-sweep after the first k=2
config alone took ~12 minutes (200-stratum-per-concept retrieval has much
higher per-config Python-loop overhead than the vectorized naive/matched/
aligned strategies -- a full 18-config grid at that rate would run
2-3 hours, not a reasonable wait). Fixed at k=5 (the value that already
showed significance in v46's original strategy comparison) x demean x
phrasing on SigLIP (4 configs) + CLIP at k=5 x demean (2 configs) = 6
configs total.
"""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ablate"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_cub_attributes import PREFIX_TEMPLATES  # noqa: E402
from ablate_cards_cub_attributes import ALT_PREFIX_TEMPLATES, ENCODER_CONFIGS  # noqa: E402

from cards.data.cub_attributes import groundable_attributes, load_attribute_names  # noqa: E402
from cards.data.cub_parts import load_images_txt  # noqa: E402
from cards.models.backbones import BACKBONES  # noqa: E402
from cards.pipeline import instantiate_encoder  # noqa: E402
from cards.retrieval.confound import stratified_retrieval  # noqa: E402
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool  # noqa: E402
from cards.retrieval.pool import CandidatePool  # noqa: E402
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


def run_config(groundable, attribute_names, encoder, spec, pool, native_model, k: int, use_demean: bool,
                phrasing: str, text_center: torch.Tensor | None) -> dict[tuple[int, int], float]:
    scores: dict[tuple[int, int], float] = {}
    for attr_idx, (prefix, _part_names) in groundable.items():
        value = attribute_names[attr_idx].split("::", 1)[1]
        t_c = build_query(prefix, value, encoder, phrasing)
        if use_demean:
            t_c = demean_query(t_c, text_center)

        present_indices, absent_indices = stratified_retrieval(pool, t_c, k)

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


def build_labeled_pool_for_encoder(encoder_cfg: dict, encoder, image_paths, class_labels, test_ids):
    cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": encoder_cfg, "cache_dir": "embedding_cache"})
    cfg.dataset = {"name": "cub", "root": str(CUB_ROOT)}
    cfg.pool_source = "test"
    pairs = [(image_paths[i], class_labels[i]) for i in test_ids]
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    return CandidatePool(paths=pool.paths, embeddings=pool.embeddings, labels=[label for _, label in pairs])


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

    spec = BACKBONES["resnet18_cub"]
    native_model = spec.load_native().to(DEVICE).eval()

    results = []

    print("\nLoading SigLIP + labeled pool...", flush=True)
    siglip_cfg = OmegaConf.create({"device": DEVICE, **ENCODER_CONFIGS["siglip"]})
    siglip_encoder = instantiate_encoder(OmegaConf.create({"encoder": siglip_cfg, "device": DEVICE}))
    siglip_pool = build_labeled_pool_for_encoder(ENCODER_CONFIGS["siglip"], siglip_encoder, image_paths, class_labels, test_ids)
    siglip_text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, siglip_encoder)

    configs = list(itertools.product([5], [False, True], ["baseline", "ensemble"]))
    for k, use_demean, phrasing in configs:
        label = f"stratified_siglip_k{k}_demean{use_demean}_{phrasing}"
        scores = run_config(groundable, attribute_names, siglip_encoder, spec, siglip_pool, native_model, k,
                             use_demean, phrasing, siglip_text_center)
        evaluate_and_log(label, faithfulness_records, scores, results)

    print("\nLoading CLIP (ViT-B-32/openai) + labeled pool...", flush=True)
    clip_cfg = OmegaConf.create({"device": DEVICE, **ENCODER_CONFIGS["clip"]})
    clip_encoder = instantiate_encoder(OmegaConf.create({"encoder": clip_cfg, "device": DEVICE}))
    clip_pool = build_labeled_pool_for_encoder(ENCODER_CONFIGS["clip"], clip_encoder, image_paths, class_labels, test_ids)
    clip_text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, clip_encoder)

    for use_demean in [False, True]:
        label = f"stratified_clip_k5_demean{use_demean}_baseline"
        scores = run_config(groundable, attribute_names, clip_encoder, spec, clip_pool, native_model, 5,
                             use_demean, "baseline", clip_text_center)
        evaluate_and_log(label, faithfulness_records, scores, results)

    with open(RESULTS_DIR / "cards_cub_attribute_ablation_stratified.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for label, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([label, "", "", "", "", "", ""])
            else:
                writer.writerow([label, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])

    print(f"\nSaved {len(results)} configs to results/cards_cub_attribute_ablation_stratified.csv")
    print("\n=== summary, sorted by sign agreement descending ===")
    scored = [(label, r, s) for label, r, s in results if r is not None]
    scored.sort(key=lambda t: -t[2].agreement_frac)
    for label, r, s in scored:
        print(f"{label:<50s} rho={r.spearman_rho:+.4f} (p={r.spearman_p:.4g})  "
              f"sign={s.agreement_frac:.1%} (p={s.binom_p:.4g})")


if __name__ == "__main__":
    main()
