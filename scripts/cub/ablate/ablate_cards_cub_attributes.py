"""Same ablation grid as ablate_cards_cub.py (K x demean_query x phrasing
ensembling, SigLIP + CLIP), run against the 87-attribute bank directly
rather than the 8-part bank -- per direct pushback that a config found to
help on one bank isn't guaranteed to transfer to the other (already
demonstrated repeatedly this session: CARDS/PCBM's correlations differ,
sometimes in sign, between the two banks). This is the bank that actually
matters for conclusions (literature-comparable, already fully wired with
CARDS/PCBM/TCAV scores), so ablating it directly rather than extrapolating
from the cheaper 8-part bank.

Reuses run_cards_cub_attributes.py's PREFIX_TEMPLATES/
build_attribute_query_text for the baseline phrasing; ALT_PREFIX_TEMPLATES
below is a second, differently-structured phrasing per prefix (e.g.
"a {value} winged bird" vs. "a bird with {value} wings") for the
"ensemble" condition, averaged with the baseline phrasing -- the
attribute-bank analogue of the 8-part ablation's synonym-phrasing
ensemble (full synonym sets for all 87 attributes would be a much larger
hand-authoring effort; a second template per prefix tests the same
underlying question -- does averaging multiple phrasings help -- at a
bounded cost).
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

from cards.data.cub_attributes import groundable_attributes, load_attribute_names  # noqa: E402
from cards.data.cub_parts import load_images_txt  # noqa: E402
from cards.models.backbones import BACKBONES  # noqa: E402
from cards.pipeline import instantiate_encoder  # noqa: E402
from cards.retrieval.confound import matched_retrieval  # noqa: E402
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

ALT_PREFIX_TEMPLATES: dict[str, str] = {
    "has_bill_shape": "a {value}-billed bird",
    "has_bill_length": "a bird whose bill is {value}",
    "has_bill_color": "a {value}-billed bird",
    "has_wing_color": "a {value}-winged bird",
    "has_wing_shape": "a bird that has {value}",
    "has_wing_pattern": "wings with a {value} pattern",
    "has_breast_pattern": "a breast with a {value} pattern",
    "has_breast_color": "a {value}-breasted bird",
    "has_back_color": "a bird with a {value} colored back",
    "has_back_pattern": "a back with a {value} pattern",
    "has_tail_shape": "a bird that has a {value}",
    "has_tail_pattern": "a tail with a {value} pattern",
    "has_upper_tail_color": "a bird with a {value} colored upper tail",
    "has_under_tail_color": "a bird with a {value} colored under tail",
    "has_throat_color": "a {value}-throated bird",
    "has_eye_color": "a {value}-eyed bird",
    "has_forehead_color": "a {value}-foreheaded bird",
    "has_nape_color": "a bird with a {value} colored nape",
    "has_belly_color": "a {value}-bellied bird",
    "has_belly_pattern": "a belly with a {value} pattern",
    "has_leg_color": "a {value}-legged bird",
    "has_crown_color": "a {value}-crowned bird",
}

ENCODER_CONFIGS = {
    "siglip": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
               "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
    "clip": {"name": "clip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
             "model_name": "ViT-B-32", "pretrained": "openai", "device": DEVICE},
}


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

        present_indices, _ = retrieve_top_bottom_k(pool, t_c, k)
        absent_indices = matched_retrieval(pool, present_indices, t_c)

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

    print("\nLoading SigLIP + pool...", flush=True)
    siglip_cfg = OmegaConf.create({"device": DEVICE, **ENCODER_CONFIGS["siglip"]})
    siglip_encoder = instantiate_encoder(OmegaConf.create({"encoder": siglip_cfg, "device": DEVICE}))
    siglip_pool = build_pool_for_encoder(ENCODER_CONFIGS["siglip"], siglip_encoder, image_paths, class_labels, test_ids)
    siglip_text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, siglip_encoder)

    configs = list(itertools.product([15, 30, 50], [False, True], ["baseline", "ensemble"]))
    for k, use_demean, phrasing in configs:
        label = f"siglip_k{k}_demean{use_demean}_{phrasing}"
        scores = run_config(groundable, attribute_names, siglip_encoder, spec, siglip_pool, native_model, k,
                             use_demean, phrasing, siglip_text_center)
        evaluate_and_log(label, faithfulness_records, scores, results)

    print("\nLoading CLIP (ViT-B-32/openai) + pool...", flush=True)
    clip_cfg = OmegaConf.create({"device": DEVICE, **ENCODER_CONFIGS["clip"]})
    clip_encoder = instantiate_encoder(OmegaConf.create({"encoder": clip_cfg, "device": DEVICE}))
    clip_pool = build_pool_for_encoder(ENCODER_CONFIGS["clip"], clip_encoder, image_paths, class_labels, test_ids)
    clip_text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, clip_encoder)

    for use_demean in [False, True]:
        label = f"clip_k30_demean{use_demean}_baseline"
        scores = run_config(groundable, attribute_names, clip_encoder, spec, clip_pool, native_model, 30,
                             use_demean, "baseline", clip_text_center)
        evaluate_and_log(label, faithfulness_records, scores, results)

    with open(RESULTS_DIR / "cards_cub_attribute_ablation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n_pairs", "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for label, rho_result, sign_result in results:
            if rho_result is None:
                writer.writerow([label, "", "", "", "", "", ""])
            else:
                writer.writerow([label, rho_result.n_pairs, rho_result.spearman_rho, rho_result.spearman_p,
                                  sign_result.agreement_frac, sign_result.n_agree, sign_result.binom_p])

    print(f"\nSaved {len(results)} configs to results/cards_cub_attribute_ablation.csv")
    print("\n=== summary, sorted by |spearman_rho| descending ===")
    scored = [(label, r, s) for label, r, s in results if r is not None]
    scored.sort(key=lambda t: -abs(t[1].spearman_rho))
    for label, r, s in scored:
        print(f"{label:<45s} rho={r.spearman_rho:+.4f} (p={r.spearman_p:.4g})  "
              f"sign={s.agreement_frac:.1%} (p={s.binom_p:.4g})")


if __name__ == "__main__":
    main()
