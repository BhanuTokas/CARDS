"""CUB-track analogue of score_all_methods_against_faithfulness.py.

Two separate comparisons, since they use different concept banks and the
ground truth built for each is scoped differently:

1. **8-part ground truth** (results/cub_faithfulness.csv, v36 -- real
   CUB70 masks + silhouette-scaled keypoint-patch approximation) vs. the
   three methods actually run against the new part-crop concept bank:
   CARDS (results/cards_cub_scores.csv), TCAV (results/
   tcav_cub_matching_pairs.csv, a 14-pair validation slice, not the full
   matrix), and PCBM's own surrogate-fit classifier weight
   (trained_models_new/cub_parts/resnet18_cub/...ckpt).

2. **87-attribute ground truth** (results/cub_attribute_faithfulness.csv,
   v38) vs. PCBM's OFFICIAL 112-concept classifier weight
   (post_hoc_cbm's own pre-existing checkpoint, fit against ground-truth
   labels per the original paper's own setup, not our surrogate) --
   reusable for free since both use the identical 0-111 attribute
   indexing (new_attributes.txt's own line order). CARDS/TCAV were never
   run against the 87-attribute bank (would need new text-query/image-
   crop concept exemplars per attribute, e.g. "a brown wing" or crops of
   specifically brown wings) -- out of scope here, noted plainly rather
   than silently skipped.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

import torch

from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

RESULTS_DIR = Path("results")
CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")

PART_NAME_TO_ID = {
    "back": 1, "beak": 2, "belly": 3, "breast": 4, "crown": 5, "forehead": 6,
    "left_eye": 7, "left_leg": 8, "left_wing": 9, "nape": 10, "right_eye": 11,
    "right_leg": 12, "right_wing": 13, "tail": 14, "throat": 15,
}


def load_species_to_idx() -> dict[str, int]:
    result = {}
    for line in (CUB_ROOT / "classes.txt").read_text().splitlines():
        class_id, raw_name = line.split(maxsplit=1)
        name = raw_name.split(".", 1)[1] if "." in raw_name else raw_name
        result[name] = int(class_id) - 1
    return result


def load_part_faithfulness_records() -> list[FaithfulnessResult]:
    records = []
    with open(RESULTS_DIR / "cub_faithfulness.csv", newline="") as f:
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


def load_attribute_faithfulness_records() -> list[FaithfulnessResult]:
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


def load_cards_scores() -> dict[tuple[int, int], float]:
    scores = {}
    with open(RESULTS_DIR / "cards_cub_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            part_id = PART_NAME_TO_ID[row["part"]]
            scores[(part_id, int(row["native_class_idx"]))] = float(row["raw_score"])
    return scores


def load_tcav_scores(species_to_idx: dict[str, int]) -> dict[tuple[int, int], float]:
    scores = {}
    with open(RESULTS_DIR / "tcav_cub_matching_pairs.csv", newline="") as f:
        for row in csv.DictReader(f):
            part_id = PART_NAME_TO_ID[row["part"]]
            class_idx = species_to_idx[row["species"]]
            scores[(part_id, class_idx)] = float(row["mean_sign_count"])
    return scores


def load_pcbm_part_scores() -> dict[tuple[int, int], float]:
    ckpt_path = "trained_models_new/cub_parts/resnet18_cub/pcbm_cub__resnet18_cub__cub_parts__surrogate__seed_42__linear.ckpt"
    model = torch.load(ckpt_path, weights_only=False)
    weight = model.classifier.weight.detach().cpu().numpy()  # (200, 8)
    concept_col = {name: i for i, name in enumerate(model.names)}  # part name -> column
    scores = {}
    for part_name, col in concept_col.items():
        part_id = PART_NAME_TO_ID[part_name]
        for local_idx in range(weight.shape[0]):
            scores[(part_id, local_idx)] = float(weight[local_idx, col])
    return scores


def load_cards_official_scores() -> dict[tuple[int, int], float]:
    """Reads results/cards_cub_attribute_scores.csv (scripts/
    run_cards_cub_attributes.py) -- same 87-attribute bank, natural-
    language text queries, no image exemplars needed at all."""
    scores = {}
    with open(RESULTS_DIR / "cards_cub_attribute_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            scores[(int(row["attribute_index"]), int(row["native_class_idx"]))] = float(row["raw_score"])
    return scores


def load_tcav_official_scores() -> dict[tuple[int, int], float]:
    """Reads results/tcav_cub_attribute_scores.csv (scripts/
    run_tcav_cub_official_attributes.py) -- TCAV run against the SAME
    whole-image, species-level concept definition PCBM's official
    112-attribute bank uses (no new crops needed, see notes v40)."""
    scores = {}
    with open(RESULTS_DIR / "tcav_cub_attribute_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            scores[(int(row["attribute_index"]), int(row["native_class_idx"]))] = float(row["mean_sign_count"])
    return scores


def load_tcav_targeted_scores() -> dict[tuple[int, int], float]:
    """Reads results/tcav_cub_targeted_v49.csv (scripts/
    run_tcav_cub_targeted_v49.py) -- TCAV run specifically against all 522
    (attribute, predicted_class) pairs the v49 stratified ground truth
    supports (full targeted coverage, not a coincidental overlap of a
    broader independently-scoped run -- see notes v52/v55 for why that
    distinction matters)."""
    scores = {}
    with open(RESULTS_DIR / "tcav_cub_targeted_v49.csv", newline="") as f:
        for row in csv.DictReader(f):
            scores[(int(row["attribute_index"]), int(row["native_class_idx"]))] = float(row["mean_sign_count"])
    return scores


def load_pcbm_clip_concepts_scores() -> dict[tuple[int, int], float]:
    """Reads results/pcbm_clip_concepts_cub_scores.csv (scripts/
    train_pcbm_clip_concepts_cub.py, v2) -- PCBM's own "CLIP concepts"
    variant, concept vectors = pure CLIP text embeddings, no CAV/image-
    dataset fitting at all. Best lam=1e-5 from the sweep (test
    fidelity=15.41%, 69.7% nonzero weights) -- far lower fidelity than the
    CAV-based official bank, noted plainly, not silently assumed
    comparable."""
    scores = {}
    with open(RESULTS_DIR / "pcbm_clip_concepts_cub_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            scores[(int(row["attribute_index"]), int(row["native_class_idx"]))] = float(row["weight"])
    return scores


def load_pcbm_official_siglip_scores(groundtruth: bool = False) -> dict[tuple[int, int], float]:
    """Reads results/pcbm_official_siglip_cub_scores.csv (surrogate
    framing) or pcbm_official_siglip_groundtruth_cub_scores.csv
    (ground-truth framing) -- scripts/
    train_pcbm_surrogate_cub_official_siglip.py's dual-framing output.
    Same official 112-attribute concept definition/image splits as the
    resnet18_cub bank, CAVs refit in SigLIP's own embedding space."""
    fname = "pcbm_official_siglip_groundtruth_cub_scores.csv" if groundtruth else "pcbm_official_siglip_cub_scores.csv"
    scores = {}
    with open(RESULTS_DIR / fname, newline="") as f:
        for row in csv.DictReader(f):
            scores[(int(row["attribute_index"]), int(row["native_class_idx"]))] = float(row["weight"])
    return scores


def load_pcbm_siglip_concepts_scores() -> dict[tuple[int, int], float]:
    """Reads results/pcbm_siglip_concepts_cub_scores.csv (scripts/
    train_pcbm_siglip_concepts_cub.py) -- PCBM's "concepts" variant with
    SigLIP swapped in for CLIP-RN50 as the joint image+text backbone."""
    scores = {}
    with open(RESULTS_DIR / "pcbm_siglip_concepts_cub_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            scores[(int(row["attribute_index"]), int(row["native_class_idx"]))] = float(row["weight"])
    return scores


def load_pcbm_official_scores() -> dict[tuple[int, int], float]:
    ckpt_path = (
        "../post_hoc_cbm/trained_models_new/cub/resnet18_cub/"
        "pcbm_cub__resnet18_cub__cub_resnet18_cub_0__lam_0.0002__alpha_0.99__seed_42__linear.ckpt"
    )
    model = torch.load(ckpt_path, weights_only=False)
    weight = model.classifier.weight.detach().cpu().numpy()  # (200, 112)
    scores = {}
    for attr_idx in range(weight.shape[1]):
        for local_idx in range(weight.shape[0]):
            scores[(attr_idx, local_idx)] = float(weight[local_idx, attr_idx])
    return scores


def report(method_name: str, records: list[FaithfulnessResult], scores: dict[tuple[int, int], float],
           pair_label: str, method_threshold: float = 0.0, indent: str = "") -> None:
    """Prints Spearman rho (exact ranking, fragile at small n) alongside
    sign agreement (coarser, just "did they point the same way," more
    robust at the small n_pairs this investigation typically has)."""
    rho_result = score_method_agreement(records, scores, min_samples_per_pair=3)
    sign_result = score_sign_agreement(records, scores, min_samples_per_pair=3, method_threshold=method_threshold)

    if rho_result is None:
        print(f"{indent}{method_name}: too few overlapping ({pair_label}, predicted_class) pairs "
              f"with >=3 faithfulness samples each for a meaningful comparison.")
        return

    print(f"{indent}{method_name}: n_pairs={rho_result.n_pairs}, "
          f"Spearman rho={rho_result.spearman_rho:.4f}, p={rho_result.spearman_p:.4g} | "
          f"sign agreement={sign_result.agreement_frac:.1%} ({sign_result.n_agree}/{sign_result.n_pairs}), "
          f"binom p={sign_result.binom_p:.4g}")


def main():
    species_to_idx = load_species_to_idx()

    print("=" * 70)
    print("PART 1: 8-part ground truth vs. CARDS / TCAV / PCBM (part-crop bank)")
    print("=" * 70)
    part_records = load_part_faithfulness_records()
    print(f"{len(part_records)} part-level faithfulness records loaded.")

    cards_scores = load_cards_scores()
    tcav_scores = load_tcav_scores(species_to_idx)
    pcbm_scores = load_pcbm_part_scores()
    print(f"CARDS scores: {len(cards_scores)} pairs | TCAV scores: {len(tcav_scores)} pairs "
          f"(validation slice only) | PCBM scores: {len(pcbm_scores)} pairs")

    for method_name, scores, threshold in [
        ("CARDS", cards_scores, 0.0),
        ("TCAV (mean_sign_count)", tcav_scores, 0.5),
        ("PCBM (part-crop surrogate weight)", pcbm_scores, 0.0),
    ]:
        report(method_name, part_records, scores, pair_label="part", method_threshold=threshold, indent="\n")

    print("\n" + "=" * 70)
    print("PART 2: 87-attribute ground truth vs. CARDS / PCBM (OFFICIAL bank) / TCAV")
    print("(all three now scored against the same 87-attribute definition)")
    print("=" * 70)
    attr_records = load_attribute_faithfulness_records()
    print(f"{len(attr_records)} attribute-level faithfulness records loaded.")

    cards_official_scores = load_cards_official_scores()
    pcbm_official_scores = load_pcbm_official_scores()
    tcav_official_scores = load_tcav_official_scores()
    tcav_targeted_scores = load_tcav_targeted_scores()
    pcbm_clip_concepts_scores = load_pcbm_clip_concepts_scores()
    print(f"CARDS (official bank) scores: {len(cards_official_scores)} pairs")
    print(f"PCBM (official bank) scores: {len(pcbm_official_scores)} pairs")
    print(f"TCAV (official bank, broad/partial-coverage run) scores: {len(tcav_official_scores)} pairs")
    print(f"TCAV (targeted v49, full-coverage run) scores: {len(tcav_targeted_scores)} pairs")
    print(f"PCBM (CLIP-concepts) scores: {len(pcbm_clip_concepts_scores)} pairs")

    pcbm_official_siglip_surrogate = load_pcbm_official_siglip_scores(groundtruth=False)
    pcbm_official_siglip_groundtruth = load_pcbm_official_siglip_scores(groundtruth=True)
    pcbm_siglip_concepts_scores = load_pcbm_siglip_concepts_scores() if (RESULTS_DIR / "pcbm_siglip_concepts_cub_scores.csv").exists() else {}

    for method_name, scores, threshold in [
        ("CARDS (official bank)", cards_official_scores, 0.0),
        ("PCBM (official 112-bank, resnet18_cub backbone, ground-truth)", pcbm_official_scores, 0.0),
        ("PCBM (official 112-bank, SigLIP backbone, surrogate)", pcbm_official_siglip_surrogate, 0.0),
        ("PCBM (official 112-bank, SigLIP backbone, ground-truth)", pcbm_official_siglip_groundtruth, 0.0),
        ("TCAV (broad run, STALE PARTIAL COVERAGE -- see v55)", tcav_official_scores, 0.5),
        ("TCAV (targeted v49, FULL COVERAGE -- the trustworthy number)", tcav_targeted_scores, 0.5),
        ("PCBM (CLIP-RN50-concepts weight)", pcbm_clip_concepts_scores, 0.0),
        ("PCBM (SigLIP-concepts weight)", pcbm_siglip_concepts_scores, 0.0),
    ]:
        report(method_name, attr_records, scores, pair_label="attribute", method_threshold=threshold, indent="\n")

        # calibrated-only vs. heuristic-only subsets, since v38 flagged
        # heuristic-area records as less trustworthy.
        for tag in ("calibrated", "heuristic"):
            subset = [r for r in attr_records if r.category == tag]
            report(f"[{tag} subset, n={len(subset)}]", subset, scores, pair_label="attribute",
                   method_threshold=threshold, indent="  ")


if __name__ == "__main__":
    main()
