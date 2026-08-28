"""Connects the Phase 3 faithfulness ground truth (results/
broden_faithfulness_records.csv) to CARDS' raw_score (results/
cards_imagenet_scores.csv, scripts/run_cards_imagenet.py) and PCBM's own
fitted surrogate weight -- the two methods that hadn't been scored
against it yet (TCAV was already validated on its own 6-pair slice in
Phase 2; not re-run here, see the note printed at the end).

Uses cards.validation.broden_faithfulness.score_method_agreement
directly, once per method, on the identical faithfulness_records list --
same ground truth, different method_scores input, per the metric's own
design.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

import torch  # noqa: E402

from build_imagenet_slice import TARGET_CLASSES  # noqa: E402
from cards.validation.broden_faithfulness import FaithfulnessResult, score_method_agreement  # noqa: E402

RESULTS_DIR = Path("results")
CONCEPT_TO_LABEL = {"car": 38, "cat": 105, "dog": 93, "chair": 36, "bottle": 70}  # same as Phase 3's driver
NATIVE_LABEL_IDX = {name: idx for idx, _, name in TARGET_CLASSES}  # class_name -> native 1000-way index


def load_faithfulness_records(suffix: str = "_targeted_strict") -> list[FaithfulnessResult]:
    """Defaults to v30's strict-targeted records (each concept restricted
    to its own genuinely-matching class(es), e.g. bottle->water_bottle
    only) -- the properly-scoped ground truth. The plain volume/loose-
    targeted files exist too but mixing them back in would reintroduce
    the dilution v29/v30 diagnosed, so this stays strict-only by
    default."""
    records = []
    with open(RESULTS_DIR / f"broden_faithfulness_records{suffix}.csv", newline="") as f:
        for row in csv.DictReader(f):
            concept_name = row["concept"]
            records.append(
                FaithfulnessResult(
                    image=row["image"],
                    concept_number=CONCEPT_TO_LABEL[concept_name],
                    category="object",
                    predicted_class=int(row["predicted_class"]),
                    p0=float(row["p0"]),
                    p_masked=float(row["p_masked"]),
                    delta_p=float(row["delta_p"]),
                    delta_logit=float(row["delta_logit"]),
                    random_delta_p_mean=float(row["random_delta_p_mean"]),
                    random_delta_p_std=float(row["random_delta_p_std"]),
                    z_score=float(row["z_score"]),
                    n_random_fallbacks=int(row["n_random_fallbacks"]),
                )
            )
    return records


def load_cards_scores() -> dict[tuple[int, int], float]:
    scores = {}
    with open(RESULTS_DIR / "cards_imagenet_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            label_number = CONCEPT_TO_LABEL[row["broden_concept"]]
            scores[(label_number, int(row["native_class_idx"]))] = float(row["raw_score"])
    return scores


def load_pcbm_scores() -> dict[tuple[int, int], float]:
    ckpt_path = (
        "trained_models_new/imagenet_slice_baseline/resnet18_torchvision/"
        "pcbm_imagenet_slice__resnet18_torchvision__broden_baseline__surrogate__seed_42__linear.ckpt"
    )
    model = torch.load(ckpt_path, weights_only=False)
    weight = model.classifier.weight.detach().cpu().numpy()  # (25, 143)
    concept_col = {name: i for i, name in enumerate(model.names)}

    scores = {}
    for concept_name, label_number in CONCEPT_TO_LABEL.items():
        if concept_name not in concept_col:
            print(f"WARNING: {concept_name!r} not in PCBM's own concept bank names -- skipping")
            continue
        col = concept_col[concept_name]
        for local_idx, class_name in model.idx_to_class.items():
            native_idx = NATIVE_LABEL_IDX[class_name]
            scores[(label_number, native_idx)] = float(weight[local_idx, col])
    return scores


def load_tcav_scores() -> dict[tuple[int, int], float]:
    """Reads results/tcav_matching_pairs.csv (scripts/run_tcav.py) --
    mean_sign_count is the primary comparable scalar, per the original
    plan's own reasoning (captum's own headline TCAV score, [0,1]-ranged,
    most directly analogous to "does this concept increase this class'
    logit"). Includes every row in the file, not just ones labeled
    "matching" -- the 4 pairs originally validated as "positive" in Phase
    2 (e.g. car->sports_car) are equally valid entries for this table,
    just computed earlier under a different label."""
    scores = {}
    with open(RESULTS_DIR / "tcav_matching_pairs.csv", newline="") as f:
        for row in csv.DictReader(f):
            label_number = CONCEPT_TO_LABEL[row["broden_concept"]]
            if row["target_class"] not in NATIVE_LABEL_IDX:
                continue
            native_idx = NATIVE_LABEL_IDX[row["target_class"]]
            scores[(label_number, native_idx)] = float(row["mean_sign_count"])
    return scores


def main():
    faithfulness_records = load_faithfulness_records()
    print(f"{len(faithfulness_records)} faithfulness records loaded.")
    predicted_classes_hit = sorted({r.predicted_class for r in faithfulness_records})
    print(f"{len(predicted_classes_hit)} distinct native-model predicted classes appear across all faithfulness images.")
    in_scope = [c for c in predicted_classes_hit if c in NATIVE_LABEL_IDX.values()]
    print(f"Of those, {len(in_scope)} fall within our 25 target classes: {in_scope}")

    cards_scores = load_cards_scores()
    pcbm_scores = load_pcbm_scores()
    tcav_scores = load_tcav_scores()
    print(f"\nCARDS scores available for {len(cards_scores)} (concept, class) pairs.")
    print(f"PCBM scores available for {len(pcbm_scores)} (concept, class) pairs.")
    print(f"TCAV scores available for {len(tcav_scores)} (concept, class) pairs.")

    for method_name, scores in [
        ("CARDS", cards_scores),
        ("PCBM (surrogate weight)", pcbm_scores),
        ("TCAV (mean_sign_count)", tcav_scores),
    ]:
        result = score_method_agreement(faithfulness_records, scores, min_samples_per_pair=3)
        if result is None:
            print(f"\n{method_name}: too few overlapping (concept, predicted_class) pairs "
                  f"(need >=3 with >=3 faithfulness samples each) for a meaningful correlation.")
        else:
            print(f"\n{method_name}: n_pairs={result.n_pairs}, "
                  f"Spearman rho={result.spearman_rho:.4f}, p={result.spearman_p:.4g}")


if __name__ == "__main__":
    main()
