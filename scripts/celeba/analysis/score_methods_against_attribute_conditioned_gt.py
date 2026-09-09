"""Re-scores the masking hybrid (v96 best config), TCAV, and PCBM against
the corrected, attribute-conditioned ground truth
(`celeba_full_faithfulness_attribute_conditioned.csv` -- both label
directions pooled, candidates filtered by the concept's own attribute
value, N_PER_ATTRIBUTE=90, see run_celeba_full_faithfulness_attribute_
conditioned.py's own docstring for the full rationale), prompted
directly ("Yes, please score them against the corrected ground truth
please!").

No method needs recomputing -- every method's own score is already a
per-(concept,class) scalar, independent of which images ground truth
was built from (score_method_agreement/score_sign_agreement's own
documented design). This just re-aggregates the NEW ground truth and
re-correlates the SAME already-computed score tables against it,
printed side by side with the ORIGINAL ground truth's own numbers for
direct comparison.

Also writes `results/celeba_attribute_conditioned_gt_comparison.csv` --
previously this only printed to the terminal (the v113 table in notes/
celeba_correlation_investigation.md was recovered from a screenshot of
that output after the session that produced it got compacted), prompted
directly ("can you check if the code is available to regenerate these
results?" / "Yes please!" -- add a CSV write so this is never
screenshot-dependent again).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")  # needed to unpickle the saved PosthocLinearCBM checkpoint

from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS, TARGET_CLASSES
from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

RESULTS_DIR = Path("results")
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}


def load_records_by_task(csv_name: str) -> dict[str, list[FaithfulnessResult]]:
    by_task: dict[str, list[FaithfulnessResult]] = {t: [] for t in TARGET_CLASSES}
    with open(RESULTS_DIR / csv_name, newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]].append(FaithfulnessResult(
                image=row["image"], concept_number=CONCEPT_TO_IDX[row["concept_name"]], category=row["category"],
                predicted_class=int(row["predicted_class"]), p0=float(row["p0"]), p_masked=float(row["p_masked"]),
                delta_p=float(row["delta_p"]), delta_logit=float(row["delta_logit"]),
                random_delta_p_mean=float(row["random_delta_p_mean"]), random_delta_p_std=float(row["random_delta_p_std"]),
                z_score=float(row["z_score"]), n_random_fallbacks=int(row["n_random_fallbacks"]),
            ))
    return by_task


def load_hybrid_scores() -> dict[str, dict[tuple[int, int], float]]:
    by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_threshold_zscore_orthogonalize_ablation_raw_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            if float(row["k"]) == 1.0:  # this CSV predates the k->alpha rename; real header is still "k"
                by_task[row["target_task"]][(CONCEPT_TO_IDX[row["concept_name"]], 1)] = float(row["hybrid_raw_score"])
    return by_task


def load_tcav_scores() -> dict[str, dict[tuple[int, int], float]]:
    by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "tcav_celeba_full_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]][(CONCEPT_TO_IDX[row["concept_name"]], 1)] = float(row["mean_sign_count"])
    return by_task


def load_tcav_magnitude_scores() -> dict[str, dict[tuple[int, int], float]]:
    """TCAV's `mean_magnitude` column (same CSV as `load_tcav_scores`) --
    the mean magnitude of the directional derivative along the CAV
    direction, averaged across random-concept pairings, unlike
    `mean_sign_count` (a [0,1] fraction of positive-sign derivatives).
    Always >=0 (it's a magnitude), so `method_threshold=0.5` (sign_count's
    own chance-level split) does NOT apply here -- report() is called with
    `skip_sign=True` for this one, since a non-negative score has no
    meaningful "positive vs negative" split to test agreement against."""
    by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "tcav_celeba_full_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]][(CONCEPT_TO_IDX[row["concept_name"]], 1)] = float(row["mean_magnitude"])
    return by_task


def load_hybrid_official_val_scores() -> dict[str, dict[tuple[int, int], float]]:
    by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / "cards_celeba_masking_hybrid_official_val_k1_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]][(CONCEPT_TO_IDX[row["concept_name"]], 1)] = float(row["hybrid_raw_score"])
    return by_task


def load_pcbm_clip_concepts_scores(csv_name: str) -> dict[str, dict[tuple[int, int], float]]:
    """The "CLIP concepts" PCBM variant (Yuksekgonul et al. 2023, no image
    concept dataset needed -- concept vectors come directly from CLIP/
    SigLIP text embeddings of the concept names) -- v111 in notes/
    celeba_correlation_investigation.md, two backbones."""
    by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / csv_name, newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]][(CONCEPT_TO_IDX[row["concept_name"]], 1)] = float(row["weight"])
    return by_task


def load_pcbm_scores(ckpt_dir: str = "trained_models_new/celeba_full/celeba_attractive_young",
                      ckpt_prefix: str = "pcbm_celeba_full__celeba_attractive_young") -> dict[str, dict[tuple[int, int], float]]:
    by_task: dict[str, dict[tuple[int, int], float]] = {}
    for task_name in TARGET_CLASSES:
        ckpt_path = Path(ckpt_dir) / f"{ckpt_prefix}__{task_name.lower()}__surrogate__seed_42__linear.ckpt"
        posthoc_layer = torch.load(ckpt_path, weights_only=False)
        weight = posthoc_layer.classifier.weight.detach().cpu().numpy()  # (1, n_concepts) binary-task row
        names = posthoc_layer.names
        by_task[task_name] = {(CONCEPT_TO_IDX[name], 1): float(weight[0, i]) for i, name in enumerate(names)}
    return by_task


<<<<<<< Updated upstream
def _stats_for(records, scores, method_threshold):
    """-> dict[task_name] = (rho_r, sign_r), either possibly None if too few pairs."""
    out = {}
    for task_name in TARGET_CLASSES:
        rho_r = score_method_agreement(records[task_name], scores[task_name])
        sign_r = score_sign_agreement(records[task_name], scores[task_name], method_threshold=method_threshold)
        out[task_name] = (rho_r, sign_r)
    return out


def report(method: str, pool: str, old_records, new_records, scores_by_task, rows: list, method_threshold: float = 0.0):
    """Prints BOTH ground-truth versions side by side (same terminal
    output shape as before) AND appends one row per task to `rows` for
    the CSV write -- computed together per call instead of two separate
    top-level report() blocks, so old/new never have to be re-matched
    afterward."""
    print(f"\n=== {method}, {pool} ===", flush=True)
    old_stats = _stats_for(old_records, scores_by_task, method_threshold)
    new_stats = _stats_for(new_records, scores_by_task, method_threshold)
    for task_name in TARGET_CLASSES:
        rho_old, sign_old = old_stats[task_name]
        rho_new, sign_new = new_stats[task_name]
        if rho_old is None or rho_new is None:
            print(f"  [{task_name}] too few pairs")
            continue
        print(f"  [{task_name}] orig:     n={rho_old.n_pairs} rho={rho_old.spearman_rho:+.4f} (p={rho_old.spearman_p:.4g})  "
              f"sign={sign_old.agreement_frac:.1%} ({sign_old.n_agree}/{sign_old.n_pairs}, p={sign_old.binom_p:.4g})", flush=True)
        print(f"  [{task_name}] corrected: n={rho_new.n_pairs} rho={rho_new.spearman_rho:+.4f} (p={rho_new.spearman_p:.4g})  "
              f"sign={sign_new.agreement_frac:.1%} ({sign_new.n_agree}/{sign_new.n_pairs}, p={sign_new.binom_p:.4g})", flush=True)
        rows.append({
            "method": method, "pool": pool, "task": task_name,
            "n_orig": rho_old.n_pairs, "rho_orig": rho_old.spearman_rho, "rho_p_orig": rho_old.spearman_p,
            "sign_frac_orig": sign_old.agreement_frac, "sign_p_orig": sign_old.binom_p,
            "n_corrected": rho_new.n_pairs, "rho_corrected": rho_new.spearman_rho, "rho_p_corrected": rho_new.spearman_p,
            "sign_frac_corrected": sign_new.agreement_frac, "sign_p_corrected": sign_new.binom_p,
        })
=======
def load_pcbm_clip_concepts_scores(backbone_name: str) -> dict[str, dict[tuple[int, int], float]]:
    """PCBM's own "CLIP concepts" variant (Yuksekgonul et al. 2023) -- no
    CAV fitting, concept vectors are the encoder's own text embeddings.
    `backbone_name` is "clip_rn50" (standard CLIP ResNet-50) or "siglip",
    reading `results/pcbm_clip_concepts_celeba_<backbone_name>_scores.csv`
    (written by `train_pcbm_clip_concepts_celeba_full.py`)."""
    by_task: dict[str, dict[tuple[int, int], float]] = {t: {} for t in TARGET_CLASSES}
    with open(RESULTS_DIR / f"pcbm_clip_concepts_celeba_{backbone_name}_scores.csv", newline="") as f:
        for row in csv.DictReader(f):
            by_task[row["target_task"]][(CONCEPT_TO_IDX[row["concept_name"]], 1)] = float(row["weight"])
    return by_task


def report(label: str, records_by_task, scores_by_task, method_threshold: float = 0.0, skip_sign: bool = False):
    print(f"\n=== {label} ===", flush=True)
    for task_name in TARGET_CLASSES:
        rho_r = score_method_agreement(records_by_task[task_name], scores_by_task[task_name])
        if rho_r is None:
            print(f"  [{task_name}] too few pairs")
            continue
        line = (f"  [{task_name}] n={rho_r.n_pairs} rho={rho_r.spearman_rho:+.4f} (p={rho_r.spearman_p:.4g})  "
                f"pearson_r={rho_r.pearson_r:+.4f} (p={rho_r.pearson_p:.4g})")
        if not skip_sign:
            sign_r = score_sign_agreement(records_by_task[task_name], scores_by_task[task_name], method_threshold=method_threshold)
            line += f"  sign={sign_r.agreement_frac:.1%} ({sign_r.n_agree}/{sign_r.n_pairs}, p={sign_r.binom_p:.4g})"
        print(line, flush=True)
>>>>>>> Stashed changes


def main():
    old_records = load_records_by_task("celeba_full_faithfulness.csv")
    new_records = load_records_by_task("celeba_full_faithfulness_attribute_conditioned.csv")

    hybrid_scores = load_hybrid_scores()
    hybrid_official_val_scores = load_hybrid_official_val_scores()
    tcav_scores = load_tcav_scores()
    tcav_magnitude_scores = load_tcav_magnitude_scores()
    pcbm_scores = load_pcbm_scores()
<<<<<<< Updated upstream
    pcbm_whole_image_scores = load_pcbm_scores(
        ckpt_dir="trained_models_new/celeba_full_whole_image/celeba_attractive_young",
        ckpt_prefix="pcbm_celeba_full_whole_image__celeba_attractive_young",
    )
    pcbm_clip_rn50_scores = load_pcbm_clip_concepts_scores("pcbm_clip_concepts_celeba_clip_rn50_scores.csv")
    pcbm_siglip_scores = load_pcbm_clip_concepts_scores("pcbm_clip_concepts_celeba_siglip_scores.csv")

    rows: list[dict] = []
    report("Masking hybrid (alpha=1.0, orth=True, K=50, SigLIP)", "HQ-val", old_records, new_records, hybrid_scores, rows)
    report("Masking hybrid (alpha=1.0, orth=True, K=50, SigLIP)", "Official-val", old_records, new_records, hybrid_official_val_scores, rows)
    report("TCAV (mean_sign_count)", "-", old_records, new_records, tcav_scores, rows, method_threshold=0.5)
    report("PCBM (CAV-based, region-crop bank)", "-", old_records, new_records, pcbm_scores, rows)
    report("PCBM (CAV-based, whole-image bank)", "-", old_records, new_records, pcbm_whole_image_scores, rows)
    report("PCBM CLIP-concepts (CLIP RN50 backbone)", "-", old_records, new_records, pcbm_clip_rn50_scores, rows)
    report("PCBM CLIP-concepts (SigLIP backbone)", "-", old_records, new_records, pcbm_siglip_scores, rows)

    out_path = RESULTS_DIR / "celeba_attribute_conditioned_gt_comparison.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "pool", "task", "n_orig", "rho_orig", "rho_p_orig", "sign_frac_orig", "sign_p_orig",
            "n_corrected", "rho_corrected", "rho_p_corrected", "sign_frac_corrected", "sign_p_corrected",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out_path}", flush=True)
=======
    pcbm_clip_rn50_scores = load_pcbm_clip_concepts_scores("clip_rn50")
    pcbm_siglip_scores = load_pcbm_clip_concepts_scores("siglip")

    print("############## ORIGINAL ground truth (region-filtered only, target-positive only) ##############")
    report("Masking hybrid, HQ-val (alpha=1.0, orth=True, K=50, SigLIP)", old_records, hybrid_scores)
    report("Masking hybrid, OFFICIAL-val (same config)", old_records, hybrid_official_val_scores)
    report("TCAV (mean_sign_count)", old_records, tcav_scores, method_threshold=0.5)
    report("TCAV (mean_magnitude)", old_records, tcav_magnitude_scores, skip_sign=True)
    report("PCBM (CAV-based, region-crop bank)", old_records, pcbm_scores)
    report("PCBM CLIP-concepts (CLIP RN50)", old_records, pcbm_clip_rn50_scores)
    report("PCBM CLIP-concepts (SigLIP)", old_records, pcbm_siglip_scores)

    print("\n############## CORRECTED ground truth (attribute-conditioned, both label directions) ##############")
    report("Masking hybrid, HQ-val (alpha=1.0, orth=True, K=50, SigLIP)", new_records, hybrid_scores)
    report("Masking hybrid, OFFICIAL-val (same config)", new_records, hybrid_official_val_scores)
    report("TCAV (mean_sign_count)", new_records, tcav_scores, method_threshold=0.5)
    report("TCAV (mean_magnitude)", new_records, tcav_magnitude_scores, skip_sign=True)
    report("PCBM (CAV-based, region-crop bank)", new_records, pcbm_scores)
    report("PCBM CLIP-concepts (CLIP RN50)", new_records, pcbm_clip_rn50_scores)
    report("PCBM CLIP-concepts (SigLIP)", new_records, pcbm_siglip_scores)
>>>>>>> Stashed changes


if __name__ == "__main__":
    main()
