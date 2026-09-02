"""CUB analogue of score_tcav_magnitude_celeba_full.py, prompted directly
("for TCAV ... it also calculates the gradient. Should we compare the
gradient for rho?"). Rescores TCAV using its own already-saved
`mean_magnitude` (the raw signed mean directional derivative, a side
effect of the SAME captum TCAV.interpret() call that already computed
sign_count -- no new gradient computation needed) instead of
`mean_sign_count` for the Spearman rho comparison. Uses `tcav_cub_
targeted_v49.csv`, the exact file behind the v62 headline TCAV numbers
(rho=-0.015 n.s., sign=61.7% p=1.7e-44) -- CUB's own rho/sign divergence
is the most striking case for this check in either track.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cards.validation.broden_faithfulness import (
    FaithfulnessResult,
    score_method_agreement,
    score_sign_agreement,
)

RESULTS_DIR = Path("results")


def load_records() -> list[FaithfulnessResult]:
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


def main():
    records = load_records()
    print(f"{len(records)} faithfulness records loaded.", flush=True)

    sign_count_scores: dict[tuple[int, int], float] = {}
    magnitude_scores: dict[tuple[int, int], float] = {}
    with open(RESULTS_DIR / "tcav_cub_targeted_v49.csv", newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["attribute_index"]), int(row["native_class_idx"]))
            sign_count_scores[key] = float(row["mean_sign_count"])
            magnitude_scores[key] = float(row["mean_magnitude"])

    rho_sc = score_method_agreement(records, sign_count_scores, min_samples_per_pair=3)
    sign_sc = score_sign_agreement(records, sign_count_scores, min_samples_per_pair=3, method_threshold=0.5)
    print(f"TCAV sign_count (current, v62 headline): n={rho_sc.n_pairs} rho={rho_sc.spearman_rho:+.4f} "
          f"p={rho_sc.spearman_p:.4g} | sign={sign_sc.agreement_frac:.1%} ({sign_sc.n_agree}/{sign_sc.n_pairs}) "
          f"p={sign_sc.binom_p:.4g}")

    rho_mag = score_method_agreement(records, magnitude_scores, min_samples_per_pair=3)
    sign_mag = score_sign_agreement(records, magnitude_scores, min_samples_per_pair=3, method_threshold=0.0)
    print(f"TCAV magnitude (new):                     n={rho_mag.n_pairs} rho={rho_mag.spearman_rho:+.4f} "
          f"p={rho_mag.spearman_p:.4g} | sign={sign_mag.agreement_frac:.1%} ({sign_mag.n_agree}/{sign_mag.n_pairs}) "
          f"p={sign_mag.binom_p:.4g}")


if __name__ == "__main__":
    main()
