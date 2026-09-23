"""Class-pooled counterpart of `ablate_cards_celeba_masking_hybrid_
joint_orth_demean_encoder_official_train.py`'s own 64-row output --
prompted directly ("Ok, let's pool across classes"), the same request
pattern as the architecture ablation's own `pool_official_train_
architecture_comparison_by_class.py`.

Same two-way reporting as that script (never picks one silently):
  - `pooled_rho_naive_mean` -- plain arithmetic mean of the two rhos.
  - `pooled_rho_fisherz` -- Fisher z-transform (arctanh), averaged
    weighted by each task's own (n-3) [here both n=26, so this reduces
    to a simple average of z], then back-transformed (tanh).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")
IN_CSV = RESULTS_DIR / "cards_celeba_masking_hybrid_joint_orth_demean_encoder_official_train_ablation.csv"
OUT_CSV = RESULTS_DIR / "cards_celeba_masking_hybrid_joint_orth_demean_encoder_official_train_ablation_class_pooled.csv"

ENCODERS = ["siglip", "clip", "open_clip_h", "perception_encoder"]
GT_VERSIONS = ["non_overlapping", "previously_used"]
CONFIGS = [(orth, demean) for orth in (True, False) for demean in (True, False)]


def fisher_z_pool(rho_attractive: float, rho_male: float, n_attractive: int, n_male: int) -> float:
    z_a, z_m = np.arctanh(rho_attractive), np.arctanh(rho_male)
    w_a, w_m = max(n_attractive - 3, 1), max(n_male - 3, 1)
    z_pooled = (w_a * z_a + w_m * z_m) / (w_a + w_m)
    return float(np.tanh(z_pooled))


def main():
    by_key: dict[tuple[str, str, str, str, str], dict] = {}
    with open(IN_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["encoder"], row["orthogonalize"], row["demean_query"], row["target_task"], row["gt_version"])
            by_key[key] = row

    out_rows = []
    for encoder in ENCODERS:
        for orth, demean in CONFIGS:
            for gt_version in GT_VERSIONS:
                r_a = by_key.get((encoder, str(orth), str(demean), "Attractive", gt_version))
                r_m = by_key.get((encoder, str(orth), str(demean), "Male", gt_version))
                if r_a is None or r_m is None:
                    continue
                rho_a, rho_m = float(r_a["spearman_rho"]), float(r_m["spearman_rho"])
                n_a, n_m = int(r_a["n_pairs"]), int(r_m["n_pairs"])
                pooled_naive = (rho_a + rho_m) / 2
                pooled_fz = fisher_z_pool(rho_a, rho_m, n_a, n_m)
                out_rows.append({
                    "encoder": encoder, "orthogonalize": orth, "demean_query": demean, "gt_version": gt_version,
                    "rho_attractive": rho_a, "rho_male": rho_m,
                    "pooled_rho_naive_mean": pooled_naive, "pooled_rho_fisherz": pooled_fz,
                })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["encoder", "orthogonalize", "demean_query", "gt_version",
                                                "rho_attractive", "rho_male", "pooled_rho_naive_mean", "pooled_rho_fisherz"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Saved {len(out_rows)} rows to {OUT_CSV}\n", flush=True)

    for gt_version in GT_VERSIONS:
        print(f"\n{'=' * 20} gt_version = {gt_version} {'=' * 20}")
        ranked = sorted((r for r in out_rows if r["gt_version"] == gt_version),
                         key=lambda r: -abs(r["pooled_rho_fisherz"]))
        for r in ranked:
            print(f"  encoder={r['encoder']:<20s} orth={r['orthogonalize']!s:<5s} demean={r['demean_query']!s:<5s} "
                  f"pooled_rho: naive={r['pooled_rho_naive_mean']:+.4f} fisherz={r['pooled_rho_fisherz']:+.4f}")


if __name__ == "__main__":
    main()
