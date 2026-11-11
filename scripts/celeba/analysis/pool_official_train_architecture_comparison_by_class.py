"""Class-pooled counterpart of `compare_official_train_across_
architectures.py`'s own table -- prompted directly ("can we get the
class pooled results in form of a table?"), the same request pattern
as the earlier local-attribution track ("Can we have the class pooled
scores for rho?").

**Does NOT re-derive rho from pooled raw (image, score) pairs** -- that
data isn't available at this table's level (only each task's own
already-computed summary rho is), and the earlier local-attribution
check (`score_local_attribution_present_vs_complete_official_train.py`)
already showed naive raw-pooling across Attractive+Male is structurally
biased (their score distributions differ in scale/sign), so it
shouldn't be reintroduced here even if the raw pairs were on hand.

Instead pools the two ALREADY-COMPUTED per-task rhos
(architecture, method, gt_version) two ways, both reported side by
side rather than picking one silently:
  - `pooled_rho_naive_mean` -- plain arithmetic mean of the two rhos.
    Distorted near +-1 (correlations aren't on a linear scale), but
    the simpler, more legible number.
  - `pooled_rho_fisherz` -- Fisher z-transform (`arctanh`), averaged
    weighted by each task's own (n-3) [here both n=26, so this is a
    weighted average that happens to equal a simple average of z],
    then back-transformed (`tanh`). The statistically correct way to
    average correlations, and what `score_local_attribution_present_
    vs_complete_official_train.py` used as its own recommended
    "combined" number.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")
IN_CSV = RESULTS_DIR / "celeba_official_train_architecture_comparison.csv"
OUT_CSV = RESULTS_DIR / "celeba_official_train_architecture_comparison_class_pooled.csv"

ARCHITECTURES = ["resnet18", "vit", "convnext"]
ARCH_LABELS = {"resnet18": "ResNet18", "vit": "ViT-B/16", "convnext": "ConvNeXt-Tiny"}
GT_VERSIONS = ["non_overlapping", "previously_used"]
METHODS = ["CARDS (masking hybrid)", "TCAV", "PCBM (conventional)", "PCBM (CLIP-SigLIP)", "PCBM (CLIP-RN50)"]


def fisher_z_pool(rho_attractive: float, rho_male: float, n_attractive: int, n_male: int) -> float:
    z_a, z_m = np.arctanh(rho_attractive), np.arctanh(rho_male)
    w_a, w_m = max(n_attractive - 3, 1), max(n_male - 3, 1)
    z_pooled = (w_a * z_a + w_m * z_m) / (w_a + w_m)
    return float(np.tanh(z_pooled))


def main():
    by_key: dict[tuple[str, str, str, str], dict] = {}
    with open(IN_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["architecture"], row["task"], row["gt_version"], row["method"])
            by_key[key] = row

    out_rows = []
    for architecture in ARCHITECTURES:
        for gt_version in GT_VERSIONS:
            for method in METHODS:
                r_a = by_key.get((architecture, "Attractive", gt_version, method))
                r_m = by_key.get((architecture, "Male", gt_version, method))
                if r_a is None or r_m is None:
                    continue  # ResNet18 has no PCBM-CLIP-concepts rows
                rho_a, rho_m = float(r_a["spearman_rho"]), float(r_m["spearman_rho"])
                n_a, n_m = int(r_a["n_pairs"]), int(r_m["n_pairs"])
                pooled_naive = (rho_a + rho_m) / 2
                pooled_fz = fisher_z_pool(rho_a, rho_m, n_a, n_m)
                out_rows.append({
                    "architecture": architecture, "gt_version": gt_version, "method": method,
                    "rho_attractive": rho_a, "rho_male": rho_m,
                    "pooled_rho_naive_mean": pooled_naive, "pooled_rho_fisherz": pooled_fz,
                })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["architecture", "gt_version", "method", "rho_attractive",
                                                "rho_male", "pooled_rho_naive_mean", "pooled_rho_fisherz"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Saved {len(out_rows)} rows to {OUT_CSV}\n", flush=True)

    for gt_version in GT_VERSIONS:
        print(f"\n{'=' * 100}\ngt_version = {gt_version}\n{'=' * 100}")
        print(f"  {'method':<24s} {'resnet18':>22s} {'vit':>22s} {'convnext':>22s}")
        print(f"  {'':<24s} {'(naive / fisher-z)':>22s} {'(naive / fisher-z)':>22s} {'(naive / fisher-z)':>22s}")
        for method in METHODS:
            cells = []
            for architecture in ARCHITECTURES:
                row = next((r for r in out_rows if r["architecture"] == architecture
                            and r["gt_version"] == gt_version and r["method"] == method), None)
                cells.append(f"{row['pooled_rho_naive_mean']:+.3f} / {row['pooled_rho_fisherz']:+.3f}" if row else "--")
            print(f"  {method:<24s} {cells[0]:>22s} {cells[1]:>22s} {cells[2]:>22s}")


if __name__ == "__main__":
    main()
