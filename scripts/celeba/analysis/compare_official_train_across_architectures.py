"""3-architecture comparison for the official-train CelebA main pipeline
(ResNet18 / ViT-B/16 / ConvNeXt-Tiny), prompted directly ("The full
pipeline with PCBM variants" -> ViT+ConvNeXt scope decision -> whole-
image PCBM redo decision -> "download all three ... locally and I can
build the 3-architecture comparison").

Reads the three already-computed `score_all_methods_against_official_
train[_vit|_convnext]_faithfulness.csv` files (each produced independently
on Sol) and combines them into one long table plus a readable summary.

**Method-set mismatch is real, not a bug**: ResNet18's own file only has
3 methods (`CARDS (masking hybrid)`, `TCAV`, `PCBM`) -- the PCBM-CLIP-
concepts variants were never backfilled for ResNet18 (the earlier scope
decision was "ViT + ConvNeXt only" for that specific extension), while
ViT/ConvNeXt have 5 (the same two plus `PCBM (conventional)`, `PCBM
(CLIP-SigLIP)`, `PCBM (CLIP-RN50)`). ResNet18's plain `PCBM` is the same
whole-image-sourced conventional CAV-based method as ViT/ConvNeXt's own
`PCBM (conventional)` (all three were redone from crops to whole-images
in the same pass), so it's renamed to that label here for a fair 3-way
comparison -- the two CLIP-concepts variants simply have no ResNet18 row
and are left blank there, not estimated or dropped from the other two
architectures.
"""

from __future__ import annotations

import csv
from pathlib import Path

RESULTS_DIR = Path("results")
ARCH_FILES = {
    "resnet18": RESULTS_DIR / "score_all_methods_against_official_train_faithfulness.csv",
    "vit": RESULTS_DIR / "score_all_methods_against_official_train_vit_faithfulness.csv",
    "convnext": RESULTS_DIR / "score_all_methods_against_official_train_convnext_faithfulness.csv",
}
ARCHITECTURES = ["resnet18", "vit", "convnext"]
METHOD_RENAME = {"PCBM": "PCBM (conventional)"}
METHODS = ["CARDS (masking hybrid)", "TCAV", "PCBM (conventional)", "PCBM (CLIP-SigLIP)", "PCBM (CLIP-RN50)"]
OUT_CSV = RESULTS_DIR / "celeba_official_train_architecture_comparison.csv"


def load_arch_rows(architecture: str) -> list[dict]:
    rows = []
    with open(ARCH_FILES[architecture], newline="") as f:
        for row in csv.DictReader(f):
            method = METHOD_RENAME.get(row["method"], row["method"])
            rows.append({
                "architecture": architecture, "task": row["task"], "gt_version": row["gt_version"],
                "method": method, "n_pairs": int(row["n_pairs"]),
                "spearman_rho": float(row["spearman_rho"]), "spearman_p": float(row["spearman_p"]),
                "sign_agreement": float(row["sign_agreement"]), "n_agree": int(row["n_agree"]),
                "binom_p": float(row["binom_p"]),
            })
    return rows


def main():
    all_rows = []
    for architecture in ARCHITECTURES:
        rows = load_arch_rows(architecture)
        all_rows.extend(rows)
        print(f"{architecture}: {len(rows)} rows, methods = {sorted({r['method'] for r in rows})}", flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["architecture", "task", "gt_version", "method", "n_pairs",
                          "spearman_rho", "spearman_p", "sign_agreement", "n_agree", "binom_p"])
        for row in all_rows:
            writer.writerow([row["architecture"], row["task"], row["gt_version"], row["method"],
                              row["n_pairs"], row["spearman_rho"], row["spearman_p"],
                              row["sign_agreement"], row["n_agree"], row["binom_p"]])
    print(f"\nSaved {len(all_rows)} rows to {OUT_CSV}\n", flush=True)

    by_key = {(r["architecture"], r["task"], r["gt_version"], r["method"]): r for r in all_rows}
    for gt_version in ["non_overlapping", "previously_used"]:
        print(f"\n{'=' * 90}\ngt_version = {gt_version}\n{'=' * 90}")
        for task in ["Attractive", "Male"]:
            print(f"\n  -- {task} -- Spearman rho (sign_agreement)")
            print(f"  {'method':<24s} {'resnet18':>18s} {'vit':>18s} {'convnext':>18s}")
            for method in METHODS:
                cells = []
                for architecture in ARCHITECTURES:
                    r = by_key.get((architecture, task, gt_version, method))
                    cells.append(f"{r['spearman_rho']:+.3f} ({r['sign_agreement']:.0%})" if r else "--")
                print(f"  {method:<24s} {cells[0]:>18s} {cells[1]:>18s} {cells[2]:>18s}")

    print(f"\n{'=' * 90}\nWinner (highest rho) per (task, gt_version, architecture), among methods present\n{'=' * 90}")
    for architecture in ARCHITECTURES:
        for task in ["Attractive", "Male"]:
            for gt_version in ["non_overlapping", "previously_used"]:
                candidates = [r for r in all_rows if r["architecture"] == architecture
                              and r["task"] == task and r["gt_version"] == gt_version]
                best = max(candidates, key=lambda r: r["spearman_rho"])
                print(f"  {architecture:<10s} {task:<10s} {gt_version:<16s} -> {best['method']:<24s} "
                      f"rho={best['spearman_rho']:+.3f}")


if __name__ == "__main__":
    main()
