"""Balance faithfulness CSV by concept image count, then compute rho + agreement.

Run without --min_count to print per-concept image/instance counts so you can
decide the balancing threshold. Run with --min_count N to subsample each
concept to N images and compute rho + sign agreement on the balanced set.

Usage:
  # Step 1: inspect counts
  python balance_faithfulness_and_score.py

  # Step 2: balance and score
  python balance_faithfulness_and_score.py --min_count 50

Prerequisites:
  compute_coco_stuff_faithfulness.py must have run (produces faithfulness CSV
  and detection counts CSV).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np

from cards.data.coco_stuff import CO_OCCUR_BRODEN_CONCEPTS, COCO_STUFF_20_CLASSES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR  = Path("results/coco_stuff")
FAITH_CSV    = RESULTS_DIR / "coco_stuff_faithfulness.csv"
COUNTS_CSV   = RESULTS_DIR / "coco_stuff_faithfulness_detection_counts.csv"
SEED         = 42
MIN_SAMPLES  = 3   # min images per (concept, class) pair to include in rho


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

STRATEGIES = ("black_mask", "black_bbox")


def load_faithfulness_raw(
    path: Path, strategy: str = "black_mask"
) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """Return concept -> class -> list of (image_path, delta_p) for the given strategy."""
    col = f"delta_p_{strategy}"
    data: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data[row["concept_name"]][row["class_name"]].append(
                (row["image_path"], float(row[col]))
            )
    return data


def load_counts(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------

def get_image_counts(faith_raw: dict) -> dict[str, int]:
    """Distinct images per concept across all classes."""
    counts: dict[str, int] = {}
    for concept, class_dict in faith_raw.items():
        images: set[str] = set()
        for img_dp_list in class_dict.values():
            for img, _ in img_dp_list:
                images.add(img)
        counts[concept] = len(images)
    return counts


def balance(
    faith_raw: dict[str, dict[str, list[tuple[str, float]]]],
    min_count: int,
    rng: np.random.Generator,
) -> dict[tuple[str, str], list[float]]:
    """Subsample each concept to min_count images, return (concept, class) -> [delta_p]."""
    result: dict[tuple[str, str], list[float]] = defaultdict(list)

    for concept, class_dict in faith_raw.items():
        # Collect all distinct images for this concept
        all_images: set[str] = set()
        for img_dp_list in class_dict.values():
            for img, _ in img_dp_list:
                all_images.add(img)

        if len(all_images) < min_count:
            continue  # skip concepts with too few images

        sampled = set(rng.choice(sorted(all_images), size=min_count, replace=False))

        for cls_name, img_dp_list in class_dict.items():
            for img, dp in img_dp_list:
                if img in sampled:
                    result[(concept, cls_name)].append(dp)

    return dict(result)


# ---------------------------------------------------------------------------
# Pair-count summary (after balancing)
# ---------------------------------------------------------------------------

def _print_pair_counts(
    faith: dict[tuple[str, str], list[float]],
    label: str,
) -> None:
    """Print n_all and n_co-occur pairs per class for the balanced faithfulness dict."""
    n_all_list, n_co_list = [], []

    print(f"\n[{label}]  {'Class':<22}  {'n_all':>6}  {'n_co':>6}", flush=True)
    print("-" * 50, flush=True)

    for cls_name in COCO_STUFF_20_CLASSES:
        n_all, n_co = 0, 0
        for (c, cl), dp_list in faith.items():
            if cl != cls_name or len(dp_list) < MIN_SAMPLES:
                continue
            n_all += 1
            if c in CO_OCCUR_BRODEN_CONCEPTS:
                n_co += 1
        print(f"  {cls_name:<22}  {n_all:>6d}  {n_co:>6d}", flush=True)
        if n_all >= 2:
            n_all_list.append(n_all)
        if n_co >= 2:
            n_co_list.append(n_co)

    mean_all = float(np.mean(n_all_list)) if n_all_list else 0.0
    mean_co  = float(np.mean(n_co_list))  if n_co_list  else 0.0
    print(f"\n  Mean pairs/class (all):     {mean_all:.1f}", flush=True)
    print(f"  Mean pairs/class (co-occur): {mean_co:.1f}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_count", type=int, default=None,
                        help="Images per concept to subsample to. Omit to just print counts.")
    args = parser.parse_args()

    if not FAITH_CSV.exists():
        raise FileNotFoundError(f"Faithfulness CSV not found: {FAITH_CSV}")

    # Image counts are strategy-agnostic — load once from black_mask
    faith_raw_bm = load_faithfulness_raw(FAITH_CSV, strategy="black_mask")
    image_counts  = get_image_counts(faith_raw_bm)

    # --- Print per-concept image/instance counts ---
    print(f"\n{'Concept':<30s}  {'n_images':>8}  {'n_instances':>11}", flush=True)
    print("-" * 55, flush=True)

    inst_counts: dict[str, int] = {}
    if COUNTS_CSV.exists():
        for row in load_counts(COUNTS_CSV):
            inst_counts[row["concept_name"]] = int(row["n_instances"])

    for concept, n_img in sorted(image_counts.items(), key=lambda x: -x[1]):
        n_ins = inst_counts.get(concept, 0)
        print(f"  {concept:<30s}  {n_img:>8d}  {n_ins:>11d}", flush=True)

    n_zero = sum(1 for n in image_counts.values() if n == 0)
    print(f"\nConcepts with 0 images:  {n_zero}", flush=True)
    print(f"Concepts with >=1 image: {len(image_counts) - n_zero}", flush=True)

    if args.min_count is None:
        print("\nRe-run with --min_count N to balance and score.", flush=True)
        return

    # --- Balance and print pair counts for both strategies ---
    # Use same seed per strategy so identical image subsets are selected.
    for strategy in STRATEGIES:
        faith_raw  = load_faithfulness_raw(FAITH_CSV, strategy=strategy)
        balanced   = balance(faith_raw, args.min_count, np.random.default_rng(SEED))
        n_concepts = len({c for c, _ in balanced})
        print(f"\nBalanced ({strategy}): {n_concepts} concepts  "
              f"(min_count={args.min_count} images/concept)", flush=True)
        _print_pair_counts(balanced, label=strategy)


if __name__ == "__main__":
    main()
