"""Recovers human-readable names for the CUB concept bank baked into
post_hoc_cbm's trained CUB PCBM checkpoint, and reports each concept's
train/test CAV accuracy.

The checkpoint's concept bank keys are anonymous integers (0-111), not
names -- they're the standard Concept Bottleneck Models 312->112 CUB
attribute filtering (Koh et al.), where concept-bank index order matches
line order in new_attributes.txt as shipped with the CUB metadata release
(each line is "<raw_attribute_id> <name>"; the raw id itself is NOT the
concept-bank index). This was verified against the class-level
majority-vote attribute labels used to build
../Datasets/CUB_200_2011/class_attr_data_10/*.pkl: 4.16% mismatch on a
200-image spot check, consistent with threshold-rounding noise at the 50%
presence boundary rather than a wrong/shuffled mapping (a wrong mapping
would land close to 50% mismatch, not ~4%).

Usage (from the repo root):
    uv run python scripts/cub_concept_bank_accuracies.py
    uv run python scripts/cub_concept_bank_accuracies.py --concept-bank <path> --output results/cub_concepts.csv
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

DEFAULT_CONCEPT_BANK = Path(
    "../post_hoc_cbm/trained_concepts_new/cub/resnet18_cub/cub_resnet18_cub_0.1_100.pkl"
)
DEFAULT_ATTRIBUTE_NAMES = Path("../Datasets/CUB_200_2011/attributes/new_attributes.txt")
DEFAULT_OUTPUT = Path("results/cub_concept_accuracies.csv")


def load_attribute_names(path: Path) -> dict[int, str]:
    """0-indexed concept-bank index -> attribute name."""
    names = {}
    for index, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        _, name = line.split(maxsplit=1)
        names[index] = name
    return names


def load_concept_accuracies(bank_path: Path) -> list[tuple[int, float, float]]:
    """(concept_idx, train_acc, test_acc) per concept, from a post_hoc_cbm
    concept-bank pickle keyed by concept_idx -> (vector, train_acc,
    test_acc, intercept, margin_info).

    pickle.load runs arbitrary code embedded in the file: only point
    --concept-bank at a file you trust (the default is a local, known
    checkpoint), never at an untrusted or externally-supplied one.
    """
    with open(bank_path, "rb") as f:
        bank = pickle.load(f)
    return [(idx, bank[idx][1], bank[idx][2]) for idx in sorted(bank.keys())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--concept-bank", type=Path, default=DEFAULT_CONCEPT_BANK)
    parser.add_argument("--attribute-names", type=Path, default=DEFAULT_ATTRIBUTE_NAMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    names = load_attribute_names(args.attribute_names)
    accuracies = load_concept_accuracies(args.concept_bank)

    rows = [
        (idx, names.get(idx, f"<unknown:{idx}>"), train_acc, test_acc)
        for idx, train_acc, test_acc in accuracies
    ]
    rows.sort(key=lambda row: row[3])

    for idx, name, train_acc, test_acc in rows:
        print(f"[{idx:>3}] {name:<45} train={train_acc:.3f} test={test_acc:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_idx", "name", "train_acc", "test_acc"])
        for idx, name, train_acc, test_acc in rows:
            writer.writerow([idx, name, round(train_acc, 4), round(test_acc, 4)])

    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
