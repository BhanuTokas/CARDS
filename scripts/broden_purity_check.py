"""Broden ground-truth retrieval-purity check (design doc Section 2, item
5): measures how well CARDS' Step 1-2 CLIP retrieval agrees with Broden's
already-labeled positives/negatives, per concept.

Usage (from the repo root):
    uv run python scripts/broden_purity_check.py
    uv run python scripts/broden_purity_check.py --concepts dog cat bird
    uv run python scripts/broden_purity_check.py --output results/broden_purity.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cards.encoders.open_clip_encoder import OpenClipEncoder
from cards.validation.broden_purity import check_all_concepts_purity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("../Datasets/broden_concepts"))
    parser.add_argument("--concepts", nargs="*", default=None, help="Subset of concepts to check (default: all)")
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("results/broden_purity_summary.csv"))
    args = parser.parse_args()

    encoder = OpenClipEncoder(args.model_name, args.pretrained, device=args.device)
    results = check_all_concepts_purity(args.root, encoder, concepts=args.concepts)
    results.sort(key=lambda result: result.average_precision)

    for result in results:
        print(
            f"{result.concept:<20} n_pos={result.n_positives:>4} n_neg={result.n_negatives:>4} "
            f"precision@k={result.precision_at_k:.3f} neg_recall@k={result.negative_recall_at_k:.3f} "
            f"AP={result.average_precision:.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["concept", "n_positives", "n_negatives", "precision_at_k", "negative_recall_at_k", "average_precision"]
        )
        for result in results:
            writer.writerow(
                [
                    result.concept,
                    result.n_positives,
                    result.n_negatives,
                    round(result.precision_at_k, 4),
                    round(result.negative_recall_at_k, 4),
                    round(result.average_precision, 4),
                ]
            )
    print(f"\nWrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
