"""Flags Broden ground-truth images that CARDS' CLIP retrieval strongly
disagrees with -- candidates for manual label-quality review. Follows up
on the design doc's Section 2 item 5 validation check; this surfaced real
labeling errors in the local Broden copy during initial validation (e.g.
several `air_conditioner`-labeled images that were actually airport
control towers).

Usage (from the repo root):
    uv run python scripts/broden_label_flags.py
    uv run python scripts/broden_label_flags.py --concepts air_conditioner --flag-fraction 0.2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cards.encoders.open_clip_encoder import OpenClipEncoder
from cards.validation.broden_purity import flag_all_concepts_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("../Datasets/broden_concepts"))
    parser.add_argument("--concepts", nargs="*", default=None, help="Subset of concepts to check (default: all)")
    parser.add_argument("--flag-fraction", type=float, default=0.15, help="Fraction of each side to flag as suspect")
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("results/broden_label_flags.csv"))
    args = parser.parse_args()

    encoder = OpenClipEncoder(args.model_name, args.pretrained, device=args.device)
    flags = flag_all_concepts_labels(
        args.root, encoder, concepts=args.concepts, flag_fraction=args.flag_fraction
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept", "flag", "path", "rank", "pool_size", "similarity"])
        for flag in flags:
            writer.writerow(
                [flag.concept, flag.flag, str(flag.path), flag.rank, flag.pool_size, round(flag.similarity, 4)]
            )

    print(f"Wrote {len(flags)} flags to {args.output}")


if __name__ == "__main__":
    main()
