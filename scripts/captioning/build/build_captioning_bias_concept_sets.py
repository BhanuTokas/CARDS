"""Materializes the positive/negative concept sets and the Gender target-
class split from `cards.data.captioning_bias` into one CSV, so downstream
scoring (once HPC-generated captions + gender-word scores land) can just
join on `img_name` rather than re-deriving sets from the raw pickle every
time.

Scope decided directly with the user: Gender only for now (Race deferred --
no equivalent race word list exists for the log-prob scoring approach, see
`captioning_bias.py`'s own docstring). `bb_skin` is still written to the
output CSV as a raw passthrough column in case Race work resumes later, but
no Race-specific set-building happens here.

Wide format (one row per image, one 0/1 column per concept) chosen to match
this project's own attribute-file convention (list_attr_celeba.txt /
CelebA-HQ-attribute-anno.txt), not a long (concept, image, label) format --
makes joining a per-model gender-score table on `img_name` a single pandas
merge rather than a groupby.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cards.data.captioning_bias import (
    TARGET_CLASS_VALUES,
    groundable_concepts,
    load_bias_captioning_records,
)

RESULTS_DIR = Path("results")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    records = load_bias_captioning_records()
    concepts = groundable_concepts(records)
    print(f"{len(records)} images, {len(concepts)} groundable (non-'person') concepts.", flush=True)

    male_value, female_value = TARGET_CLASS_VALUES["Gender"]
    n_excluded_gender = sum(1 for r in records if r["bb_gender"] not in (male_value, female_value))
    print(f"Gender: {sum(1 for r in records if r['bb_gender'] == male_value)} {male_value}, "
          f"{sum(1 for r in records if r['bb_gender'] == female_value)} {female_value}, "
          f"{n_excluded_gender} other/excluded.", flush=True)

    out_path = RESULTS_DIR / "captioning_bias_concept_sets.csv"
    fieldnames = ["img_name", "image_id", "split", "gender", "bb_skin", *concepts]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {
                "img_name": r["img_name"],
                "image_id": r["img_id"],
                "split": r["split"],
                "gender": r["bb_gender"] if r["bb_gender"] in (male_value, female_value) else "",
                "bb_skin": r["bb_skin"],
            }
            present = set(r["rmdup_object_list"])
            for c in concepts:
                row[c] = int(c in present)
            writer.writerow(row)

    print(f"Saved {len(records)} rows x {len(fieldnames)} columns to {out_path}", flush=True)

    print("\nPer-concept positive counts (min/max sanity check):")
    counts = {c: sum(1 for r in records if c in r["rmdup_object_list"]) for c in concepts}
    for c in sorted(counts, key=counts.get)[:5]:
        print(f"  rarest: {c:<20s} n_positive={counts[c]}")
    for c in sorted(counts, key=counts.get, reverse=True)[:5]:
        print(f"  most common: {c:<20s} n_positive={counts[c]}")


if __name__ == "__main__":
    main()
