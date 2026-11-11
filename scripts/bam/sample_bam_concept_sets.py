"""Choose the BAM training images for TCAV's and PCBM's concept sets.

Mirrors the CelebA setup (`run_tcav_celeba_official_train.py` and
`fit_celeba_official_train_cavs.py`), with BAM's 20 classes as concepts and its
training composites in place of labelled training images:
  TCAV: 40 exemplars per concept, 6 random and 6 control sets of 25 images,
        and 40 inputs per class for each model.
  PCBM: 100 positives and 100 negatives per concept for its CAVs, which
        post_hoc_cbm splits 50/50 into fit and held-out halves.
Writes tcav_sets.csv, tcav_inputs.csv and pcbm_cav_sets.csv to --out.
"""

import argparse
import csv
import random
from pathlib import Path

SEED = 42
OBJECTS = ["backpack", "bird", "dog", "elephant", "kite", "pizza", "stop_sign", "toilet", "truck", "zebra"]
SCENES = ["bamboo_forest", "bedroom", "bowling_alley", "bus_interior", "cockpit", "corn_field",
          "laundromat", "runway", "ski_slope", "track"]
N_RANDOM, N_CONTROL, N_PER_RANDOM_SET = 6, 6, 25
N_CONCEPT_EXEMPLARS, N_INPUTS = 40, 40
N_CAV_POS = N_CAV_NEG = 100


def write_csv(path: Path, rows: list) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path.name}: {len(rows)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with open(args.train_manifest, newline="") as handle:
        rows = list(csv.DictReader(handle))
    rng = random.Random(SEED)
    every = [r["file"] for r in rows]
    has = {c: [r["file"] for r in rows if r["object"] == c] for c in OBJECTS}
    has.update({c: [r["file"] for r in rows if r["scene"] == c] for c in SCENES})

    tcav_rows, input_rows, cav_rows = [], [], []
    shuffled = every.copy()
    rng.shuffle(shuffled)
    cursor = 0
    for kind, count in (("random", N_RANDOM), ("control", N_CONTROL)):
        for i in range(count):
            for name in shuffled[cursor:cursor + N_PER_RANDOM_SET]:
                tcav_rows.append({"set": f"{kind}_{i}", "file": name})
            cursor += N_PER_RANDOM_SET
    used = set(shuffled[:cursor])

    for concept in OBJECTS + SCENES:
        positives = [f for f in has[concept] if f not in used]
        rng.shuffle(positives)
        for name in positives[:N_CONCEPT_EXEMPLARS]:
            tcav_rows.append({"set": concept, "file": name})
        for name in positives[N_CONCEPT_EXEMPLARS:N_CONCEPT_EXEMPLARS + N_CAV_POS]:
            cav_rows.append({"concept": concept, "label": 1, "file": name})
        with_concept = set(has[concept])
        negatives = [f for f in every if f not in with_concept]
        rng.shuffle(negatives)
        for name in negatives[:N_CAV_NEG]:
            cav_rows.append({"concept": concept, "label": 0, "file": name})

    # TCAV inputs: images of each class, with that class's index in its own model.
    for model, classes in (("obj", OBJECTS), ("scene", SCENES)):
        for target, cls in enumerate(classes):
            pool = has[cls].copy()
            rng.shuffle(pool)
            for name in pool[:N_INPUTS]:
                input_rows.append({"model": model, "target": target, "class": cls, "file": name})

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "tcav_sets.csv", tcav_rows)
    write_csv(args.out / "tcav_inputs.csv", input_rows)
    write_csv(args.out / "pcbm_cav_sets.csv", cav_rows)


if __name__ == "__main__":
    main()
