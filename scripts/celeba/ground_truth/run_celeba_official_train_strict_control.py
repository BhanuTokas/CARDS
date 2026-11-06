"""Re-measures the random-mask control for the official-train classifier's
clean ground truth, with the overlap limit actually enforced.

`random_placements`' default fallback square skips the overlap check, so
large concepts (hair, skin) got a "random" mask that could land on the
concept itself, which biases `delta_p - random_delta_p_mean` against them.

Uses the exact (image, concept, task) records of
`celeba_official_train_faithfulness_non_overlapping.csv` rather than
resampling, so the two versions compare directly and each record's
recomputed `delta_p` validates the rerun against the stored value.

Two controls per record:
  - `strict` (primary): no overlap with the concept's own mask.
  - `strict_excl` (sensitivity): additionally avoids the regions of every
    other concept labelled present in that image.

Appends as it goes and skips finished records on restart, so an interrupted
run resumes. Needs CelebAMask-HQ, so it runs where those masks exist.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from run_celeba_official_train_faithfulness import (
    CELEBA_HQ_ROOT,
    CONCEPT_TO_IDX,
    DEVICE,
    FILL_STRATEGY,
    N_RANDOM_DRAWS,
    RESULTS_DIR,
    SEED,
    TASKS,
    OfficialTrainTaskAdapter,
    load_official_train_native,
    official_train_preprocess,
)

from cards.data.celeba import load_celebamask_hq_image_paths, load_celebamask_hq_mask
from cards.data.celeba_attributes import (
    ATTRIBUTE_TO_REGIONS,
    GROUNDABLE_CONCEPTS,
    load_attribute_labels,
    load_attribute_names,
)
from cards.validation.broden_faithfulness import compute_faithfulness

SOURCE_CSV = RESULTS_DIR / "celeba_official_train_faithfulness_non_overlapping.csv"
OUT_CSV = RESULTS_DIR / "celeba_official_train_faithfulness_non_overlapping_strict_control.csv"
FIELDS = [
    "image", "concept_name", "target_task", "delta_p", "delta_p_stored", "permissive_random_mean",
    "strict_random_mean", "strict_n_valid", "strict_excl_random_mean", "strict_excl_n_valid",
]
# Largest |recomputed - stored delta_p| tolerated before the rerun is treated
# as not reproducing the source ground truth.
MAX_DELTA_P_DIFF = 1e-3
LIMIT = int(os.environ.get("STRICT_CONTROL_LIMIT", "0"))  # >0 = smoke test


def hq_index(image_path: str) -> int:
    return int(re.search(r"(\d+)\.jpg", image_path).group(1))


def other_present_masks(concept: str, hq_idx: int, attr_names, attr_labels, hw) -> list[np.ndarray]:
    """One mask per distinct region set of the other concepts labelled present."""
    own_regions = set(ATTRIBUTE_TO_REGIONS[concept])
    labels = attr_labels[f"{hq_idx}.jpg"]
    region_sets = {
        tuple(ATTRIBUTE_TO_REGIONS[c])
        for c in GROUNDABLE_CONCEPTS
        if labels[attr_names.index(c)] and not own_regions & set(ATTRIBUTE_TO_REGIONS[c])
    }
    masks = [load_celebamask_hq_mask(CELEBA_HQ_ROOT, hq_idx, list(r), target_hw=hw) for r in sorted(region_sets)]
    return [m for m in masks if m.any()]


def load_done() -> set[tuple[str, str, str]]:
    if not OUT_CSV.exists():
        return set()
    with open(OUT_CSV, newline="") as f:
        return {(r["image"], r["concept_name"], r["target_task"]) for r in csv.DictReader(f)}


def main():
    with open(SOURCE_CSV, newline="") as f:
        records = list(csv.DictReader(f))
    if LIMIT:
        records = records[:: max(1, len(records) // LIMIT)][:LIMIT]
    done = load_done()
    todo = [r for r in records if (r["image"], r["concept_name"], r["target_task"]) not in done]
    print(f"{len(records)} records, {len(done)} already done, {len(todo)} to run on {DEVICE}", flush=True)

    image_paths = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_file = CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt"
    attr_names = load_attribute_names(attr_file)
    attr_labels = load_attribute_labels(attr_file)
    native = load_official_train_native()
    preprocess = official_train_preprocess()
    adapters = {
        task: OfficialTrainTaskAdapter(native, DEVICE, slice(i * 2, i * 2 + 2), preprocess)
        for i, task in enumerate(TASKS)
    }

    is_new = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        for n, rec in enumerate(todo):
            concept, task = rec["concept_name"], rec["target_task"]
            hq_idx = hq_index(rec["image"])
            image = Image.open(image_paths[hq_idx]).convert("RGB")
            hw = (image.height, image.width)
            mask = load_celebamask_hq_mask(CELEBA_HQ_ROOT, hq_idx, ATTRIBUTE_TO_REGIONS[concept], target_hw=hw)
            others = other_present_masks(concept, hq_idx, attr_names, attr_labels, hw)

            measured = {}
            for name, exclude in (("strict", None), ("strict_excl", others)):
                seed = SEED + hq_idx * 100 + CONCEPT_TO_IDX[concept] + (1 if exclude is not None else 0)
                measured[name] = compute_faithfulness(
                    image=image, image_path=rec["image"], concept_number=CONCEPT_TO_IDX[concept], category="full",
                    mask=mask, model=adapters[task], rng=np.random.default_rng(seed),
                    n_random_draws=N_RANDOM_DRAWS, fill_strategy=FILL_STRATEGY, device=DEVICE, target_class=1,
                    exclude_masks=exclude, strict_control=True,
                )

            delta_p = measured["strict"].delta_p
            stored = float(rec["delta_p"])
            if not math.isclose(delta_p, stored, abs_tol=MAX_DELTA_P_DIFF):
                raise RuntimeError(f"{rec['image']} {concept}/{task}: delta_p {delta_p:.5f} != stored {stored:.5f}")

            writer.writerow({
                "image": rec["image"], "concept_name": concept, "target_task": task,
                "delta_p": delta_p, "delta_p_stored": stored,
                "permissive_random_mean": rec["random_delta_p_mean"],
                "strict_random_mean": measured["strict"].random_delta_p_mean,
                "strict_n_valid": measured["strict"].n_random_valid,
                "strict_excl_random_mean": measured["strict_excl"].random_delta_p_mean,
                "strict_excl_n_valid": measured["strict_excl"].n_random_valid,
            })
            out.flush()
            if (n + 1) % 100 == 0:
                print(f"  {n + 1}/{len(todo)} ({concept}/{task})", flush=True)
    print(f"Saved to {OUT_CSV}", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
