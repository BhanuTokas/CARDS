"""Phase 3 driver (notes/pcbm_correlation_investigation.md, CARDS vs.
TCAV vs. PCBM plan): runs the masking-based faithfulness metric on real
Broden images for the same 5 concepts validated in Phase 2's TCAV slice
(car/cat/dog/chair/bottle), against the native resnet18 model directly.

Two separate sampling procedures, run independently and compared (not
one sliced two ways, to avoid confounding "more data" with "targeted
selection"):

- "volume": a large uniform-random sample per concept (N_IMAGES_PER_CONCEPT),
  the original design -- does masking the real concept region hurt the
  model's own prediction (whatever it is) more than masking a random
  region? Self-contained, no alignment with the 25-class ImageNet subset
  needed.
- "targeted": a cheap unmasked-prediction-only scan (no masking, no
  random draws -- ~1/7 the cost per image of full faithfulness scoring)
  across many more candidates, keeping only images whose native-model
  prediction already lands on one of the 25 CARDS/TCAV/PCBM-scored
  target classes, up to N_TARGET_IN_SCOPE_PER_CONCEPT hits or
  MAX_SCAN_PER_CONCEPT candidates scanned. Directly fixes v28's
  confirmed sampling bottleneck (only 10/25 target classes ever showed
  up by chance in the volume sample), at the cost of a real, named
  selection bias: these are no longer "images containing concept c" but
  "images containing concept c *and* already predicted as one of our 25
  classes" -- a narrower, possibly-easier population. Comparing the two
  procedures' conclusions directly tests whether that bias matters.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from build_imagenet_slice import TARGET_CLASSES  # noqa: E402
from cards.data.broden_raw import build_concept_index, concept_pixel_mask, load_index, load_label_table  # noqa: E402
from cards.models.backbones import BACKBONES  # noqa: E402
from cards.validation.broden_faithfulness import compute_faithfulness  # noqa: E402

NETDISSECT_ROOT = Path(r"C:\Users\btokas\Projects\NetDissect\dataset\broden1_224")
RESULTS_DIR = Path("results")
SEED = 42
DEVICE = "cpu"  # matches Phase 2's device choice; see notes v26 for why
N_IMAGES_PER_CONCEPT = 100  # volume procedure (>3x v27/v28's original 30)
N_TARGET_IN_SCOPE_PER_CONCEPT = 20  # targeted procedure: stop once this many in-scope hits found
MAX_SCAN_PER_CONCEPT = 3000  # targeted procedure: give up after scanning this many candidates
N_RANDOM_DRAWS = 5

# Same 5 concepts as Phase 2's TCAV validation slice, by their NetDissect
# *global* label number (confirmed directly against label.csv).
TEST_CONCEPTS = {"car": 38, "cat": 105, "dog": 93, "chair": 36, "bottle": 70}
IN_SCOPE_NATIVE_INDICES = {idx for idx, _, _ in TARGET_CLASSES}  # the 25 CARDS/TCAV/PCBM-scored classes

# v29's targeted run used ANY of the 25 target classes as "in scope",
# which turned out to admit a lot of indirect/compositional matches
# (e.g. "bottle" mostly landed on dining_table, only 3/20 on its own
# literal water_bottle match) -- this tightens it to each concept's own
# specific namesake class(es) only, for a clean, directly-comparable test
# per concept. Built by inspecting v29's own predicted-class breakdown
# directly, not guessed.
CONCEPT_MATCHING_CLASSES = {
    "car": {817, 511},  # sports_car, convertible (motor_scooter deliberately excluded -- a different vehicle type, not a car)
    "cat": {281, 284, 285},  # tabby_cat, siamese_cat, egyptian_cat
    "dog": {207, 208},  # golden_retriever, labrador_retriever
    "chair": {765, 559},  # rocking_chair, folding_chair
    "bottle": {898},  # water_bottle -- the only direct match among the 25 target classes
}


class NativeModelAdapter:
    """Adapts a BackboneSpec's native model to the MultiClassModel
    protocol compute_faithfulness expects (preprocess + full-logit
    __call__)."""

    def __init__(self, spec, device: str):
        self.model = spec.load_native().to(device).eval()
        self._preprocess = spec.preprocess
        self.device = device

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch.to(self.device))

    @torch.no_grad()
    def quick_predict(self, image: Image.Image) -> int:
        """Unmasked top-1 prediction only -- no masking, no random draws.
        ~1/7 the cost of a full compute_faithfulness call, used by the
        targeted procedure's cheap scan."""
        x = self.preprocess(image).unsqueeze(0)
        return int(torch.argmax(self(x)[0]).item())


def process_one(record, label_number, category, model, rng_seed) -> object | None:
    mask = concept_pixel_mask(record, category, label_number)
    if mask is None or not mask.any():
        return None
    image = Image.open(record.image).convert("RGB")
    if mask.shape != (image.height, image.width):
        return None
    rng_np = np.random.default_rng(rng_seed)
    return compute_faithfulness(
        image=image,
        image_path=str(record.image),
        concept_number=label_number,
        category=category,
        mask=mask,
        model=model,
        rng=rng_np,
        n_random_draws=N_RANDOM_DRAWS,
        fill_strategy="blur",
        device=DEVICE,
    )


def summarize(all_results: list[tuple[str, object]], label: str) -> list[dict]:
    per_concept_summary = []
    for concept_name in TEST_CONCEPTS:
        concept_deltas = [r.delta_p for name, r in all_results if name == concept_name]
        random_deltas = [r.random_delta_p_mean for name, r in all_results if name == concept_name]
        mean_concept_delta = float(np.mean(concept_deltas)) if concept_deltas else float("nan")
        mean_random_delta = float(np.mean(random_deltas)) if random_deltas else float("nan")
        if len(concept_deltas) >= 2:
            t_stat, p_value = ttest_rel(concept_deltas, random_deltas)
        else:
            t_stat, p_value = float("nan"), float("nan")
        print(f"[{label}] {concept_name}: n={len(concept_deltas)}, mean concept delta_p={mean_concept_delta:.4f}, "
              f"mean random-baseline delta_p={mean_random_delta:.4f}, paired t={t_stat:.4f}, p={p_value:.4g}", flush=True)
        per_concept_summary.append(
            {
                "concept": concept_name,
                "label_number": TEST_CONCEPTS[concept_name],
                "n_images": len(concept_deltas),
                "mean_concept_delta_p": mean_concept_delta,
                "mean_random_delta_p": mean_random_delta,
                "paired_t": t_stat,
                "paired_p": p_value,
            }
        )
    return per_concept_summary


def save_csvs(all_results: list[tuple[str, object]], per_concept_summary: list[dict], suffix: str):
    with open(RESULTS_DIR / f"broden_faithfulness_records{suffix}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["concept", "image", "predicted_class", "p0", "p_masked", "delta_p", "delta_logit",
             "random_delta_p_mean", "random_delta_p_std", "z_score", "n_random_fallbacks"]
        )
        for concept_name, r in all_results:
            writer.writerow(
                [concept_name, r.image, r.predicted_class, r.p0, r.p_masked, r.delta_p, r.delta_logit,
                 r.random_delta_p_mean, r.random_delta_p_std, r.z_score, r.n_random_fallbacks]
            )
    with open(RESULTS_DIR / f"broden_faithfulness_summary{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_concept_summary[0].keys()))
        writer.writeheader()
        writer.writerows(per_concept_summary)
    print(f"Saved results/broden_faithfulness_records{suffix}.csv and results/broden_faithfulness_summary{suffix}.csv")


def run_volume(object_index, model) -> tuple[list, list]:
    print("\n\n########## VOLUME PROCEDURE ##########", flush=True)
    rng_py = random.Random(SEED)
    all_results = []
    for concept_name, label_number in TEST_CONCEPTS.items():
        candidates = object_index.get(label_number, [])
        print(f"\n=== [volume] {concept_name} (label {label_number}): {len(candidates)} candidate images ===", flush=True)
        sample = rng_py.sample(candidates, min(N_IMAGES_PER_CONCEPT, len(candidates)))
        in_scope_count = 0
        for i, record in enumerate(sample):
            result = process_one(record, label_number, "object", model, SEED + i)
            if result is None:
                continue
            all_results.append((concept_name, result))
            if result.predicted_class in IN_SCOPE_NATIVE_INDICES:
                in_scope_count += 1
            if (i + 1) % 50 == 0:
                n_so_far = [r for n, r in all_results if n == concept_name]
                print(f"  [{i + 1}/{len(sample)}] n={len(n_so_far)}, "
                      f"in-scope-prediction hits so far: {in_scope_count}", flush=True)
        print(f"[volume] {concept_name}: {in_scope_count} of {len([r for n, r in all_results if n == concept_name])} "
              f"landed on one of the 25 target classes.", flush=True)

    summary = summarize(all_results, "volume")
    save_csvs(all_results, summary, "_volume")
    return all_results, summary


def run_targeted(object_index, model, matching_classes: dict[str, set[int]], label: str, suffix: str, seed_offset: int) -> tuple[list, list]:
    print(f"\n\n########## TARGETED PROCEDURE ({label}) ##########", flush=True)
    rng_py = random.Random(SEED + seed_offset)  # distinct seed/order per procedure, genuinely independent draws
    all_results = []
    for concept_name, label_number in TEST_CONCEPTS.items():
        in_scope = matching_classes[concept_name]
        candidates = list(object_index.get(label_number, []))
        rng_py.shuffle(candidates)
        print(f"\n=== [{label}] {concept_name} (label {label_number}, matching classes {sorted(in_scope)}): "
              f"scanning up to {min(MAX_SCAN_PER_CONCEPT, len(candidates))} of {len(candidates)} candidates for "
              f"{N_TARGET_IN_SCOPE_PER_CONCEPT} in-scope hits ===", flush=True)

        hits = 0
        scanned = 0
        for record in candidates:
            if scanned >= MAX_SCAN_PER_CONCEPT or hits >= N_TARGET_IN_SCOPE_PER_CONCEPT:
                break
            scanned += 1
            mask = concept_pixel_mask(record, "object", label_number)
            if mask is None or not mask.any():
                continue
            image = Image.open(record.image).convert("RGB")
            if mask.shape != (image.height, image.width):
                continue
            pred = model.quick_predict(image)
            if pred not in in_scope:
                continue
            # Found an in-scope hit -- now run the real (masked + random-draw) measurement.
            result = process_one(record, label_number, "object", model, SEED + seed_offset + hits)
            if result is None:
                continue
            all_results.append((concept_name, result))
            hits += 1
            if hits % 5 == 0:
                print(f"  scanned={scanned}, in-scope hits found={hits}/{N_TARGET_IN_SCOPE_PER_CONCEPT}", flush=True)

        if hits < N_TARGET_IN_SCOPE_PER_CONCEPT:
            print(f"[{label}] {concept_name}: only found {hits}/{N_TARGET_IN_SCOPE_PER_CONCEPT} in-scope hits "
                  f"after scanning {scanned} candidates (exhausted budget, not the full candidate pool "
                  f"unless scanned == len(candidates)) -- reporting honestly, not padding.", flush=True)
        else:
            print(f"[{label}] {concept_name}: found all {hits} in-scope hits after scanning {scanned} candidates.", flush=True)

    summary = summarize(all_results, label)
    save_csvs(all_results, summary, suffix)
    return all_results, summary


def load_saved_summary(suffix: str) -> list[dict] | None:
    path = RESULTS_DIR / f"broden_faithfulness_summary{suffix}.csv"
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["n_images"] = int(row["n_images"])
        row["mean_concept_delta_p"] = float(row["mean_concept_delta_p"])
        row["paired_p"] = float(row["paired_p"])
    return rows


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    spec = BACKBONES["resnet18"]
    model = NativeModelAdapter(spec, DEVICE)

    print("Loading Broden index (63,305 records)...", flush=True)
    labels = load_label_table(NETDISSECT_ROOT)
    records = load_index(NETDISSECT_ROOT)
    print("Building object-category concept index (decodes each object mask once)...", flush=True)
    object_index = build_concept_index(records, "object")

    # v29's volume (_volume) and loose-targeted (_targeted, any-of-25) runs
    # are already saved and unchanged by this run -- only the new strict,
    # per-concept-namesake-only targeted procedure runs here.
    volume_summary = load_saved_summary("_volume")
    loose_targeted_summary = load_saved_summary("_targeted")
    if volume_summary is None or loose_targeted_summary is None:
        raise RuntimeError("expected results/broden_faithfulness_summary_{volume,targeted}.csv from the prior "
                            "v29 run to already exist -- run_volume/run_targeted (loose) haven't been re-run here.")

    _strict_results, strict_summary = run_targeted(
        object_index, model, CONCEPT_MATCHING_CLASSES, label="targeted_strict", suffix="_targeted_strict", seed_offset=1999
    )

    print("\n\n########## COMPARISON: volume vs. loose-targeted (any of 25) vs. strict-targeted (own namesake class only) ##########")
    vol_by_concept = {row["concept"]: row for row in volume_summary}
    loose_by_concept = {row["concept"]: row for row in loose_targeted_summary}
    strict_by_concept = {row["concept"]: row for row in strict_summary}
    for concept_name in TEST_CONCEPTS:
        v = vol_by_concept[concept_name]
        lo = loose_by_concept[concept_name]
        st = strict_by_concept[concept_name]
        print(
            f"{concept_name:>8s}  "
            f"volume: n={v['n_images']:<4d} delta_p={v['mean_concept_delta_p']:.4f} p={v['paired_p']:.4g}   |   "
            f"loose: n={lo['n_images']:<4d} delta_p={lo['mean_concept_delta_p']:.4f} p={lo['paired_p']:.4g}   |   "
            f"strict: n={st['n_images']:<4d} delta_p={st['mean_concept_delta_p']:.4f} p={st['paired_p']:.4g}"
        )


if __name__ == "__main__":
    main()
