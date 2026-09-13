"""Computes A->T and T->A attribution per concept AND per-model overall
(matching DBAC Table 9's own one-A->T/one-T->A-per-model layout), Gender
only (no Race word list exists, per cards.data.captioning_bias's own
docstring). Direct request: "given the gender and task words, can you
calculate the mean attribution of gender on task words and task on
gender words?"

T->A (task affects attribute -- ALREADY BUILT, reused verbatim here):
`score_captioning_bias_attribution.py`'s own `mean_gender_delta` per
concept -- masks the task concept in the image, measures the resulting
change in the caption's gender-word probability
(gender_score = masc_prob_sum - fem_prob_sum).

A->T (attribute affects task -- NEW): gender isn't a maskable image
region, so this direction uses no masking at all -- computed directly
from the ORIGINAL (unmasked) captions, split by the image's REAL
ground-truth gender label (`bb_gender`):
    task_word_score(caption, concept) = 1.0 if concept_name (optionally
        +"s") appears as a whole-word match in the caption text, else 0.0
    A_to_T[concept] = mean(task_word_score | Male images)
                     - mean(task_word_score | Female images)
(positive = concept mentioned MORE often when the person is male).

NOTE ON ASYMMETRY, stated directly rather than hidden: T->A is a
continuous, PROBABILITY-weighted score (sum of exp(logprob) over matched
gender-word tokens); A->T here is a simpler BINARY presence match on the
concept's name in the decoded caption text, not a probability-weighted
score. A token-probability-weighted A->T would need aligning a
(sometimes multi-word, e.g. "wine glass", "parking meter") concept
phrase to a token SPAN in the generated sequence, which -- unlike single
gender words, almost all single-token -- risks real misalignment for
multi-token phrases without extra work not done here. Presence-based
matching is simpler, directly verifiable, and avoids that risk; it's a
real methodological difference from T->A, not an oversight.

Per-model overall A->T/T->A = mean across all 79 concepts (unweighted,
matching DBAC's own single-number-per-model Table 9 layout) -- the
per-concept breakdown is also written out for the CSV.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cards.data.captioning_bias import groundable_concepts, load_bias_captioning_records

RESULTS_DIR = Path("results")
CAPTIONS_DIR = RESULTS_DIR / "captioning_bias_captions"
ATTRIBUTION_DIR = RESULTS_DIR / "captioning_bias_attribution"
MODEL_TYPES = ["vit_gpt2", "blip", "florence", "llava", "bakllava", "gpt4o", "fc", "att2in", "transformer", "updown"]


def load_rows(model_type: str, image_set: str) -> list[dict]:
    path = CAPTIONS_DIR / f"{model_type}__{image_set}.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_t_to_a(model_type: str) -> dict[str, float]:
    """concept_name -> mean_gender_delta, from the already-computed CSV."""
    path = ATTRIBUTION_DIR / f"{model_type}_gender_attribution.csv"
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["concept_name"]] = float(row["mean_gender_delta"])
    return out


def make_concept_matcher(concept_name: str):
    pattern = re.compile(r"\b" + re.escape(concept_name.lower()) + r"s?\b")
    return lambda caption_text: bool(pattern.search(caption_text.lower()))


def main():
    print("Loading ground-truth records (gender labels, concept bank)...", flush=True)
    records = load_bias_captioning_records()
    gender_by_img = {r["img_name"]: r["bb_gender"] for r in records}
    concepts = groundable_concepts(records)
    matchers = {c: make_concept_matcher(c) for c in concepts}

    overall_rows = []
    for model_type in MODEL_TYPES:
        print(f"\n=== {model_type} ===", flush=True)
        t_to_a = load_t_to_a(model_type)

        original_rows = load_rows(model_type, "original")
        print(f"  {len(original_rows)} original captions -- computing A->T presence scores...", flush=True)

        presence_by_concept: dict[str, dict[str, list[int]]] = {c: {"Male": [], "Female": []} for c in concepts}
        for i, row in enumerate(original_rows, 1):
            if i % 3000 == 0:
                print(f"    {i}/{len(original_rows)}", flush=True)
            gender = gender_by_img.get(row["img_name"])
            if gender not in ("Male", "Female"):
                continue
            caption_text = row["caption"]
            for c in concepts:
                presence_by_concept[c][gender].append(1 if matchers[c](caption_text) else 0)

        a_to_t: dict[str, float] = {}
        for c in concepts:
            male_vals, female_vals = presence_by_concept[c]["Male"], presence_by_concept[c]["Female"]
            if not male_vals or not female_vals:
                continue
            a_to_t[c] = (sum(male_vals) / len(male_vals)) - (sum(female_vals) / len(female_vals))

        out_path = ATTRIBUTION_DIR / f"{model_type}_a2t_t2a.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["concept_name", "A_to_T", "T_to_A"])
            for c in concepts:
                writer.writerow([c, a_to_t.get(c, ""), t_to_a.get(c, "")])
        print(f"  Saved per-concept A->T/T->A to {out_path}", flush=True)

        common = [c for c in concepts if c in a_to_t and c in t_to_a]
        mean_a_to_t = sum(a_to_t[c] for c in common) / len(common)
        mean_t_to_a = sum(t_to_a[c] for c in common) / len(common)
        print(f"  OVERALL (mean over {len(common)} concepts): A->T={mean_a_to_t:+.4f}  T->A={mean_t_to_a:+.4f}", flush=True)
        overall_rows.append((model_type, len(common), mean_a_to_t, mean_t_to_a))

    summary_path = ATTRIBUTION_DIR / "a2t_t2a_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_type", "n_concepts", "A_to_T", "T_to_A"])
        writer.writerows(overall_rows)
    print("\n=== Summary (one A->T/T->A per model, DBAC Table 9 layout) ===")
    print(f"{'model':<10s} {'A->T':>10s} {'T->A':>10s}")
    for model_type, n, a2t, t2a in overall_rows:
        print(f"{model_type:<10s} {a2t:>+10.4f} {t2a:>+10.4f}")
    print(f"Saved to {summary_path}")


if __name__ == "__main__":
    main()
