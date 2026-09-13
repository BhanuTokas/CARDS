"""Computes the ACTUAL Wang & Russakovsky (2021) Directional Bias
Amplification T->A metric (`biasamp_task_to_attribute`, reused directly
from the reference implementation at C:\\Users\\btokas\\Projects\\
directional-bias-amp\\directional_biasamp.py -- NOT reimplemented here)
per model, and compares it against this track's own masking-based
mean_gender_delta T->A numbers. Direct request: "calculate something
like DBA for the T->A direction and compare it with our attributions."

The two metrics measure genuinely different things, not two estimates of
the same quantity:
  - Our mean_gender_delta (score_captioning_bias_attribution.py): a
    CAUSAL, same-image masking intervention -- mask concept c, does the
    caption's gender-word probability change. No ground-truth
    co-occurrence baseline involved at all.
  - DBA's y_AT * delta_AT (this script): a STATISTICAL comparison of the
    MODEL'S predicted gender/task co-occurrence against the REAL
    ground-truth co-occurrence in the dataset, signed so that a positive
    value means the model's predictions push the already-existing
    real-world stereotype (whichever direction it runs) FURTHER, not
    just that a stereotype exists. Requires no masking at all -- built
    entirely from (a) real object-presence labels, (b) real gender
    labels, (c) a model-DERIVED gender prediction per image.

Attribute prediction (attribute_preds), built fresh here since DBA needs
one: p_male_pred(caption) = masc_prob_sum / (masc_prob_sum + fem_prob_sum)
if that denominator is nonzero, else 0.5 (neutral/uninformative -- no
gender word present at all, matching gender_score's own "0 when none of
them" convention translated into probability space). masc_prob_sum/
fem_prob_sum are the SAME per-caption quantities score_captioning_bias_
attribution.py already computes (sum of exp(logprob) over matched
gender-word tokens) -- reused here via the identical token-classification
logic, not recomputed differently.

Per-concept `values = y_at * delta_at` IS returned by the reference
function only when very verbose (it only PRINTS top-10s with `names=`,
doesn't return the full matrix) -- the per-concept loop below is a
faithful, side-by-side reimplementation of the reference function's own
y_at/delta_at math (not a different formula), cross-checked directly
against the reference function's own returned scalar
(mean(per-concept values) must equal biasamp_task_to_attribute's own
return value) before trusting it.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, r"C:\Users\btokas\Projects\directional-bias-amp")

from directional_biasamp import biasamp_task_to_attribute

from cards.data.captioning_bias import (
    FEMININE_WORDS,
    MASCULINE_WORDS,
    groundable_concepts,
    load_bias_captioning_records,
)

RESULTS_DIR = Path("results")
CAPTIONS_DIR = RESULTS_DIR / "captioning_bias_captions"
ATTRIBUTION_DIR = RESULTS_DIR / "captioning_bias_attribution"
MODEL_TYPES = ["vit_gpt2", "blip", "florence", "llava", "bakllava", "gpt4o", "fc", "att2in", "transformer", "updown"]

MASCULINE_SET = set(MASCULINE_WORDS)
FEMININE_SET = set(FEMININE_WORDS)
_WORD_STRIP_RE = re.compile(r"[^a-z]+")


def make_token_classifier():
    """Text-token classifier (all 10 models' token_ids_json is either
    already text -- the 5 externally-sourced models -- or, for the 5 this
    project generated itself, an int that ALSO decodes correctly via a
    plain str() fallback is NOT attempted here -- see note in main():
    this script reads token_ids_json for both int- and str-typed sources
    and only needs to compare against MASCULINE_SET/FEMININE_SET, which
    are themselves plain words, so a str-typed token compares directly
    and an int-typed one is handled by the caller passing pre-decoded text
    (see gender_probs_for_row)."""
    cache: dict[str, str | None] = {}

    def classify(text: str) -> str | None:
        if text not in cache:
            norm = _WORD_STRIP_RE.sub("", text.strip().lower())
            if norm in MASCULINE_SET:
                cache[text] = "masculine"
            elif norm in FEMININE_SET:
                cache[text] = "feminine"
            else:
                cache[text] = None
        return cache[text]

    return classify


def load_rows(model_type: str, image_set: str) -> list[dict]:
    path = CAPTIONS_DIR / f"{model_type}__{image_set}.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def p_male_pred_for_row(row: dict, classify, tokenizer) -> float:
    tokens = json.loads(row["token_ids_json"])
    logprobs = json.loads(row["token_logprobs_json"])
    masc_prob = fem_prob = 0.0
    for tok, lp in zip(tokens, logprobs):
        text = tok if isinstance(tok, str) else tokenizer.decode([tok])
        cls = classify(text)
        if cls == "masculine":
            masc_prob += math.exp(lp)
        elif cls == "feminine":
            fem_prob += math.exp(lp)
    denom = masc_prob + fem_prob
    return (masc_prob / denom) if denom > 0 else 0.5


def load_tokenizer_if_needed(model_type: str):
    external = {"gpt4o", "fc", "att2in", "transformer", "updown"}
    if model_type in external:
        return None
    if model_type == "vit_gpt2":
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    elif model_type == "blip":
        from transformers import BlipProcessor
        return BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large").tokenizer
    elif model_type == "florence":
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).tokenizer
    elif model_type in ("llava", "bakllava"):
        from transformers import AutoProcessor
        name = "llava-hf/llava-1.5-7b-hf" if model_type == "llava" else "llava-hf/bakLlava-v1-hf"
        return AutoProcessor.from_pretrained(name).tokenizer
    raise ValueError(model_type)


def main():
    print("Loading ground-truth records (gender + object-presence labels)...", flush=True)
    records = load_bias_captioning_records()
    concepts = groundable_concepts(records)
    concept_to_idx = {c: i for i, c in enumerate(concepts)}
    gender_by_img = {r["img_name"]: r["bb_gender"] for r in records}
    task_present_by_img: dict[str, set[str]] = {r["img_name"]: set(r["rmdup_object_list"]) & set(concepts) for r in records}

    our_t_to_a: dict[str, dict[str, float]] = {}
    for m in MODEL_TYPES:
        path = ATTRIBUTION_DIR / f"{m}_gender_attribution.csv"
        with open(path, newline="") as f:
            our_t_to_a[m] = {row["concept_name"]: float(row["mean_gender_delta"]) for row in csv.DictReader(f)}

    summary_rows = []
    per_concept_rows = []
    for model_type in MODEL_TYPES:
        print(f"\n=== {model_type} ===", flush=True)
        classify = make_token_classifier()
        tokenizer = load_tokenizer_if_needed(model_type)

        original_rows = load_rows(model_type, "original")
        img_names, p_male_preds = [], []
        for i, row in enumerate(original_rows, 1):
            if i % 3000 == 0:
                print(f"  {i}/{len(original_rows)}", flush=True)
            gender = gender_by_img.get(row["img_name"])
            if gender not in ("Male", "Female"):
                continue
            img_names.append(row["img_name"])
            p_male_preds.append(p_male_pred_for_row(row, classify, tokenizer))

        n = len(img_names)
        task_labels = np.zeros((n, len(concepts)), dtype=int)
        attribute_labels = np.zeros((n, 2), dtype=int)  # [Male, Female]
        attribute_preds = np.zeros((n, 2), dtype=float)
        for i, img_name in enumerate(img_names):
            for c in task_present_by_img[img_name]:
                task_labels[i, concept_to_idx[c]] = 1
            gender = gender_by_img[img_name]
            attribute_labels[i, 0 if gender == "Male" else 1] = 1
            attribute_preds[i, 0] = p_male_preds[i]
            attribute_preds[i, 1] = 1.0 - p_male_preds[i]

        dba_scalar = biasamp_task_to_attribute(task_labels, attribute_labels, attribute_preds)
        print(f"  DBA T->A (reference implementation, n={n} images): {dba_scalar:+.4f}", flush=True)

        # Faithful side-by-side replication of the reference function's own
        # y_at/delta_at math, to also get the per-concept breakdown.
        num_t, num_a = task_labels.shape[1], attribute_labels.shape[1]
        num_train = n
        p_at = np.zeros((num_a, num_t))
        p_a_p_t = np.zeros((num_a, num_t))
        for a in range(num_a):
            for t in range(num_t):
                t_idx = np.where(task_labels[:, t] == 1)[0]
                a_idx = np.where(attribute_labels[:, a] == 1)[0]
                p_a_p_t[a][t] = (len(t_idx) / num_train) * (len(a_idx) / num_train)
                p_at[a][t] = len(set(t_idx) & set(a_idx)) / num_train
        y_at = np.sign(p_at - p_a_p_t)

        a_cond_t = np.zeros((num_a, num_t))
        ahat_cond_t = np.zeros((num_a, num_t))
        for a in range(num_a):
            for t in range(num_t):
                t_idx = np.where(task_labels[:, t] == 1)[0]
                a_cond_t[a][t] = np.mean(attribute_labels[:, a][t_idx])
                ahat_cond_t[a][t] = np.mean(attribute_preds[:, a][t_idx])
        delta_at = ahat_cond_t - a_cond_t
        values = y_at * delta_at

        assert abs(np.nanmean(values) - dba_scalar) < 1e-9, "local replication doesn't match reference function"

        for t, c in enumerate(concepts):
            dba_male = values[0, t]  # Male-column DBA value for this concept (Female column is its mirror)
            per_concept_rows.append((model_type, c, dba_male, our_t_to_a[model_type].get(c, "")))

        common = [c for c in concepts if c in our_t_to_a[model_type]]
        our_mean = sum(our_t_to_a[model_type][c] for c in common) / len(common)
        summary_rows.append((model_type, n, dba_scalar, our_mean))
        print(f"  our mean_gender_delta T->A (mean over {len(common)} concepts): {our_mean:+.4f}", flush=True)

    with open(ATTRIBUTION_DIR / "dba_t2a_per_concept.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_type", "concept_name", "dba_t2a", "our_mean_gender_delta"])
        writer.writerows(per_concept_rows)

    with open(ATTRIBUTION_DIR / "dba_t2a_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_type", "n_images", "dba_t2a", "our_mean_gender_delta_t2a"])
        writer.writerows(summary_rows)

    print("\n=== Summary: reference DBA T->A vs. our masking-based T->A ===")
    print(f"{'model':<12s} {'DBA T->A':>10s} {'our T->A':>10s}")
    for model_type, n, dba_scalar, our_mean in summary_rows:
        print(f"{model_type:<12s} {dba_scalar:>+10.4f} {our_mean:>+10.4f}")

    dba_vals = [r[2] for r in summary_rows]
    our_vals = [r[3] for r in summary_rows]
    corr = np.corrcoef(dba_vals, our_vals)[0, 1]
    print(f"\nPearson correlation between the two metrics' per-model overall scores (n=10): {corr:+.4f}")
    print("Saved per-concept comparison to results/captioning_bias_attribution/dba_t2a_per_concept.csv")
    print("Saved summary to results/captioning_bias_attribution/dba_t2a_summary.csv")


if __name__ == "__main__":
    main()
