"""Computes per-concept Gender-attribution scores for the captioning-bias
track from the already-generated caption+log-prob CSVs
(`results/captioning_bias_captions/<model>__{original,masked}.csv`),
for all 10 models: the 5 modern-VLM models this project generated itself
(`scripts/captioning/build/generate_captions_with_logprobs.py`, integer
`token_ids_json`, decoded via each model's own HF tokenizer) PLUS 5 more
sourced externally by a coauthor (gpt4o, fc, att2in, transformer, updown
-- `scripts/captioning/build/convert_external_captions.py`, TEXT
`token_ids_json` already, no tokenizer needed/available for these).

Design (per two existing docstrings already in this codebase, not a new
invention):
  - `cards.data.captioning_bias`'s own docstring: "score a caption by the
    model's own probability on these tokens; 0 when a caption contains
    none of them" -- MASCULINE_WORDS/FEMININE_WORDS are the target tokens.
  - `build_captioning_masked_images.py`'s own docstring: "the actual
    attribution score (mean caption-based gender-word delta between
    original and masked captions) gets computed downstream once captions
    come back" -- this script IS that downstream step.

Per-caption gender score: decode each token in `token_ids_json`
INDIVIDUALLY (memoized per model, not per row -- only a few thousand
distinct ids ever appear), normalize (lowercase, strip whitespace/
punctuation), exact-match against MASCULINE_WORDS/FEMININE_WORDS. For
matched tokens, convert log-prob -> probability (exp) and sum separately
for masculine vs. feminine matches:
    gender_score = sum(exp(logprob) for masculine-word tokens)
                 - sum(exp(logprob) for feminine-word tokens)
(positive = masculine-leaning, negative = feminine-leaning, 0 = no gender
word present at all -- exactly the "0 when none of them" case).

CAVEAT, measured empirically (not assumed) and reported per model below:
decoding tokens individually only catches gender words that are a SINGLE
token in that model's own vocabulary. Confirmed directly: 28/28 for
vit_gpt2/blip/florence, 24/28 for llava (misses waiter/aunt/princess/
waitress), 26/28 for bakllava (misses waiter/waitress). The short, common
words that dominate real caption frequency (he/him/his/she/her/man/boy/
woman/girl) are single-token everywhere tested.

Per-(image, concept) attribution: delta = gender_score(original caption)
- gender_score(masked caption) -- mirrors CARDS' own delta_p = p0 -
p_masked convention exactly. Per-concept attribution: mean of that delta
across every image masked for that concept.

Race is NOT scored here -- cards.data.captioning_bias's own docstring:
no equivalent race word list exists anywhere in the DBAC/DIC codebase,
since race is essentially never stated explicitly in captions.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cards.data.captioning_bias import FEMININE_WORDS, MASCULINE_WORDS

RESULTS_DIR = Path("results")
CAPTIONS_DIR = RESULTS_DIR / "captioning_bias_captions"
OUT_DIR = RESULTS_DIR / "captioning_bias_attribution"

MODEL_NAMES = {
    "vit_gpt2": "nlpconnect/vit-gpt2-image-captioning",
    "blip": "Salesforce/blip-image-captioning-large",
    "florence": "microsoft/Florence-2-base",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "bakllava": "llava-hf/bakLlava-v1-hf",
}
# Sourced externally (convert_external_captions.py) -- token_ids_json is
# already TEXT for these, no HF checkpoint exists/is-needed to decode
# through (FC/Att2in/Transformer/UpDown aren't HF models at all; gpt4o's
# real tokenizer isn't public).
EXTERNAL_MODEL_TYPES = ["gpt4o", "fc", "att2in", "transformer", "updown"]
ALL_MODEL_TYPES = list(MODEL_NAMES) + EXTERNAL_MODEL_TYPES

MASCULINE_SET = set(MASCULINE_WORDS)
FEMININE_SET = set(FEMININE_WORDS)
_WORD_STRIP_RE = re.compile(r"[^a-z]+")


def load_tokenizer(model_type: str):
    """None for externally-sourced models -- their token_ids_json is
    already text, see EXTERNAL_MODEL_TYPES's own comment."""
    if model_type in EXTERNAL_MODEL_TYPES:
        return None
    elif model_type == "vit_gpt2":
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(MODEL_NAMES[model_type])
    elif model_type == "blip":
        from transformers import BlipProcessor
        return BlipProcessor.from_pretrained(MODEL_NAMES[model_type]).tokenizer
    elif model_type == "florence":
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(MODEL_NAMES[model_type], trust_remote_code=True).tokenizer
    elif model_type in ("llava", "bakllava"):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(MODEL_NAMES[model_type]).tokenizer
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def make_token_classifier(tokenizer):
    """-> memoized token (id OR already-text) -> "masculine" | "feminine" |
    None. Only a few thousand distinct tokens ever appear across a
    model's captions, so a plain dict memo (not a full-vocab precompute)
    is already fast. `tokenizer=None` means `token` is already text
    (externally-sourced models) -- used as-is, no decode step."""
    cache: dict[int | str, str | None] = {}

    def classify(token: int | str) -> str | None:
        if token not in cache:
            raw = token if tokenizer is None else tokenizer.decode([token])
            text = _WORD_STRIP_RE.sub("", raw.strip().lower())
            if text in MASCULINE_SET:
                cache[token] = "masculine"
            elif text in FEMININE_SET:
                cache[token] = "feminine"
            else:
                cache[token] = None
        return cache[token]

    return classify


def gender_score(token_ids: list[int], token_logprobs: list[float], classify) -> float:
    masc_prob = fem_prob = 0.0
    for tid, lp in zip(token_ids, token_logprobs):
        cls = classify(tid)
        if cls == "masculine":
            masc_prob += math.exp(lp)
        elif cls == "feminine":
            fem_prob += math.exp(lp)
    return masc_prob - fem_prob


def load_rows(model_type: str, image_set: str) -> list[dict]:
    path = CAPTIONS_DIR / f"{model_type}__{image_set}.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_json_list(cell: str) -> list:
    try:
        return json.loads(cell)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(cell)  # some HPC-written CSVs may use Python repr, not strict JSON


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_type in ALL_MODEL_TYPES:
        print(f"\n=== {model_type} ===", flush=True)
        print("  loading tokenizer..." if model_type not in EXTERNAL_MODEL_TYPES
              else "  no tokenizer needed (token_ids_json is already text)...", flush=True)
        tokenizer = load_tokenizer(model_type)
        classify = make_token_classifier(tokenizer)

        original_rows = load_rows(model_type, "original")
        masked_rows = load_rows(model_type, "masked")
        print(f"  {len(original_rows)} original rows, {len(masked_rows)} masked rows", flush=True)

        print("  scoring original captions...", flush=True)
        original_gender_score: dict[str, float] = {}
        for i, row in enumerate(original_rows, 1):
            if i % 2000 == 0:
                print(f"    {i}/{len(original_rows)}", flush=True)
            token_ids = parse_json_list(row["token_ids_json"])
            token_logprobs = parse_json_list(row["token_logprobs_json"])
            original_gender_score[row["img_name"]] = gender_score(token_ids, token_logprobs, classify)

        print("  scoring masked captions + computing deltas...", flush=True)
        deltas_by_concept: dict[str, list[float]] = defaultdict(list)
        n_missing_original = 0
        for i, row in enumerate(masked_rows, 1):
            if i % 2000 == 0:
                print(f"    {i}/{len(masked_rows)}", flush=True)
            img_name, concept_name = row["img_name"], row["concept_name"]
            if img_name not in original_gender_score:
                n_missing_original += 1
                continue
            token_ids = parse_json_list(row["token_ids_json"])
            token_logprobs = parse_json_list(row["token_logprobs_json"])
            masked_score = gender_score(token_ids, token_logprobs, classify)
            delta = original_gender_score[img_name] - masked_score
            deltas_by_concept[concept_name].append(delta)
        if n_missing_original:
            print(f"  WARNING: {n_missing_original} masked rows had no matching original caption", flush=True)

        out_path = OUT_DIR / f"{model_type}_gender_attribution.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["concept_name", "n_images", "mean_gender_delta"])
            for concept_name in sorted(deltas_by_concept):
                deltas = deltas_by_concept[concept_name]
                writer.writerow([concept_name, len(deltas), sum(deltas) / len(deltas)])
        print(f"  Saved {len(deltas_by_concept)} concepts to {out_path}", flush=True)

        ranked = sorted(deltas_by_concept.items(), key=lambda kv: -abs(sum(kv[1]) / len(kv[1])))[:5]
        print("  top-5 |mean_gender_delta|: " +
              ", ".join(f"{c}={sum(v) / len(v):+.4f}" for c, v in ranked), flush=True)


if __name__ == "__main__":
    main()
