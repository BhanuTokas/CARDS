"""Converts the coauthor's externally-generated captions+logprobs (GPT-4o
and 4 of DBAC's "classic" models -- FC, Att2in, Transformer, UpDown, via
ruotianluo/ImageCaptioning.pytorch, Table 9 of arXiv:2503.07878) into this
project's own `results/captioning_bias_captions/<model>__{original,masked}.csv`
schema, so every existing downstream script (score_captioning_bias_
attribution.py, score_captioning_bias_a2t_t2a.py) works on them unchanged.

Source layout (all 5 cover the SAME 6,352 pairs / 4,771 unique originals /
11,123 total captioned images as this project's own masked-image manifest):
  - results/4o_captioning_results/captions_with_logprobs.jsonl (+ its own
    captions_summary.csv for the pair_id -> (coco_image_id, concept_name)
    join -- needed because the classic-model JSONLs don't carry those
    fields directly, only `file_path`, whose basename IS the custom_id).
  - results/captioning_results_other_models/<model>/<model>_captions_
    with_logprobs.jsonl (+ its own _captions_summary.csv).

ONE schema difference from this project's own generate_captions_with_
logprobs.py output, called out explicitly rather than silently forced to
match: `token_ids_json` here holds TOKEN TEXT (JSON list of strings, e.g.
["a", "man", ...]), not integer vocab ids -- these sources give token
text directly, so there's no tokenizer to decode ids through (and
wouldn't even mean the same thing across an 9k-word FC vocab and gpt-4o's
own subword BPE vocab). Both downstream analysis scripts already handle
this (they special-case a str-typed token exactly like this).

GPT-4o refusals (22/6,352 pairs, see results/4o_captioning_results_
refusal_rerun/README.md): the 14 that got a real caption on re-run are
merged in HERE (overriding the refusal in the main file); the 8 that
refused again on retry are DROPPED entirely (no usable caption on either
attempt -- scoring refusal text as if it were a real caption would be
wrong, not just noisy). This drops 9 pairs total from gpt4o's own output
(8 masked-side-only + the 1 pair -- 550/bed -- refused on BOTH sides).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "captioning_bias_captions"

REFUSAL_PATTERNS = ["i'm sorry", "can't help", "cannot help", "i can't"]

FIELDNAMES = ["img_name", "concept_name", "image_type", "model_type", "caption", "caption_source",
              "n_tokens", "sum_logprob", "mean_logprob", "token_ids_json", "token_logprobs_json"]

SOURCES = {
    "gpt4o": {
        "jsonl": RESULTS_DIR / "4o_captioning_results" / "captions_with_logprobs.jsonl",
        "summary_csv": RESULTS_DIR / "4o_captioning_results" / "captions_summary.csv",
        "rerun_jsonl": RESULTS_DIR / "4o_captioning_results_refusal_rerun" / "captions_with_logprobs_RERUN.jsonl",
    },
    "fc": {
        "jsonl": RESULTS_DIR / "captioning_results_other_models" / "fc" / "fc_captions_with_logprobs.jsonl",
        "summary_csv": RESULTS_DIR / "captioning_results_other_models" / "fc" / "fc_captions_summary.csv",
    },
    "att2in": {
        "jsonl": RESULTS_DIR / "captioning_results_other_models" / "att2in" / "att2in_captions_with_logprobs.jsonl",
        "summary_csv": RESULTS_DIR / "captioning_results_other_models" / "att2in" / "att2in_captions_summary.csv",
    },
    "transformer": {
        "jsonl": RESULTS_DIR / "captioning_results_other_models" / "transformer" / "transformer_captions_with_logprobs.jsonl",
        "summary_csv": RESULTS_DIR / "captioning_results_other_models" / "transformer" / "transformer_captions_summary.csv",
    },
    "updown": {
        "jsonl": RESULTS_DIR / "captioning_results_other_models" / "updown" / "updown_captions_with_logprobs.jsonl",
        "summary_csv": RESULTS_DIR / "captioning_results_other_models" / "updown" / "updown_captions_summary.csv",
    },
}


def is_refusal(caption: str) -> bool:
    lower = caption.lower()
    return any(p in lower for p in REFUSAL_PATTERNS)


def load_pair_lookup(summary_csv: Path) -> dict[str, tuple[str, str]]:
    """pair_id (str) -> (coco_image_id, concept_name)."""
    lookup = {}
    with open(summary_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[row["pair_id"]] = (row["coco_image_id"], row["concept_name"])
    return lookup


def custom_id_of(record: dict) -> str:
    if "custom_id" in record:
        return record["custom_id"]
    return Path(record["file_path"]).stem  # e.g. "orig_496854" or "masked_2616"


def img_name_of(coco_image_id: str) -> str:
    return f"COCO_val2014_{int(coco_image_id):012d}.jpg"


def load_jsonl(path: Path) -> dict[str, dict]:
    """custom_id -> record."""
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records[custom_id_of(record)] = record
    return records


def convert_one_model(model_type: str, paths: dict):
    print(f"\n=== {model_type} ===", flush=True)
    pair_lookup = load_pair_lookup(paths["summary_csv"])
    records = load_jsonl(paths["jsonl"])
    print(f"  {len(records)} records, {len(pair_lookup)} pairs in summary", flush=True)

    n_dropped_refusal = 0
    if "rerun_jsonl" in paths and paths["rerun_jsonl"].exists():
        rerun_records = load_jsonl(paths["rerun_jsonl"])
        n_recovered = 0
        for custom_id, rerun_record in rerun_records.items():
            if not is_refusal(rerun_record["caption"]):
                records[custom_id] = rerun_record  # override the original refusal
                n_recovered += 1
        print(f"  merged {n_recovered}/{len(rerun_records)} successful reruns", flush=True)

    original_out: list[dict] = []
    masked_out: list[dict] = []
    for custom_id, record in records.items():
        caption = record["caption"]
        if is_refusal(caption):
            n_dropped_refusal += 1
            continue

        if custom_id.startswith("orig_"):
            image_type = "original"
            coco_image_id = record.get("coco_image_id", custom_id[len("orig_"):])
            concept_name = ""
        elif custom_id.startswith("masked_"):
            image_type = "masked"
            pair_id = record.get("pair_id", custom_id[len("masked_"):])
            coco_image_id, concept_name = pair_lookup[pair_id]
        else:
            raise ValueError(f"Unrecognized custom_id format: {custom_id!r}")

        tokens = record["tokens"]
        token_texts = [t["token"] for t in tokens]
        token_logprobs = [t["logprob"] for t in tokens]
        n_tokens = len(tokens)
        sum_logprob = sum(token_logprobs)

        row = {
            "img_name": img_name_of(coco_image_id), "concept_name": concept_name, "image_type": image_type,
            "model_type": model_type, "caption": caption, "caption_source": "generated",
            "n_tokens": n_tokens, "sum_logprob": sum_logprob,
            "mean_logprob": (sum_logprob / n_tokens) if n_tokens else "",
            "token_ids_json": json.dumps(token_texts), "token_logprobs_json": json.dumps(token_logprobs),
        }
        (original_out if image_type == "original" else masked_out).append(row)

    if n_dropped_refusal:
        print(f"  dropped {n_dropped_refusal} still-refused rows entirely (no usable caption)", flush=True)

    for image_type, rows in [("original", original_out), ("masked", masked_out)]:
        out_path = OUT_DIR / f"{model_type}__{image_type}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {image_type}: {len(rows)} rows -> {out_path}", flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model_type, paths in SOURCES.items():
        convert_one_model(model_type, paths)


if __name__ == "__main__":
    main()
