"""Generates per-token log-probabilities for the captioning-bias track's 5
"modern VLM" ground-truth models (DBAC paper Table 9's own 13-model list,
WACV 2026 -- confirmed directly against the paper's Section 5 model list and
per-model A->T/T->A scores; the other 8 "classic" models (FC, Att2In,
UpDown, Oscar, NIC+Equalizer, SAT, NIC, Transformer) only exist locally as
precomputed caption pickles on standard COCO val images -- no runnable
checkpoint for any of them exists in this workspace, deferred per direct
discussion):

    vit_gpt2 (nlpconnect/vit-gpt2-image-captioning)
    blip     (Salesforce/blip-image-captioning-large)
    florence (microsoft/Florence-2-base)
    llava    (llava-hf/llava-1.5-7b-hf)
    bakllava (llava-hf/bakLlava-v1-hf)

Two image sets, handled DIFFERENTLY -- corrected per direct user feedback
("We should not need to generate the original captions again as they are
available in the .pkl files in DPAC [DIC]"):

  - "original": all 10,780 images in `captioning_bias_concept_sets.csv`.
    These 5 models' own captions on these exact images ALREADY EXIST at
    `DIC/data/new_models/no_masking/<model_type>.pkl` (a list of
    {"img_id": <int>, "pred": <str>} dicts -- confirmed directly, all 5
    files have exactly 10,780 entries whose `img_id` matches
    `captioning_bias_concept_sets.csv`'s own `image_id` column 1:1, zero
    missing/extra either direction). What's genuinely missing is only the
    LOG-PROBS (neither `DIC/utils/caption_gen.py` nor `DPAC/utils/
    captioner.py`, the scripts that produced these pickles, capture them).
    So for "original", this script does NOT regenerate captions -- it
    TEACHER-FORCES the existing caption text through each model (tokenized
    under that model's own tokenizer) and records the log-prob the model
    assigns to each of those exact tokens. `caption` in the output CSV is
    the reused pickle text, unchanged.
  - "masked": the 6,353 (img_name, concept_name) rows in
    `captioning_bias_masked_images_manifest.csv` (masked_path column, built
    by `build_captioning_masked_images.py`). No captions exist for these
    anywhere -- free generation via `generate()`, same as before.

The two paths use genuinely DIFFERENT mechanics, not a shared code path --
"masked" (`caption()`) uses free `generate(..., output_scores=True,
return_dict_in_generate=True)`, where `scores` IS the model's real,
unconstrained per-step distribution. "original" (`score_existing()`) was
FIRST attempted via the same `generate()` machinery constrained through
`prefix_allowed_tokens_fn` (forcing the given tokens one at a time while
still reading `output_scores`) -- caught as WRONG by a direct sanity check
(every recorded log-prob came back exactly 0.0). Root cause: HF's `scores`
are captured AFTER all LogitsProcessors run, including the one
`prefix_allowed_tokens_fn` installs, which sets every non-forced token's
logit to -inf -- so it was measuring "the one allowed token has probability
1 after masking everything else," not the model's real confidence. Fixed
by scoring "original" with a single, unconstrained forward pass per
architecture instead (`labels=` for the 3 seq2seq-style models --
vit_gpt2, blip's exception aside, florence -- and a manual causal shift for
llava/bakllava's prompt+target sequence) -- each convention verified
empirically (see conversation) against real images before being used at
scale, not assumed from documentation alone.

For "masked", `scores` is one (1, vocab) logit tensor per newly generated
step for every architecture here, never including prompt-token logits, so
it always lines up with the last `len(scores)` tokens of `sequences`
regardless of whether `sequences` also contains the prompt (true for
llava/bakllava, false for the encoder-decoder vit_gpt2/blip). No beam
search anywhere (Florence's original reference used `num_beams=3`;
switched to greedy here so log-prob attribution doesn't need beam
re-ranking/`beam_indices` bookkeeping -- a deliberate, documented deviation
from the reference script) -- every model uses `do_sample=False,
num_beams=1`.

One (model_type, image_set) pair per invocation -- matches this project's
own one-job-per-unit HPC convention (e.g. FTW's per-concept split), so 5
models x 2 image sets = 10 independent, parallelizable SLURM jobs.
Resumable: on start, reads any existing output CSV and skips (img_name,
concept_name) pairs already written, flushing one row at a time so a
killed job loses at most the in-flight image.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Both env-overridable (FTW's own `_ROOT` convention, notes/ftw_correlation_
# investigation.md) -- the hardcoded defaults are this machine's own local
# paths and will NOT resolve on any other machine, HPC included. On a new
# host, set CAPTIONING_COCO_ROOT to wherever COCO2014/val2014 lives there,
# and CAPTIONING_EXISTING_CAPTIONS_DIR to wherever the 5 DIC no_masking
# pickles (vit_gpt2/blip/florence/llava/bakllava.pkl) were copied -- that
# directory is from a SEPARATE sibling repo (DIC), not part of CARDS at
# all, so it must be transferred there explicitly, it won't come along
# with a CARDS checkout.
COCO_ROOT = Path(os.environ.get("CAPTIONING_COCO_ROOT", r"C:\Users\btokas\Projects\Datasets\COCO2014\val2014"))
RESULTS_DIR = Path("results")
CONCEPT_SETS_CSV = RESULTS_DIR / "captioning_bias_concept_sets.csv"
MASKED_MANIFEST_CSV = RESULTS_DIR / "captioning_bias_masked_images_manifest.csv"
OUT_DIR = RESULTS_DIR / "captioning_bias_captions"
EXISTING_CAPTIONS_DIR = Path(os.environ.get(
    "CAPTIONING_EXISTING_CAPTIONS_DIR", r"C:\Users\btokas\Projects\DIC\data\new_models\no_masking"))

MODEL_NAMES = {
    "vit_gpt2": "nlpconnect/vit-gpt2-image-captioning",
    "blip": "Salesforce/blip-image-captioning-large",
    "florence": "microsoft/Florence-2-base",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "bakllava": "llava-hf/bakLlava-v1-hf",
}

# Neutral prompt (prompt_type=2 in both reference scripts' own convention --
# their own default), used verbatim for llava/bakllava.
LLAVA_PROMPT = "USER:\n Describe what is going on in image? <image>\nASSISTANT:"
FLORENCE_PROMPT = "<MORE_DETAILED_CAPTION>"

FIELDNAMES = ["img_name", "concept_name", "image_type", "model_type", "caption", "caption_source",
              "n_tokens", "sum_logprob", "mean_logprob", "token_ids_json", "token_logprobs_json"]


def load_original_items() -> list[tuple[str, str, Path, int]]:
    """(img_name, concept_name="", path, image_id) for all 10,780 concept_sets.csv images."""
    items = []
    with open(CONCEPT_SETS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            items.append((row["img_name"], "", COCO_ROOT / row["img_name"], int(row["image_id"])))
    return items


def load_masked_items() -> list[tuple[str, str, Path, int]]:
    """(img_name, concept_name, path, image_id=-1) for all 6,353 masked-manifest rows.

    `masked_path` was written by `build_captioning_masked_images.py` on
    Windows (`str(masked_path)`), which bakes in backslash separators --
    fine on Windows, but on Linux/HPC backslash is just an ordinary
    filename character, not a separator, so `Path(row["masked_path"])`
    there resolves to a single nonexistent file literally named with
    backslashes in it. Confirmed directly as the root cause of EVERY
    "masked" image_set job failing on HPC while "original" jobs (whose
    path is built fresh from COCO_ROOT + a bare filename, never read
    from this CSV) succeeded. Normalizing backslash -> forward-slash
    here is a no-op on Windows and fixes it on Linux -- forward slash is
    a valid separator on both.
    """
    items = []
    with open(MASKED_MANIFEST_CSV, newline="") as f:
        for row in csv.DictReader(f):
            items.append((row["img_name"], row["concept_name"], Path(row["masked_path"].replace("\\", "/")), -1))
    return items


def load_existing_captions(model_type: str) -> dict[int, str]:
    """img_id -> caption text, from DIC's own `caption_gen.py` output. Confirmed
    directly: all 5 modern-VLM pickles here have exactly 10,780 entries whose
    img_id matches captioning_bias_concept_sets.csv's image_id column 1:1."""
    path = EXISTING_CAPTIONS_DIR / f"{model_type}.pkl"
    with open(path, "rb") as f:
        entries = pickle.load(f)
    return {e["img_id"]: e["pred"] for e in entries}


def token_logprobs_from_generate_output(gen_out) -> tuple[list[int], list[float]]:
    """Per-generated-token (token_id, log_prob) pairs. `gen_out.scores` has
    exactly one (1, vocab) tensor per newly generated/forced step for every
    architecture used here -- never includes prompt-token logits -- so it
    always lines up with `sequences[0, -len(scores):]`, regardless of
    whether `sequences` also contains the prompt (true for llava/bakllava,
    false for the encoder-decoder vit_gpt2/blip)."""
    n_generated = len(gen_out.scores)
    if n_generated == 0:
        return [], []
    gen_token_ids = gen_out.sequences[0, -n_generated:].tolist()
    logprobs = []
    for step_logits, token_id in zip(gen_out.scores, gen_token_ids):
        log_probs_step = torch.log_softmax(step_logits[0].float(), dim=-1)
        logprobs.append(float(log_probs_step[token_id]))
    return gen_token_ids, logprobs


# Teacher-forcing was FIRST attempted via `generate(prefix_allowed_tokens_fn=...)`
# (forcing generate() to emit given tokens one at a time while still reading
# `output_scores`) -- caught as WRONG by a direct sanity check (every
# recorded log-prob came back exactly 0.0, i.e. probability 1.0 for every
# token, which is essentially never true). Root cause, confirmed against
# the HF docs: `output_scores`/`scores` in `generate()` are captured AFTER
# all LogitsProcessors run, INCLUDING the PrefixConstrainedLogitsProcessor
# that `prefix_allowed_tokens_fn` installs -- which sets every non-forced
# token's logit to -inf. So `scores` was measuring "the one allowed token
# has probability 1 after masking everything else," not the model's real
# confidence. Replaced below with a direct single forward pass per
# architecture (`labels=`/manual causal shift) -- verified empirically for
# all 4 architectures (see conversation) to produce varied, plausible
# negative log-probs instead of the degenerate 0.0.


class ModelRunner:
    """Loads one model_type and exposes `caption(image)` (free generation)
    and `score_existing(image, text)` (teacher-forced log-probs for given
    text), both returning (text, token_ids, logprobs)."""

    def __init__(self, model_type: str, max_len: int):
        self.model_type = model_type
        self.max_len = max_len
        model_name = MODEL_NAMES[model_type]

        if model_type == "vit_gpt2":
            from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(DEVICE).eval()
            self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        elif model_type == "blip":
            from transformers import BlipForConditionalGeneration, BlipProcessor
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=DTYPE).to(DEVICE).eval()

        elif model_type == "florence":
            from transformers import AutoModelForCausalLM, AutoProcessor
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=DTYPE, trust_remote_code=True).to(DEVICE).eval()
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        elif model_type in ("llava", "bakllava"):
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=DTYPE).to(DEVICE).eval()
            self.processor = AutoProcessor.from_pretrained(model_name)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _tokenizer(self):
        return self.tokenizer if self.model_type == "vit_gpt2" else self.processor.tokenizer

    def _build_generate_inputs(self, image: Image.Image) -> dict:
        """Architecture-specific kwargs for `model.generate(**kwargs)`, shared
        between free generation and teacher-forced scoring."""
        mt = self.model_type
        if mt == "vit_gpt2":
            pixel_values = self.feature_extractor(images=[image], return_tensors="pt").pixel_values.to(DEVICE)
            return {"pixel_values": pixel_values}
        elif mt == "blip":
            inputs = self.processor(images=image, return_tensors="pt").to(DEVICE, DTYPE)
            return dict(inputs)
        elif mt == "florence":
            inputs = self.processor(text=FLORENCE_PROMPT, images=image, return_tensors="pt").to(DEVICE, DTYPE)
            return {"input_ids": inputs["input_ids"], "pixel_values": inputs["pixel_values"]}
        elif mt in ("llava", "bakllava"):
            inputs = self.processor(image, LLAVA_PROMPT, return_tensors="pt").to(DEVICE, DTYPE)
            return dict(inputs)
        else:
            raise ValueError(f"Unknown model_type: {mt}")

    def _extract_text(self, gen_out) -> str:
        mt = self.model_type
        if mt == "vit_gpt2":
            return self.tokenizer.decode(gen_out.sequences[0], skip_special_tokens=True).strip()
        elif mt == "blip":
            return self.processor.decode(gen_out.sequences[0], skip_special_tokens=True).strip()
        elif mt == "florence":
            generated_text = self.processor.batch_decode(gen_out.sequences, skip_special_tokens=False)[0]
            parsed = self.processor.post_process_generation(
                generated_text, task=FLORENCE_PROMPT, image_size=(self._last_image_size))
            return dict(parsed)[FLORENCE_PROMPT]
        elif mt in ("llava", "bakllava"):
            full_text = self.processor.decode(gen_out.sequences[0], skip_special_tokens=True)
            return full_text.split("ASSISTANT:")[-1].strip()
        else:
            raise ValueError(f"Unknown model_type: {mt}")

    @torch.no_grad()
    def caption(self, image: Image.Image) -> tuple[str, list[int], list[float]]:
        """Free generation -- the model's own greedy caption. `scores` here
        IS the model's real, unconstrained per-step distribution (no
        `prefix_allowed_tokens_fn`/logits processor narrows anything), so
        `token_logprobs_from_generate_output` is valid for this path."""
        self._last_image_size = (image.width, image.height)  # florence's post_process_generation needs it
        gen_inputs = self._build_generate_inputs(image)
        gen_out = self.model.generate(
            **gen_inputs, max_new_tokens=self.max_len, do_sample=False, num_beams=1,
            output_scores=True, return_dict_in_generate=True)
        text = self._extract_text(gen_out)
        token_ids, logprobs = token_logprobs_from_generate_output(gen_out)
        return text, token_ids, logprobs

    @torch.no_grad()
    def score_existing(self, image: Image.Image, existing_text: str) -> tuple[str, list[int], list[float]]:
        """Teacher-forced log-probs for `existing_text` via a single forward
        pass (see the module docstring for why NOT `generate()` with a
        forcing constraint) -- tokenized under this model's own tokenizer.
        Returns `existing_text` unchanged as the `caption` field (NOT
        re-decoded from the forced tokens -- retokenizing an already-
        finalized string is not guaranteed to round-trip to byte-identical
        ids for every BPE/WordPiece tokenizer, confirmed directly on a rare
        BLIP-invented word ("arafed"); the vast majority of common words DO
        round-trip, and re-decoding could silently diverge from the text
        that was actually scored)."""
        mt = self.model_type
        target_token_ids = self._tokenizer()(existing_text, add_special_tokens=False).input_ids
        if len(target_token_ids) == 0:
            return existing_text, [], []
        target = torch.tensor([target_token_ids], device=DEVICE)

        if mt == "vit_gpt2":
            pixel_values = self.feature_extractor(images=[image], return_tensors="pt").pixel_values.to(DEVICE)
            # `labels=` auto-constructs decoder_input_ids = shift_right(labels)
            # internally (VisionEncoderDecoderModel's own seq2seq convention) --
            # logits[0, i] predicts target[i] directly, verified empirically.
            logits = self.model(pixel_values=pixel_values, labels=target).logits

        elif mt == "blip":
            inputs = self.processor(images=image, return_tensors="pt").to(DEVICE, DTYPE)
            # Blip's text decoder needs an explicit input_ids (no auto-shift
            # from `labels=` -- confirmed directly, raises otherwise). Its own
            # `generate()` (no input_ids passed) starts from `text_config.
            # bos_token_id` internally (confirmed: matches gen_out.sequences[0,0]
            # in a free-generation probe) -- reconstructed manually here as
            # [bos] + target, so logits[0, 0:n] predict target[0:n].
            bos = self.model.config.text_config.bos_token_id
            input_ids = torch.tensor([[bos] + target_token_ids], device=DEVICE)
            attn_mask = torch.ones_like(input_ids)
            full_logits = self.model(pixel_values=inputs["pixel_values"], input_ids=input_ids,
                                      attention_mask=attn_mask).logits
            logits = full_logits[:, :len(target_token_ids), :]

        elif mt == "florence":
            inputs = self.processor(text=FLORENCE_PROMPT, images=image, return_tensors="pt").to(DEVICE, DTYPE)
            # Same auto-shift-from-labels convention as vit_gpt2 (Florence-2 is
            # also seq2seq internally, `Florence2ForConditionalGeneration`
            # despite being loaded via AutoModelForCausalLM) -- verified empirically.
            logits = self.model(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                 labels=target).logits

        elif mt in ("llava", "bakllava"):
            inputs = self.processor(image, LLAVA_PROMPT, return_tensors="pt").to(DEVICE, DTYPE)
            prompt_len = inputs["input_ids"].shape[-1]
            full_ids = torch.cat([inputs["input_ids"], target], dim=1)
            full_attn = torch.ones_like(full_ids)
            full_logits = self.model(input_ids=full_ids, attention_mask=full_attn,
                                      pixel_values=inputs["pixel_values"]).logits
            # Decoder-only causal shift: logits at position i predict token i+1,
            # so the prompt's LAST position (index prompt_len-1) predicts the
            # first target token -- verified empirically.
            logits = full_logits[:, prompt_len - 1: prompt_len - 1 + len(target_token_ids), :]

        else:
            raise ValueError(f"Unknown model_type: {mt}")

        logprobs = [float(torch.log_softmax(logits[0, i].float(), dim=-1)[tid])
                    for i, tid in enumerate(target_token_ids)]
        return existing_text, target_token_ids, logprobs


def load_done_keys(out_path: Path) -> set[tuple[str, str]]:
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["img_name"], row["concept_name"]))
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=sorted(MODEL_NAMES))
    parser.add_argument("--image_set", required=True, choices=["original", "masked"])
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None, help="cap on number of images, for smoke testing")
    args = parser.parse_args()

    items = load_original_items() if args.image_set == "original" else load_masked_items()
    if args.limit is not None:
        items = items[:args.limit]

    existing_captions = load_existing_captions(args.model_type) if args.image_set == "original" else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.model_type}__{args.image_set}.csv"
    done_keys = load_done_keys(out_path)
    remaining = [item for item in items if (item[0], item[1]) not in done_keys]
    print(f"{args.model_type}/{args.image_set}: {len(items)} total, {len(done_keys)} already done, "
          f"{len(remaining)} remaining -> {out_path}", flush=True)

    if not remaining:
        return

    print(f"Loading {args.model_type} on {DEVICE}...", flush=True)
    runner = ModelRunner(args.model_type, args.max_len)

    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        n_failed = 0
        for i, (img_name, concept_name, image_path, image_id) in enumerate(remaining, 1):
            if i % 50 == 0 or i == 1:
                print(f"\r[{args.model_type}/{args.image_set}] {i}/{len(remaining)} "
                      f"(failed={n_failed})", end="", flush=True)
            if not image_path.exists():
                n_failed += 1
                # Previously silent -- gave zero signal beyond the final
                # summary count when every image failed to resolve (e.g.
                # backslash-separator paths breaking on Linux while
                # working fine on Windows, caught directly on HPC). Print
                # the first few in full, then rate-limit so a systematic
                # path bug is obvious in .out immediately rather than only
                # inferable after the whole job finishes.
                if n_failed <= 3 or n_failed % 500 == 0:
                    print(f"\n[{img_name}/{concept_name}] image not found: {image_path}", flush=True)
                continue
            try:
                image = Image.open(image_path).convert("RGB")
                if existing_captions is not None:
                    text, token_ids, logprobs = runner.score_existing(image, existing_captions[image_id])
                    caption_source = "reused_dic_pkl"
                else:
                    text, token_ids, logprobs = runner.caption(image)
                    caption_source = "generated"
            except Exception as e:  # noqa: BLE001 -- one bad image must not kill an hours-long job
                n_failed += 1
                print(f"\n[{img_name}/{concept_name}] failed: {e}", flush=True)
                continue

            writer.writerow({
                "img_name": img_name, "concept_name": concept_name, "image_type": args.image_set,
                "model_type": args.model_type, "caption": text, "caption_source": caption_source,
                "n_tokens": len(token_ids),
                "sum_logprob": sum(logprobs) if logprobs else "",
                "mean_logprob": (sum(logprobs) / len(logprobs)) if logprobs else "",
                "token_ids_json": json.dumps(token_ids), "token_logprobs_json": json.dumps(logprobs),
            })
            f.flush()

        print(f"\nDone. {len(remaining) - n_failed}/{len(remaining)} succeeded, {n_failed} failed.", flush=True)


if __name__ == "__main__":
    main()
