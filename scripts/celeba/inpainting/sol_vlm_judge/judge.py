# ruff: noqa: SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""VLM-as-judge diagnostic for the inpainting ground truths (diagnostic only; never used to select fills).
Judge: Qwen2.5-VL-7B-Instruct (own ViT, not SigLIP/CLIP-text based like CARDS). Score = logit of P(yes) vs P(no) for
the next token after a yes/no question (no free-form generation), summed over "yes"/"Yes" and "no"/"No" variants.

Parts (--part):
  val     judge validation: 1,000 CelebAMask-HQ images OUTSIDE the GT set (seed 0) x 26 concept questions
  retain  every GT pair: original, and for each run the concept fill (m0) and first random-shape fill (m1), asked the
          pair's concept question
  real    realism question for each GT original and each run's m0 and m1
All images go through the same 512 px LANCZOS + JPEG q95 round trip as the saved fills (fills are already in it),
then are shown at 448x448. Output: one CSV per (part, shard): part, run, hq_idx, concept, k, score.
Resumable: rows already in the shard's CSV are skipped."""
import argparse
import csv
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
RUNS = ["lama_s0", "sd2_quality_s0", "sdxl_quality_s0", "flux_none_s0", "sd2_concept_s0", "sdxl_concept_s0"]
Q = {
    "Arched_Eyebrows": "Does this person have arched eyebrows?",
    "Bags_Under_Eyes": "Does this person have bags under the eyes?",
    "Bald": "Is this person bald?",
    "Bangs": "Does this person have bangs?",
    "Big_Lips": "Does this person have big lips?",
    "Big_Nose": "Does this person have a big nose?",
    "Black_Hair": "Does this person have black hair?",
    "Blond_Hair": "Does this person have blond hair?",
    "Brown_Hair": "Does this person have brown hair?",
    "Bushy_Eyebrows": "Does this person have bushy eyebrows?",
    "Eyeglasses": "Is this person wearing eyeglasses?",
    "Gray_Hair": "Does this person have gray hair?",
    "Mouth_Slightly_Open": "Is this person's mouth slightly open?",
    "Narrow_Eyes": "Does this person have narrow eyes?",
    "Pale_Skin": "Does this person have pale skin?",
    "Pointy_Nose": "Does this person have a pointy nose?",
    "Receding_Hairline": "Does this person have a receding hairline?",
    "Rosy_Cheeks": "Does this person have rosy cheeks?",
    "Smiling": "Is this person smiling?",
    "Straight_Hair": "Does this person have straight hair?",
    "Wavy_Hair": "Does this person have wavy hair?",
    "Wearing_Earrings": "Is this person wearing earrings?",
    "Wearing_Hat": "Is this person wearing a hat?",
    "Wearing_Lipstick": "Is this person wearing lipstick?",
    "Wearing_Necklace": "Is this person wearing a necklace?",
    "Wearing_Necktie": "Is this person wearing a necktie?",
}
REAL_Q = "Has this photo been digitally edited, or does it contain visible artifacts or unnatural regions?"
SUFFIX = " Answer with yes or no."
SIDE, BATCH, N_VAL = 448, 16, 1000


def roundtrip(im):
    b = io.BytesIO(); im.convert("RGB").resize((512, 512), Image.LANCZOS).save(b, "JPEG", quality=95); b.seek(0)
    return Image.open(b).convert("RGB")


def tasks(part, a):
    """Yields (key, image_loader, question); key = (part, run, hq_idx, concept, k)."""
    man = [(int(r["hq_idx"]), r["concept_name"], int(r["n_masks"])) for r in csv.DictReader(open(a.bundle / "manifest.csv"))]
    hq = lambda i: (lambda: roundtrip(Image.open(a.hq / f"{i}.jpg")))
    fill = lambda run, i, c, k: (lambda: Image.open(a.fills / run / f"{i}_{c}_m{k}.jpg").convert("RGB"))
    if part == "val":
        gt = {i for i, _, _ in man}
        pool = sorted(int(p.stem) for p in a.hq.glob("*.jpg") if int(p.stem) not in gt)
        for i in sorted(np.random.default_rng(0).choice(pool, N_VAL, replace=False).tolist()):
            for c, q in Q.items(): yield ("val", "orig", i, c, -1), hq(i), q
    elif part == "retain":
        for i, c, nm in man:
            yield ("retain", "orig", i, c, -1), hq(i), Q[c]
            for run in RUNS:
                for k in range(min(2, nm)): yield ("retain", run, i, c, k), fill(run, i, c, k), Q[c]
    else:
        for i in sorted({i for i, _, _ in man}): yield ("real", "orig", i, "", -1), hq(i), REAL_Q
        for i, c, nm in man:
            for run in RUNS:
                for k in range(min(2, nm)): yield ("real", run, i, c, k), fill(run, i, c, k), REAL_Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["val", "retain", "real"], required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--hq", type=Path, required=True, help="CelebAMask-HQ/CelebA-HQ-img")
    ap.add_argument("--fills", type=Path, required=True, help="folder with one subfolder per run")
    ap.add_argument("--bundle", type=Path, required=True, help="sol_inpaint folder (manifest.csv)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"{a.part}_{a.shard:02d}.csv"
    done = set()
    if path.exists():
        done = {(r["part"], r["run"], int(r["hq_idx"]), r["concept"], int(r["k"])) for r in csv.DictReader(open(path))}
    todo = [t for n, t in enumerate(tasks(a.part, a)) if n % a.nshards == a.shard and t[0] not in done]
    if a.limit: todo = todo[: a.limit]
    print(f"part {a.part} shard {a.shard}/{a.nshards}: {len(todo)} questions to run ({len(done)} done)", flush=True)
    if not todo: return

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=SIDE * SIDE, max_pixels=SIDE * SIDE)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    tok = proc.tokenizer
    ids = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})
    YES, NO = ids(["yes", "Yes", " yes", " Yes"]), ids(["no", "No", " no", " No"])
    print("yes ids", YES, "no ids", NO, flush=True)

    new = not path.exists()
    f = open(path, "a", newline=""); w = csv.writer(f)
    if new: w.writerow(["part", "run", "hq_idx", "concept", "k", "score"])
    for s in range(0, len(todo), BATCH):
        chunk = todo[s:s + BATCH]
        ims = [ld().resize((SIDE, SIDE), Image.BICUBIC) for _, ld, _ in chunk]
        texts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q + SUFFIX}]}],
                                          add_generation_prompt=True) for _, _, q in chunk]
        x = proc(text=texts, images=ims, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            lg = model(**x).logits[:, -1].float().log_softmax(-1)
        sc = (torch.logsumexp(lg[:, YES], -1) - torch.logsumexp(lg[:, NO], -1)).cpu().numpy()
        for (key, _, _), v in zip(chunk, sc): w.writerow([*key, f"{v:.4f}"])
        f.flush()
        if (s // BATCH) % 50 == 0: print(f"  {s + len(chunk)}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
