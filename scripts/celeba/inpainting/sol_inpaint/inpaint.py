# ruff: noqa: C408, SIM115
# One-shot research script: bare open() on small CSVs and dict() row literals are intentional.

"""Diffusion-inpainting fills for the CelebAMask-HQ faithfulness ground truth (companion to the local LaMa run).

For every (image, concept) pair in manifest.csv, fills mask k=0 (the concept region, dilated 8 px) and masks k>=1
(equal-area random shapes avoiding it) with one inpainter + one seed. The inpainter's output is upsampled to the
original resolution and pasted back ONLY inside the mask (the VAE round trip would otherwise change every pixel), then
saved like the LaMa fills: 512 px LANCZOS, JPEG q95, as OUT/{model}_{neg}_s{seed}/{hq_idx}_{concept}_m{k}.jpg.

Pre-set choices (fixed before seeing any result):
  - prompt: neutral, "a high-quality photo of a person's face" (removal, not counterfactual editing)
  - --neg quality (primary):  concept-independent negative prompt QUALITY_NEG (fewer artifacts / less OOD)
    --neg concept (sensitivity only): QUALITY_NEG + the concept's own name; pushes AWAY from the concept, so it is closer
      to a counterfactual edit and makes the GT depend on the inpainter's text encoder - never the primary GT
    --neg none: no negative prompt.  FLUX.1 Fill has no negative prompt (guidance-distilled): only --neg none.
  - sd2:  stable-diffusion-2-inpainting (sd2-community mirror; the stabilityai repo is gone), 512 px, 30 steps
  - sdxl: diffusers/stable-diffusion-xl-1.0-inpainting-0.1, 1024 px, strength 0.99 (model card), 30 steps
  - flux: black-forest-labs/FLUX.1-Fill-dev (gated: accept the licence + HF token), 1024 px, guidance 30 (model card),
          30 steps, bf16
  - at a working resolution below the image's, the mask is max-pooled (any covered pixel stays masked), as for LaMa
  - per-fill generator seed = crc32(f"{model}|{neg}|{seed}|{hq_idx}|{concept}|{k}") so any single fill is reproducible
Resumable: pairs whose files all exist are skipped. Sharding: --shard i --nshards n takes pairs[i::n]."""
import argparse
import csv
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODELS = {  # repo, working resolution, extra call kwargs, batch (masks per forward pass), supports negative prompt
    "sd2": ("sd2-community/stable-diffusion-2-inpainting", 512, {}, 6, True),
    "sdxl": ("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 1024, {"strength": 0.99}, 6, True),
    "flux": ("black-forest-labs/FLUX.1-Fill-dev", 1024, {"guidance_scale": 30.0, "max_sequence_length": 512}, 2, False),
}
PROMPT = "a high-quality photo of a person's face"
QUALITY_NEG = "blurry, distorted, deformed, disfigured, unrealistic, artifacts, low quality"
SAVE_RES = 512


def load_pipe(name):
    repo = MODELS[name][0]
    if name == "flux":
        from diffusers import FluxFillPipeline
        pipe = FluxFillPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    else:
        from diffusers import AutoPipelineForInpainting
        kw = dict(torch_dtype=torch.float16)
        if name == "sdxl": kw["variant"] = "fp16"
        pipe = AutoPipelineForInpainting.from_pretrained(repo, **kw)
        if hasattr(pipe, "safety_checker"): pipe.safety_checker = None
    pipe = pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--neg", choices=["quality", "concept", "none"], default="quality")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--img-root", type=Path, required=True, help="CelebAMask-HQ/CelebA-HQ-img")
    ap.add_argument("--bundle", type=Path, required=True, help="folder with manifest.csv and masks/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N pairs of this shard")
    a = ap.parse_args()
    _, R, extra, BATCH, has_neg = MODELS[a.model]
    if a.neg != "none" and not has_neg: raise SystemExit(f"{a.model} has no negative prompt; use --neg none")

    pairs = [(int(r["hq_idx"]), r["concept_name"], int(r["n_masks"])) for r in csv.DictReader(open(a.bundle / "manifest.csv"))]
    pairs = pairs[a.shard::a.nshards]
    if a.limit: pairs = pairs[: a.limit]
    out = a.out / f"{a.model}_{a.neg}_s{a.seed}"; out.mkdir(parents=True, exist_ok=True)
    name = lambda i, c, k: out / f"{i}_{c}_m{k}.jpg"
    todo = [p for p in pairs if not all(name(p[0], p[1], k).exists() for k in range(p[2]))]
    print(f"{a.model} neg={a.neg} seed {a.seed} shard {a.shard}/{a.nshards}: {len(pairs)} pairs, {len(todo)} to run", flush=True)
    if not todo: return
    pipe = load_pipe(a.model)
    t0 = time.time()
    for n, (i, c, nm) in enumerate(todo):
        image = Image.open(a.img_root / f"{i}.jpg").convert("RGB"); W, H = image.size
        x = np.asarray(image, dtype=np.float32)
        masks = [np.array(Image.open(a.bundle / "masks" / f"{i}_{c}_m{k}.png")).astype(bool) for k in range(nm)]
        assert all(m.shape == (H, W) for m in masks), (i, c)
        small = image.resize((R, R), Image.BICUBIC) if (W, H) != (R, R) else image
        msmall = [Image.fromarray(m.astype(np.uint8) * 255).resize((R, R), Image.BOX).point(lambda v: 255 if v > 0 else 0)
                  for m in masks]
        kw = dict(extra)
        if a.neg != "none":
            kw["negative_prompt"] = QUALITY_NEG + ("" if a.neg == "quality" else ", " + c.replace("_", " ").lower())
        ys = []
        for s in range(0, nm, BATCH):
            b = range(s, min(nm, s + BATCH))
            gens = [torch.Generator("cuda").manual_seed(zlib.crc32(f"{a.model}|{a.neg}|{a.seed}|{i}|{c}|{k}".encode()) & 0x7FFFFFFF)
                    for k in b]
            bk = dict(kw)
            if "negative_prompt" in bk: bk["negative_prompt"] = [bk["negative_prompt"]] * len(b)
            with torch.inference_mode():
                ys += pipe(prompt=[PROMPT] * len(b), image=[small] * len(b), mask_image=[msmall[k] for k in b], height=R, width=R,
                           num_inference_steps=a.steps, generator=gens, **bk).images
        for k, (y, m) in enumerate(zip(ys, masks)):
            y = np.asarray(y.resize((W, H), Image.BICUBIC) if y.size != (W, H) else y, dtype=np.float32)
            comp = np.where(m[..., None], y, x).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(comp).resize((SAVE_RES, SAVE_RES), Image.LANCZOS).save(name(i, c, k), quality=95)
        if (n + 1) % 25 == 0 or n == 0:
            el = time.time() - t0
            print(f"  {n + 1}/{len(todo)}  {el / (n + 1):.1f} s/pair  eta {el / (n + 1) * (len(todo) - n - 1) / 60:.0f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
