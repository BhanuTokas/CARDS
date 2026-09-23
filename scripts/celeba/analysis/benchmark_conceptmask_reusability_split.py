"""Fine-grained rerun of ConceptMask's full-scale per-concept phase,
splitting encoder-side (VLM) work from actual black-box-model calls,
prompted directly ("Can we reformat the table and writeup to reflect
this?" -- the "does the cost transfer to a new black-box model" axis
from the previous discussion). TCAV and PCBM already have a clean split
in the existing benchmark_computational_cost_full_scale.py run (TCAV:
100% black-box-dependent, since captum's CAV fitting IS a black-box
activation computation; PCBM: full_train_val_embed+concept_projection
are pure SigLIP encoding = reusable, native_scoring+elastic_net_fit
touch the native black-box = not reusable) -- only ConceptMask's
193.91s "scoring" phase was a single lump that needs re-measuring at
finer granularity.

Caches pool_embeds to disk so this doesn't repeat the ~105s embedding
pass from the original full-scale run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_celeba_full import CONCEPT_QUERY_TEXT
from run_cards_celeba_masking_hybrid_official_val_zscore import build_clean_official_val_paths

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.validation.broden_faithfulness import mask_region

RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
K = 50
N_TASKS = 2
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
ALPHA = 1.0
BATCH_SIZE = 128
EMBED_CACHE = RESULTS_DIR / ".official_val_pool_embeds_siglip_cache.pt"


def encode_images_batched(encoder, paths, label):
    chunks = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch = [Image.open(p).convert("RGB") for p in paths[start:start + BATCH_SIZE]]
        chunks.append(encoder.encode_images(batch))
        if (start // BATCH_SIZE) % 50 == 0:
            print(f"    [{label}] {start + len(batch)}/{len(paths)}", flush=True)
    return torch.cat(chunks, dim=0)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
    })
    encoder = instantiate_encoder(cfg)

    official_val_paths = build_clean_official_val_paths()
    print(f"loading {len(official_val_paths)} images from disk...", flush=True)
    pool_images = [Image.open(p).convert("RGB") for p in official_val_paths]

    if EMBED_CACHE.exists():
        print(f"loading cached pool embeddings from {EMBED_CACHE}...", flush=True)
        pool_embeds = torch.load(EMBED_CACHE).to(DEVICE)
    else:
        print("embedding pool (no cache found)...", flush=True)
        with torch.no_grad():
            pool_embeds = encode_images_batched(encoder, official_val_paths, "pool").to(DEVICE)
        torch.save(pool_embeds.cpu(), EMBED_CACHE)

    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)
    raw_queries = {c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], encoder), text_center)
                   for c in GROUNDABLE_CONCEPTS}
    queries = orthogonalize_queries(raw_queries)

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()

    @torch.no_grad()
    def black_box_scores(images):
        batch = torch.stack([spec.preprocess(im) for im in images]).to(DEVICE)
        return native_model(batch)[:, 1].detach().cpu()

    t_retrieval = t_localize = t_perturb_gen = t_perturb_embed = t_angle_select = t_black_box = 0.0

    print("\nrunning per-concept loop with fine-grained timers...", flush=True)
    t_start = time.perf_counter()
    for task_rep in range(N_TASKS):
        for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
            t_c = queries[concept_name].to(DEVICE)

            t0 = time.perf_counter()
            sims = (pool_embeds @ t_c).detach().cpu().numpy()
            present_indices = sims.argsort()[::-1][:K]
            t_retrieval += time.perf_counter() - t0

            sim_maps, present_images = [], []
            for idx in present_indices:
                img = pool_images[idx]
                t0 = time.perf_counter()
                sim_map = localize_concept(encoder, img, t_c, (img.height, img.width))
                t_localize += time.perf_counter() - t0
                sim_maps.append(sim_map)
                present_images.append(img)
            cutoff = concept_zscore_cutoff(sim_maps, ALPHA)

            for idx, img, sim_map in zip(present_indices, present_images, sim_maps):
                mask = threshold_mask(sim_map, method="fixed", cutoff=cutoff)
                if not mask.any() or mask.all():
                    continue
                mrng = np.random.default_rng(SEED + task_rep * 100_000 + concept_idx * 10_000 + int(idx))

                t0 = time.perf_counter()
                candidates = [mask_region(img, mask, strategy=s, rng=mrng) for s in FILL_STRATEGIES]
                t_perturb_gen += time.perf_counter() - t0

                t0 = time.perf_counter()
                with torch.no_grad():
                    cand_embeds = encoder.encode_images([img] + candidates).to(DEVICE)
                t_perturb_embed += time.perf_counter() - t0

                t0 = time.perf_counter()
                embed_orig = cand_embeds[0]
                best_angle, best_i = None, None
                for i in range(len(FILL_STRATEGIES)):
                    diff = embed_orig - cand_embeds[1 + i]
                    diff_unit = diff / diff.norm()
                    cos_sim = float(torch.clamp(diff_unit @ t_c, -1.0, 1.0))
                    angle = float(np.degrees(np.arccos(cos_sim)))
                    if best_angle is None or angle < best_angle:
                        best_angle, best_i = angle, i
                perturbed = candidates[best_i]
                t_angle_select += time.perf_counter() - t0

                t0 = time.perf_counter()
                _ = black_box_scores([img, perturbed])
                t_black_box += time.perf_counter() - t0
        print(f"  simulated task {task_rep + 1}/{N_TASKS} done, elapsed: {time.perf_counter() - t_start:.2f}s",
              flush=True)

    t_total = time.perf_counter() - t_start
    reusable = t_retrieval + t_localize + t_perturb_gen + t_perturb_embed + t_angle_select
    print(f"\n=== fine-grained breakdown, {len(GROUNDABLE_CONCEPTS)} concepts x {N_TASKS} tasks ===", flush=True)
    print(f"  retrieval (encoder-side, reusable):        {t_retrieval:8.2f}s", flush=True)
    print(f"  localization (encoder-side, reusable):     {t_localize:8.2f}s", flush=True)
    print(f"  perturbation generation (CPU, reusable):   {t_perturb_gen:8.2f}s", flush=True)
    print(f"  perturbation embedding (encoder-side, reusable): {t_perturb_embed:8.2f}s", flush=True)
    print(f"  angle/strategy selection (math, reusable): {t_angle_select:8.2f}s", flush=True)
    print(f"  --------------------------------------------------", flush=True)
    print(f"  TOTAL REUSABLE (encoder-side):              {reusable:8.2f}s", flush=True)
    print(f"  TOTAL BLACK-BOX-DEPENDENT (scoring calls):  {t_black_box:8.2f}s", flush=True)
    print(f"  sum check vs. wall total:                    {reusable + t_black_box:8.2f}s vs {t_total:8.2f}s",
          flush=True)


if __name__ == "__main__":
    main()
